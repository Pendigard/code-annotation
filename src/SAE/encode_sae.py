
import argparse
from pathlib import Path
import pandas as pd
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import collect_code_files, get_language, load_sae, load_model, build_hook_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a code dataset with a sparse autoencoder and save the token-level dataframe.")
    parser.add_argument("--code-dir", type=str, default=str(PROJECT_ROOT / "data" / "code"), help="Directory containing code samples.")
    parser.add_argument("--output-path", type=str, default=str(PROJECT_ROOT / "outputs" / "sae_activations.parquet"), help="Output file for the dataframe.")
    parser.add_argument("--model-name", type=str, required=True, help="Transformer model name used to extract activations.")
    parser.add_argument("--sae-release", type=str, default=None, help="SAE release name for sae_lens.from_pretrained.")
    parser.add_argument("--sae-id", type=str, default=None, help="SAE id inside the release when using sae_lens.from_pretrained.")
    parser.add_argument("--sae-path", type=str, default=None, help="Local SAE checkpoint path for SAE.load_from_disk.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to extract from.")
    parser.add_argument("--hook-point", type=str, default="hook_resid_post", help="Hook point suffix inside the selected layer.")
    parser.add_argument("--method", type=str, choices=["transformer_lens", "direct"], default="transformer_lens", help="Activation extraction method.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of files to process together.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on the number of files to process.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on. Defaults to CUDA when available.")
    parser.add_argument("--dtype", type=str, default="float32", help="Torch dtype used for the model and SAE.")
    return parser.parse_args()


def encode_activations(sae: SAE, activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim == 3:
        batch, sequence_length, hidden_size = activations.shape
        flat_activations = activations.reshape(batch * sequence_length, hidden_size)
        encoded = sae.encode(flat_activations)
        return encoded.reshape(batch, sequence_length, -1)
    if activations.ndim == 2:
        return sae.encode(activations)
    raise ValueError(f"Unexpected activation shape: {tuple(activations.shape)}")


def extract_with_transformer_lens(model, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    _, cache = model.run_with_cache(tokens, names_filter=hook_name, prepend_bos=False)
    return cache[hook_name]


def extract_direct(model, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    captured = {}

    def hook_fn(activation, hook):
        captured[hook.name] = activation.detach()
        return activation

    model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)], prepend_bos=False)
    if hook_name not in captured:
        raise RuntimeError(f"No activation captured for hook '{hook_name}'.")
    return captured[hook_name]


def token_strings(model, tokens: torch.Tensor, file_uses_bos: bool = False) -> list[str]:
    return model.to_str_tokens(tokens, prepend_bos=file_uses_bos)


def process_file(file_path: Path, model, sae: SAE, hook_name: str, method: str) -> list[dict]:
    code = file_path.read_text(encoding="utf-8")
    tokens = model.to_tokens(code, prepend_bos=False)

    if method == "transformer_lens":
        activations = extract_with_transformer_lens(model, tokens, hook_name)
    else:
        activations = extract_direct(model, tokens, hook_name)

    sparse_activations = encode_activations(sae, activations)
    if sparse_activations.ndim == 2:
        sparse_activations = sparse_activations.unsqueeze(0)

    tokens = tokens[0]
    token_texts = model.to_str_tokens(tokens, prepend_bos=False)

    rows: list[dict] = []
    for token_index, (token_id, token_text, sae_vector) in enumerate(zip(tokens.tolist(), token_texts, sparse_activations[0])):
        positive_mask = sae_vector > 0
        positive_values = sae_vector[positive_mask]
        positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()

        if positive_values.numel() > 0:
            sorted_indices = torch.argsort(positive_values, descending=True)
            feature_ids = positive_indices[sorted_indices].tolist()
            feature_activations = positive_values[sorted_indices].tolist()
        else:
            feature_ids = []
            feature_activations = []

        rows.append(
            {
                "language": get_language(file_path),
                "path": str(file_path.resolve()),
                "token_index": token_index,
                "token_id": token_id,
                "token": token_text,
                "feature_ids": feature_ids,
                "feature_activations": feature_activations,
            }
        )

    return rows


def extract_multi_layers_transformer_lens(model, tokens: torch.Tensor, hook_names: list[str]) -> dict[str, torch.Tensor]:
    """
    Exécute la passe avant du modèle une seule fois et extrait les activations
    pour l'ensemble des couches demandées.
    """
    _, cache = model.run_with_cache(tokens, names_filter=hook_names, prepend_bos=False)
    return {hook: cache[hook] for hook in hook_names}


def process_file_multi_layers(
    file_path: Path, 
    model: HookedTransformer, 
    saes_dict: dict[int, SAE], 
    hook_point_suffix: str
) -> dict[int, list[dict]]:
    """
    Traite un fichier unique en extrayant les activations de toutes les couches 
    en un seul forward pass, puis applique le dictionnaire de SAE correspondants.
    
    Retourne un dictionnaire : { layer_index: [records_des_tokens] }
    """
    code = file_path.read_text(encoding="utf-8")
    tokens = model.to_tokens(code, prepend_bos=False)
    
    # Construction de tous les hooks à capturer
    layers = sorted(saes_dict.keys())
    hook_names = [build_hook_name(layer, hook_point_suffix) for layer in layers]
    hook_to_layer = {build_hook_name(layer, hook_point_suffix): layer for layer in layers}

    # Extraction en une seule passe globale
    multi_activations = extract_multi_layers_transformer_lens(model, tokens, hook_names)

    # Récupération des tokens au format string
    tokens_flat = tokens[0]
    token_texts = model.to_str_tokens(tokens_flat, prepend_bos=False)
    
    # Initialisation de la structure de retour
    layer_records = {layer: [] for layer in layers}

    # Encodage par couche
    for hook_name, activations in multi_activations.items():
        layer = hook_to_layer[hook_name]
        sae = saes_dict[layer]

        sparse_activations = encode_activations(sae, activations)
        if sparse_activations.ndim == 2:
            sparse_activations = sparse_activations.unsqueeze(0)

        # Extraction des caractéristiques actives pour cette couche précise
        for token_index, (token_id, token_text, sae_vector) in enumerate(zip(tokens_flat.tolist(), token_texts, sparse_activations[0])):
            positive_mask = sae_vector > 0
            positive_values = sae_vector[positive_mask]
            positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()

            if positive_values.numel() > 0:
                sorted_indices = torch.argsort(positive_values, descending=True)
                feature_ids = positive_indices[sorted_indices].tolist()
                feature_activations = positive_values[sorted_indices].tolist()
            else:
                feature_ids = []
                feature_activations = []

            layer_records[layer].append({
                "language": get_language(file_path),
                "path": str(file_path.resolve()),
                "token_index": token_index,
                "token_id": token_id,
                "token": token_text,
                "feature_ids": feature_ids,
                "feature_activations": feature_activations,
            })

    return layer_records


def encode_all_sae_layers(
    model: HookedTransformer,
    saes_dict: dict[int, SAE],
    files: list[Path],
    hook_point_suffix: str,
) -> dict[int, pd.DataFrame]:
    """
    Fonction dédiée demandée : Encode tous les fichiers sur toutes les couches
    spécifiées en minimisant les appels au modèle de langage.
    
    Retourne un dictionnaire { layer_index: DataFrame }
    """
    # Structure pour accumuler les lignes par couche
    accumulated_records = {layer: [] for layer in saes_dict.keys()}
    
    for file_path in tqdm(files, desc="Encoding all layers simultaneously", unit="file", leave=True):
        try:
            # Traitement d'un fichier sur toutes les couches d'un coup
            file_layer_records = process_file_multi_layers(file_path, model, saes_dict, hook_point_suffix)
            
            # Distribution des résultats dans les accumulateurs correspondants
            for layer, records in file_layer_records.items():
                accumulated_records[layer].extend(records)
                
            tqdm.write(f"{file_path.name}: encoded across all layers")
        except Exception as error:
            tqdm.write(f"{file_path.name}: failed ({error})")
            
    # Conversion finale en DataFrames Pandas
    return {layer: pd.DataFrame.from_records(records) for layer, records in accumulated_records.items()}

def encode_sae_from_model(
        model : HookedTransformer | AutoModelForCausalLM,
        sae : SAE,
        files : list[Path],
        hook_name : str,
        method : str,
):
    records: list[dict] = []
    for file_path in tqdm(files, desc="Encoding", unit="file", leave=False):
        try:
            records.extend(process_file(file_path, model, sae, hook_name, method))
            tqdm.write(f"{file_path.name}: encoded")
        except Exception as error:
            tqdm.write(f"{file_path.name}: failed ({error})")
    return pd.DataFrame.from_records(records)

def encode_sae_main(
        device: str | None,
        dtype: str,
        code_dir: str | Path,
        output_path: str | Path,
        model_name: str,
        sae_release: str | None,
        sae_id: str | None,
        sae_path: str | Path | None,
        layer: int,
        hook_point: str,
        method: str,
        limit: int = 0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, dtype)

    code_dir = Path(code_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_code_files(code_dir, limit)
    hook_name = build_hook_name(layer, hook_point)

    print(f"Loading model {model_name} on {device}")
    model = load_model(model_name, device=device, dtype=dtype)
    sae = load_sae(sae_release=sae_release, sae_id=sae_id, sae_path=sae_path, device=device, dtype=dtype)
    sae.eval()

    records: list[dict] = []
    for file_path in tqdm(files, desc="Encoding", unit="file", leave=False):
        try:
            records.extend(process_file(file_path, model, sae, hook_name, method))
            tqdm.write(f"{file_path.name}: encoded")
        except Exception as error:
            tqdm.write(f"{file_path.name}: failed ({error})")

    dataframe = pd.DataFrame.from_records(records)
    if output_path.suffix.lower() == ".csv":
        dataframe.to_csv(output_path, index=False)
    elif output_path.suffix.lower() in {".pkl", ".pickle"}:
        dataframe.to_pickle(output_path)
    else:
        dataframe.to_parquet(output_path, index=False)

    print(f"Saved dataframe with {len(dataframe)} rows to {output_path}")


def main() -> None:
    args = parse_arguments()
    encode_sae_main(
        device=args.device,
        dtype=args.dtype,
        code_dir=args.code_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        sae_path=args.sae_path,
        layer=args.layer,
        hook_point=args.hook_point,
        method=args.method,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()