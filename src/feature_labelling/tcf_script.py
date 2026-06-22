from tqdm.auto import tqdm
import os
from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoConfig

import pandas as pd

from src.feature_labelling.token_labels import build_token_label_dataframe
from src.feature_labelling.feature_pmi import *
from src.SAE.encode_sae import encode_all_sae_layers
import src.utils as ut


def compute_tcf(df: pd.DataFrame, selected_concepts: list[str], dict_results: dict[str, pd.DataFrame]) -> pd.DataFrame:

    C_sparse = build_sparse_token_concept_matrix(df, selected_concepts, "concepts")
    X_continuous, feature_ids_mapping = build_sparse_token_feature_matrix_continuous(df, "feature_ids", "feature_activations")
    metrics = compute_all_concept_feature_metrics(
        df,
        concepts=selected_concepts,
        X_boolean=(X_continuous > 0).astype(int),
        C_sparse=C_sparse,
        feature_ids_mapping=feature_ids_mapping,
        min_joint_count=3,
        min_feature_count=5,
        compute_auc=False,
    )

    df_pair = metrics[(metrics['concept'] != 'ERROR') & (metrics['pmi'] > 1.5) & (metrics['count_joint'] > 10)][["concept", "feature_id", "pmi", "count_joint", "f1"]].copy()

    df_pair['rank'] = df_pair.groupby('concept')['f1'].rank(method='first', ascending=False)

    candidates_df = df_pair[df_pair['rank'] <= 20].copy()
    tcf_df = compute_tcf_metrics(
        df_metrics=candidates_df,
        df_blocks_dict=dict_results,
        X_continuous=X_continuous,
        C_sparse=C_sparse,
        concepts_list=selected_concepts,
        feature_ids_mapping=feature_ids_mapping,
        n_bins=20
    )
    return tcf_df, df_pair

def make_output_folder(base_path: str | Path
                    , model_folder: str
                    , annotation_name: str
                    , sae_folder: str):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    os.makedirs(base_path / Path("label"), exist_ok=True)
    os.makedirs(base_path / Path("output") / model_folder / sae_folder / Path(annotation_name), exist_ok=True)
    os.makedirs(base_path / Path("output") / model_folder / sae_folder / Path("ast_purity"), exist_ok=True)
    os.makedirs(base_path / Path("SAE") / model_folder / sae_folder, exist_ok=True)
    os.makedirs(base_path / Path("logs"), exist_ok=True)


def read_annotations(annotations_path: str | Path, tokenizer : AutoTokenizer, method: str | None = None) -> pd.DataFrame:
    return build_token_label_dataframe(annotations_path, tokenizer, filter_space_tokens=False, annotation_type=method)


def main():
    ap = argparse.ArgumentParser(description="Encode code files with SAE and compute TCF metrics")
    ap.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda' or 'cpu'). Defaults to CUDA if available.")
    ap.add_argument("--dtype", type=str, default="float32", help="Torch dtype for model and SAE (e.g., 'float32' or 'float16').")
    ap.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res-canonical", help="SAE release name for sae_lens.from_pretrained.")
    ap.add_argument("--sae-id", type=str, default="layer_{}/width_16k/canonical", help="SAE id inside the release when using sae_lens.from_pretrained. Use '{}' as a placeholder for the layer index.")
    ap.add_argument("--sae-path", type=str, default=None, help="Local SAE checkpoint path for SAE.load_from_disk. If provided, this will override --sae-release and --sae-id.")
    ap.add_argument("--model-name", type=str, default="google/gemma-2-2b", help="Transformer model name used to extract activations.")
    ap.add_argument("--hook-point", type=str, default="hook_resid_post", help="Hook point suffix inside the selected layer.")
    ap.add_argument("--code-dir", type=str, default="/work03/celian/gemma-2-2b-code", help="Directory containing code samples.")
    ap.add_argument("--annotations-path", type=str, default="outputs/ast_labels.parquet", help="Path to the parquet file containing AST labels.")
    ap.add_argument("--ast-path", type=str, default="outputs/ast_labels.parquet", help="Path to the parquet file containing AST labels.")
    ap.add_argument("--annotation-type", type=str, choices=["joern", "llm", "none"], default="none", help="Type of annotation format to process.")
    ap.add_argument("--num-act-threshold", type=int, default=10, help="Minimum number of activations for a concept to be considered.")
    ap.add_argument("--output-dir", type=str, default="/work03/celian/code_feature_labelling", help="Directory to save output files.")
    args = ap.parse_args()
    """
    python -m src.feature_labelling.tcf_script \
        --device cuda \
        --dtype float32 \
        --sae-release gemma-scope-2b-pt-res-canonical \
        --sae-id layer_{}/width_16k/canonical \
        --model-name google/gemma-2-2b \
        --hook-point hook_resid_post \
        --code-dir /work03/celian/code-samples \
        --annotations-path outputs/joern_annotations.jsonl \
        --ast-path outputs/ast_labels.parquet \
        --annotation-type joern \
        --num-act-threshold 10 \
        --output-dir /work03/celian/code_feature_labelling
    """
    annotation_file = Path(args.annotations_path.rsplit("/", 1)[1])
    annotation_name = annotation_file.stem

    model_folder = Path(ut.slugify(args.model_name.rsplit("/", 1)[1]))
    sae_folder = Path(ut.slugify(args.sae_release) + "_" + ut.slugify(args.sae_id.format("")))

    make_output_folder(args.output_dir, model_folder, annotation_name, sae_folder)
    tqdm.write(f"Setting up logger. Logs will be saved to {Path(args.output_dir) / Path('logs')}")
    logger = ut.setup_logger(Path(args.output_dir) / Path("logs"))
    logger.info(f"Starting TCF computation with model {args.model_name} and SAE {args.sae_release} ({args.sae_id})")

    files = ut.collect_code_files(Path(args.code_dir))

    logger.info(f"Loading AST labels from {args.ast_path}")
    df_ast = pd.read_parquet(args.ast_path)

    def _ast_type(concepts):
        if isinstance(concepts, (list, set, tuple, np.ndarray)) and len(concepts) > 0:
            return concepts[-1]
        return None

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    df_ast['ast_type'] = df_ast['concepts'].apply(_ast_type)
    logger.info(f"AST labels loaded with {len(df_ast)} entries")

    logger.info(f"Loading annotations from {args.annotations_path} with method {args.annotation_type}")
    if annotation_file.suffix == "parquet":
        # Copy the parquet file to the output directory for consistency
        output_annotation_path = Path(args.output_dir) / Path("label") / annotation_file
        if not output_annotation_path.exists():
            df_label = pd.read_parquet(args.annotations_path)
            df_label.to_parquet(output_annotation_path)
        else:
            df_label = pd.read_parquet(output_annotation_path)
    else:
        # If the annotation file is not a parquet file, read it and save as parquet
        df_label = read_annotations(args.annotations_path, tokenizer, method=args.annotation_type)
        df_label.to_parquet(Path(args.output_dir) / Path("label") / annotation_file.with_suffix(".parquet"))
    logger.info(f"Annotations loaded with {len(df_label)} entries")

    selected_concepts = df_label["concepts"].explode().value_counts()
    selected_concepts = selected_concepts[selected_concepts > args.num_act_threshold].index.tolist()
    label_mask = df_label["concepts"].apply(lambda concepts: any(concept in selected_concepts for concept in concepts))
    df_label = df_label[label_mask].reset_index(drop=True)

    logger.info(f"Fetching metadata for {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name)
    
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layers", None)
    if n_layers == 0:
        raise ValueError(f"Impossible de déterminer le nombre de couches pour {args.model_name}")
    
    logger.info(f"Model metadata resolved: {n_layers} layers found.")


    # Définition du sous-dossier de stockage des parquets SAE
    sae_cache_dir = Path(args.output_dir) / "SAE" / model_folder / sae_folder
    os.makedirs(sae_cache_dir, exist_ok=True)

    all_layers_dfs = {}
    missing_layers = []
    ref_df = None

    # 1. Analyse du cache disque couche par couche
    for layer in range(n_layers):
        sae_output_file = sae_cache_dir / f"sae_layer_{layer}.parquet"
        
        if sae_output_file.exists():
            all_layers_dfs[layer] = pd.read_parquet(sae_output_file)
            logger.info(f"[Cache] Loaded layer {layer}/{n_layers - 1} from {sae_output_file.name}")
            if ref_df is None:
                ref_df = all_layers_dfs[layer]
                logger.info(f"Reference DataFrame for consistency checks set to layer {layer}")
        else:
            missing_layers.append(layer)

    # 2. Calcul incrémental uniquement pour les couches manquantes
    if len(missing_layers) > 0:
        logger.info(f"Missing layers detected: {missing_layers}. Loading weights and starting recovery...")
        logger.info(f"Loading model {args.model_name} on device {args.device} with dtype {args.dtype}")
        model = ut.load_model(model_name=args.model_name, device=args.device, dtype=args.dtype)
        model.eval()
        logger.info(f"Model loaded with {n_layers} layers.")
        
        # Chargement sélectif des dictionnaires de poids SAE manquants
        saes_dict = {}
        for layer in missing_layers:
            saes_dict[layer] = ut.load_sae(
                sae_id=args.sae_id.format(layer),
                sae_release=args.sae_release,
                device=args.device,
                dtype=args.dtype,
                sae_path=None
            )
            saes_dict[layer].eval()

        # Encodage simultané en une seule passe des couches manquantes
        logger.info(f"Encoding activations for {len(missing_layers)} layers simultaneously...")
        computed_layers_dfs = encode_all_sae_layers(model, saes_dict, files, args.hook_point)
        
        # Sauvegarde sur disque et fusion dans le dictionnaire principal
        for layer, df_sae in computed_layers_dfs.items():
            if df_sae.empty:
                logger.warning(f"SAE DataFrame for layer {layer} is empty. Skipping save.")
                continue
            sae_output_file = sae_cache_dir / f"sae_layer_{layer}.parquet"
            if ref_df is not None:
                # As the SAE encoding script skip files that cause OOM, we need to ensure consistency by filtering the new DataFrame to only include paths present in the reference DataFrame
                mask = df_sae["path"].isin(ref_df["path"].unique()) 
                df_sae = df_sae[mask]
                if len(df_sae) < len(ref_df):
                    logger.error(
                    f"Layer {layer} structural mismatch! Expected {len(ref_df)} tokens, "
                    f"but got {len(df_sae)} after alignment. "
                    f"This implies an OOM occurred during recovery that wasn't present in the reference cache. "
                    f"To prevent matrix desynchronization, you should clear your SAE cache folder and re-run globally."
                )
                raise ValueError(f"Incompatible token dimensions at layer {layer} due to asymmetric OOM execution.")
            df_sae.to_parquet(sae_output_file)
            
            all_layers_dfs[layer] = df_sae
            logger.info(f"[Computed] Successfully saved and added layer {layer} to memory pipeline")
    else:
        logger.info("All layers already cached on disk. Step skipped.")

    dict_results = None

    for layer, df_sae in tqdm(all_layers_dfs.items(), desc="Processing layers"):
        if df_sae.empty:
            logger.warning(f"SAE DataFrame for layer {layer} is empty. Skipping this layer.")
            continue
        logger.info(f"Processing layer {layer}...")
        tcf_output_file = Path(args.output_dir) / Path("output") / model_folder / sae_folder / Path(annotation_name) / Path(f"tcf_layer_{layer}.parquet")
        ast_purity_file = Path(args.output_dir) / Path("output") / model_folder / sae_folder / Path("ast_purity") / Path(f"ast_purity_layer_{layer}.parquet")
        if tcf_output_file.exists():
            logger.info(f"Output for layer {layer} already exists. Skipping...")
            continue

        sae_mask = df_sae["language"].isin(df_label["language"].unique())
        df = df_sae[sae_mask].merge(df_label, on=["path", "token_index"], how="left")

        sae_ast_mask = df_sae["language"].isin(df_ast["language"].unique())
        df_ast_act = df_sae[sae_ast_mask].merge(df_ast, on=["path", "token_index"], how="left")

        if ast_purity_file.exists():
            logger.info(f"AST purity results for layer {layer} already exist.")
        else:
            logger.info(f"Computing AST purity for layer {layer}...")
            df_ast_purity = compute_all_features_ast_purity(
                df_ast_act,
                ast_type_col="ast_type",
                feature_id_col="feature_ids",
                feature_act_col="feature_activations"
            )
            df_ast_purity.to_parquet(ast_purity_file)
            logger.info(f"Saved AST purity results for layer {layer} to {ast_purity_file}")

        if dict_results is None:
            logger.info("Grouping by individual concepts...")
            dict_results = group_by_individual_concepts_no_aggregation(df, concepts_col="concepts")
            logger.info("Grouped by individual concepts.")
        
        logger.info(f"Computing TCF metrics for layer {layer}...")
        df_tcf, _ = compute_tcf(df, selected_concepts, dict_results)
        logger.info(f"Computed TCF metrics for layer {layer}.")

        df_tcf.to_parquet(tcf_output_file)
        logger.info(f"Saved TCF results for layer {layer} to {tcf_output_file}")


if __name__ == "__main__":
    main()