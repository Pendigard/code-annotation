import json
from pathlib import Path
import re
import unicodedata
import logging
import sys

from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def setup_logger(output_dir: Path) -> logging.Logger:
    """Configure un logger double sortie (Console + Fichier) compatible avec tqdm."""
    logger = logging.getLogger("TCF_Pipeline")
    logger.setLevel(logging.INFO)
    
    # Éviter d'ajouter plusieurs fois les mêmes handlers si la fonction est rappelée
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Handler Fichier (Sauvegarde en dur)
    log_file = Path(output_dir) / "tcf_pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler Console (Compatible avec l'affichage tqdm)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_extension_language_map() -> dict[str, str]:
    with open(PROJECT_ROOT / "data" / "languages.json", "r", encoding="utf-8") as file:
        languages = json.load(file)

    extension_map: dict[str, str] = {}
    for language, extensions in languages.items():
        for extension in extensions:
            extension_map[extension.lower()] = language
    return extension_map


EXTENSION_TO_LANGUAGE = load_extension_language_map()


def collect_code_files(code_dir: Path, limit: int = 0) -> list[Path]:
    files = [path for path in code_dir.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSION_TO_LANGUAGE]
    files.sort()
    if limit > 0:
        return files[:limit]
    return files


def get_language(file_path: Path) -> str:
    return file_path.parent.name

def load_jsonl(path: str | Path):
    with open(Path(path), "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_line_start_offsets(code: str) -> list[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def line_span_to_char_span(line_offsets: list[int], start_line: int, end_line: int, code_length: int) -> tuple[int, int]:
    start_index = max(1, start_line)
    end_index = max(start_index, end_line)
    start_char = line_offsets[start_index - 1]
    end_char = code_length if end_index >= len(line_offsets) else line_offsets[end_index]
    return start_char, end_char


def normalize_annotation(annotation: dict) -> dict:
    return {
        "category": annotation.get("category"),
        "label": annotation.get("label") or annotation.get("pattern") or annotation.get("idiom") or annotation.get("paradigm") or annotation.get("concept"),
        "label_type": annotation.get("label_type") or ("idiom" if "idiom" in annotation else "paradigm" if "paradigm" in annotation else "pattern"),
        "confidence": annotation.get("confidence"),
        "evidence": annotation.get("evidence"),
    }

def slugify(value):
    # Convert accented characters to ASCII equivalents
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    # Lowercase, strip padding, and substitute non-alphanumeric text with dashes
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def load_sae(sae_id: str, sae_release: str, device: str, dtype: str, sae_path: str | None = None) -> SAE:
    if sae_path:
        sae_path = Path(sae_path).expanduser().resolve()
        if sae_path.is_dir():
            return SAE.load_from_disk(sae_path, device=device, dtype=dtype)
        return SAE.load_from_disk(sae_path, device=device, dtype=dtype)

    print(f"Loading SAE model '{sae_id}' from release '{sae_release}'...")
    return SAE.from_pretrained(sae_release, sae_id, device=device, dtype=dtype)


def load_model(model_name: str, device: str | None, dtype: str):
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    return model


def build_hook_name(layer: int, hook_point: str) -> str:
    return f"blocks.{layer}.{hook_point}"