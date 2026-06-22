from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tree_sitter import Language, Node, Parser

import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp
import tree_sitter_javascript as tsjavascript
import tree_sitter_java as tsjava


LANGUAGE_BUILDERS = {
    "Python": tspython.language,
    "C++": tscpp.language,
    "JavaScript": tsjavascript.language,
    "Java": tsjava.language,
}


def get_language_dict(languages: Iterable[str]) -> Dict[str, Language]:
    language_dict = {}

    for lang in languages:
        if lang not in LANGUAGE_BUILDERS:
            raise ValueError(f"Unsupported language: {lang}")

        language_dict[lang] = Language(LANGUAGE_BUILDERS[lang]())

    return language_dict


def build_char_to_byte_map(text: str) -> List[int]:
    """char_index -> byte_index."""
    mapping = [0]
    byte_pos = 0

    for char in text:
        byte_pos += len(char.encode("utf-8"))
        mapping.append(byte_pos)

    return mapping


def tokenize_with_byte_offsets(
    tokenizer: PreTrainedTokenizerBase,
    code: str,
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    encoded = tokenizer(
        code,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    token_ids = encoded["input_ids"]
    char_offsets = encoded["offset_mapping"]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    char_to_byte = build_char_to_byte_map(code)

    byte_offsets = [
        (char_to_byte[start], char_to_byte[end])
        for start, end in char_offsets
    ]

    return token_ids, token_strings, byte_offsets


def find_deepest_named_node(
    node: Node,
    start_byte: int,
    end_byte: int,
) -> Optional[Node]:
    """Retourne le plus petit noeud nommé contenant le span."""
    if start_byte < node.start_byte or end_byte > node.end_byte:
        return None

    best = node if node.is_named else None

    for child in node.children:
        if start_byte >= child.start_byte and end_byte <= child.end_byte:
            candidate = find_deepest_named_node(child, start_byte, end_byte)
            if candidate is not None:
                best = candidate
            break

    return best


def iter_code_files(folder_path: Path) -> Iterable[Path]:
    for path in folder_path.rglob("*"):
        if path.is_file():
            yield path

def find_containing_named_nodes(
    node: Node,
    start_byte: int,
    end_byte: int,
) -> List[Node]:
    # 1. Si le token n'intersecte pas du tout le nœud courant, on s'arrête
    if end_byte <= node.start_byte or start_byte >= node.end_byte:
        return []

    # 2. Si le nœud intersecte le token et qu'il est nommé, on l'ajoute au chemin
    nodes = [node] if node.is_named else []

    # 3. On explore TOUS les enfants qui intersectent le token
    # (On retire le 'break' car un token BPE peut chevaucher plusieurs nœuds enfants)
    for child in node.children:
        if not (end_byte <= child.start_byte or start_byte >= child.end_byte):
            nodes.extend(find_containing_named_nodes(child, start_byte, end_byte))
            
    return nodes

def build_ast_labels(
    tokenizer: PreTrainedTokenizerBase,
    language_dict: Dict[str, Language],
    code_path: str | Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    code_root = Path(code_path)

    for lang, language in tqdm(language_dict.items(), desc="Processing languages"):
        parser = Parser(language)
        folder_path = code_root / lang

        if not folder_path.exists():
            print(f"Warning: folder not found: {folder_path}")
            continue

        for file_path in tqdm(
            list(iter_code_files(folder_path)),
            desc=f"Processing {lang}",
            leave=False,
        ):
            code = file_path.read_text(encoding="utf-8")
            code_bytes = code.encode("utf-8")

            tree = parser.parse(code_bytes)
            root_node = tree.root_node

            token_ids, token_strings, token_offsets = tokenize_with_byte_offsets(
                tokenizer,
                code,
            )

            for token_index, (token_id, token, (start_byte, end_byte)) in enumerate(
                zip(token_ids, token_strings, token_offsets)
            ):
                if start_byte == end_byte:
                    continue

                nodes = find_containing_named_nodes(root_node, start_byte, end_byte)
                concepts = [node.type for node in nodes]

                rows.append(
                    {
                        "path": str(file_path),
                        "language": lang,
                        "token_index": token_index,
                        "token_id": token_id,
                        "token": token,
                        "start_byte": start_byte,
                        "end_byte": end_byte,
                        "concepts": concepts,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    languages = ["Python", "C++", "JavaScript", "Java"]
    language_dict = get_language_dict(languages)

    code_path = Path("/work03/celian/code-samples")
    output_path = Path("outputs/ast_labels_2.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_ast_labels(tokenizer, language_dict, code_path)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()