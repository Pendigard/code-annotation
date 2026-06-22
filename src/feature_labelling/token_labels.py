import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import build_line_start_offsets, line_span_to_char_span, load_jsonl, normalize_annotation


TOKEN_LABEL_COLUMNS = [
    "path",
    "language",
    "line",
    "column",
    "code",
    "variable_name",
    "token_index",
    "token_id",
    "token",
    "annotations",
    "concepts",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a token-level dataframe from annotated JSONL data.")
    parser.add_argument("--annotations-path", type=str, required=True, help="Path to the annotated JSONL file.")
    parser.add_argument("--output-path", type=str, default=None, help="Optional path to store the dataframe.")
    parser.add_argument("--tokenizer-name", type=str, default="google/gemma-2-2b", help="Tokenizer name or path used to tokenize code.")
    parser.add_argument("--annotation-type", type=str, choices=["joern", "llm"], default="llm", help="Type of annotation format to process.")
    return parser.parse_args()


def _tokenize_with_offsets(tokenizer, code: str):
    encoded = tokenizer(
        code,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, token_strings, offsets


def _span_char_ranges(record: dict, code: str) -> list[tuple[int, int, dict]]:
    line_offsets = build_line_start_offsets(code)
    ranges = []
    for span in record.get("spans", []):
        start_char, end_char = line_span_to_char_span(
            line_offsets,
            int(span.get("start_line", 1)),
            int(span.get("end_line", 1)),
            len(code),
        )
        for annotation in span.get("annotations", []):
            ranges.append((start_char, end_char, normalize_annotation(annotation)))
    return ranges



def is_only_space(s):
    return s.replace("▁", "") == ""


def _annotation_label(annotation: dict) -> str | None:
    return annotation.get("label") or annotation.get("concept")


def _build_dataframe(rows: list[dict], filter_space_tokens: bool) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows, columns=TOKEN_LABEL_COLUMNS)
    if filter_space_tokens and not df.empty:
        df = df[~df["token"].apply(is_only_space)].reset_index(drop=True)
    return df


def make_rows(
    path: str,
    token_ids,
    token_strings,
    token_offsets,
    span_ranges,
    annotation_keys=("concept",),
    metadata=None,
):
    rows = []
    metadata = metadata or {}

    language = path.split("/")[-2]
    for token_index, (token_id, token_string, (token_start, token_end)) in enumerate(zip(token_ids, token_strings, token_offsets)):
        token_annotations = []
        for span_start, span_end, annotation in span_ranges:
            if token_end > span_start and token_start < span_end:
                token_annotations.append(annotation)

        if not token_annotations:
            continue

        deduped_annotations = []
        seen = set()
        for annotation in token_annotations:
            marker = tuple(annotation.get(key) for key in annotation_keys)
            if marker not in seen:
                seen.add(marker)
                deduped_annotations.append(annotation)

        rows.append(
            {
                "path": path,
                "language": language,
                **metadata,
                "token_index": token_index,
                "token_id": token_id,
                "token": token_string,
                "annotations": deduped_annotations,
                "concepts": [
                    label
                    for label in (_annotation_label(annotation) for annotation in deduped_annotations)
                    if label is not None
                ],
            }
        )
    return rows


def build_token_label_dataframe_joern(annotations_path, tokenizer, filter_space_tokens: bool = True) -> pd.DataFrame:
    rows = []
    file_tokens = {} # Cache for tokenization results to avoid redundant tokenization of the same file

    for record in tqdm(load_jsonl(annotations_path), desc="Building token labels", unit="annotation"):
        file_path = Path(record["path"])
        code = Path(file_path).read_text(encoding="utf-8")
        if file_path not in file_tokens:
            token_ids, token_strings, token_offsets = _tokenize_with_offsets(tokenizer, code)
            file_tokens[file_path] = (token_ids, token_strings, token_offsets)
        else:
            token_ids, token_strings, token_offsets = file_tokens[file_path]
        span_ranges = [
            (
                int(record["span_start"]),
                int(record["span_end"]),
                {"concept": record["concept"]},
            )
        ]
        rows.extend(
            make_rows(
                str(file_path),
                token_ids,
                token_strings,
                token_offsets,
                span_ranges,
                annotation_keys=("concept",),
                metadata={
                    "line": record.get("line"),
                    "column": record.get("column"),
                    "code": code,
                    "variable_name": record.get("variable_name"),
                },
            )
        )

    return _build_dataframe(rows, filter_space_tokens)


def build_token_label_dataframe_llm(annotations_path, tokenizer, filter_space_tokens: bool = True) -> pd.DataFrame:
    rows = []

    for record in tqdm(load_jsonl(annotations_path), desc="Building token labels", unit="annotations"):
        if not record.get("spans"):
            continue

        file_path = Path(record["path"])
        code = file_path.read_text(encoding="utf-8")
        token_ids, token_strings, token_offsets = _tokenize_with_offsets(tokenizer, code)

        span_ranges = _span_char_ranges(record, code)
        rows.extend(
            make_rows(
                str(file_path),
                token_ids,
                token_strings,
                token_offsets,
                span_ranges,
                annotation_keys=("category", "label", "label_type", "confidence"),
            )
        )

    return _build_dataframe(rows, filter_space_tokens)


def build_token_label_dataframe(
    annotations_path,
    tokenizer,
    annotation_type: str = "llm",
    filter_space_tokens: bool = True,
) -> pd.DataFrame:
    if annotation_type == "joern":
        return build_token_label_dataframe_joern(annotations_path, tokenizer, filter_space_tokens)
    if annotation_type == "llm":
        return build_token_label_dataframe_llm(annotations_path, tokenizer, filter_space_tokens)
    raise ValueError(f"Unknown annotation_type: {annotation_type!r}")


def main() -> None:
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataframe = build_token_label_dataframe(args.annotations_path, tokenizer, args.annotation_type)

    if args.output_path:
        output_path = Path(args.output_path)
        if output_path.suffix.lower() == ".csv":
            dataframe.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in {".pkl", ".pickle"}:
            dataframe.to_pickle(output_path)
        else:
            dataframe.to_parquet(output_path, index=False)
        print(f"Saved dataframe with {len(dataframe)} rows to {output_path}")
    else:
        print(dataframe.head())


if __name__ == "__main__":
    main()