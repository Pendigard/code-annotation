import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Adjusted to your modern refactored module setup
from src.llm_annotator.build_prompt import (
    PROMPT_REGISTRY,
    add_line_numbers_to_code,
    build_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "google/gemma-4-E4B-it"
DEFAULT_CODE_DIR = PROJECT_ROOT / "data" / "code"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "annotations_data.jsonl"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a code dataset with an LLM and write JSONL output.")
    parser.add_argument("--code-dir", type=str, default=str(DEFAULT_CODE_DIR), help="Directory containing code files.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to the JSONL output file.")
    # Choices now dynamically align with the registry config keys
    parser.add_argument("--concept", type=str, default="idioms_and_structures", choices=sorted(PROMPT_REGISTRY.keys()), help="Concept taxonomy to use for annotation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Hugging Face model name or path.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of generated tokens.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on the number of files to annotate.")
    return parser.parse_args()


def load_extension_language_map() -> dict[str, str]:
    with open(PROJECT_ROOT / "data" / "languages.json", "r", encoding="utf-8") as file:
        languages = json.load(file)

    extension_map: dict[str, str] = {}
    for language, extensions in languages.items():
        for extension in extensions:
            extension_map[extension.lower()] = language
    return extension_map


EXTENSION_TO_LANGUAGE = load_extension_language_map()


def get_language(file_path: Path) -> str:
    return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), file_path.parent.name)


def collect_code_files(code_dir: Path, limit: int = 0) -> list[Path]:
    files = [path for path in code_dir.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSION_TO_LANGUAGE]
    files.sort()
    return files[:limit] if limit > 0 else files


def clean_snippet(text: str) -> str:
    return "\n".join(re.sub(r"^\s*\d+:\s*", "", line) for line in text.splitlines()).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def levenshtein_distance(left: str, right: str) -> int:
    if left == right: return 0
    if not left: return len(right)
    if not right: return len(left)

    previous_row = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current_row = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (left_char != right_char)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def build_messages(code: str, prompt_parts: dict) -> list[dict[str, str]]:
    user_prompt = prompt_parts["user_prompt_template"].format(code=add_line_numbers_to_code(code))
    system_prompt = f"{prompt_parts['system_prompt']}\n\n{prompt_parts['taxonomy']}\n\n{prompt_parts['annotation_instructions']}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text}")
    return json.loads(match.group(0))


def annotate_code(code: str, prompt_parts: dict, processor: AutoProcessor, model: AutoModelForCausalLM, max_new_tokens: int) -> dict:
    messages = build_messages(code, prompt_parts)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:].cpu().tolist()
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_json(response)


def normalize_annotation(raw_annotation: dict, concept: str) -> dict:
    label = raw_annotation.get("pattern") or raw_annotation.get("idiom") or raw_annotation.get("paradigm") or raw_annotation.get("label") or ""
    
    label_type = "pattern"
    if "idiom" in raw_annotation:
        label_type = "idiom"
    elif "paradigm" in raw_annotation:
        label_type = "paradigm"

    return {
        "category": raw_annotation.get("category") or concept,
        "label": label,
        "label_type": label_type,
        "confidence": raw_annotation.get("confidence", "low"),
        "evidence": raw_annotation.get("evidence", ""),
    }


def match_span(code_lines: list[str], snippet: str, start_hint: int | None, end_hint: int | None) -> dict | None:
    snippet = clean_snippet(snippet)
    if not snippet or not code_lines:
        return None

    # Step 1: Trust start and end boundaries returned by the model if realistic
    if start_hint is not None and end_hint is not None:
        s_idx = max(1, min(start_hint, len(code_lines)))
        e_idx = max(s_idx, min(end_hint, len(code_lines)))
        window_text = "\n".join(code_lines[s_idx - 1 : e_idx])
        
        if normalize_text(snippet) == normalize_text(window_text):
            return {"start_line": s_idx, "end_line": e_idx, "text": window_text, "match_method": "exact_bounds"}

    # Step 2: Fallback to searching sliding windows if bounds deviated slightly
    line_hint = start_hint or 1
    normalized_snippet = normalize_text(snippet)
    exact_candidates = []
    fallback_candidates = []
    max_window = min(15, len(code_lines)) # scan up to 15 continuous lines

    for window_size in range(1, max_window + 1):
        for start_line in range(1, len(code_lines) - window_size + 2):
            end_line = start_line + window_size - 1
            window_text = "\n".join(code_lines[start_line - 1 : end_line])
            normalized_window = normalize_text(window_text)

            fallback_candidates.append((
                levenshtein_distance(normalized_snippet, normalized_window),
                abs(start_line - line_hint),
                window_size,
                start_line,
                end_line,
                window_text,
            ))

            if snippet in window_text or window_text in snippet:
                exact_candidates.append((abs(start_line - line_hint), window_size, start_line, end_line, window_text))

    if exact_candidates:
        _, _, start_line, end_line, text = min(exact_candidates)
        return {"start_line": start_line, "end_line": end_line, "text": text, "match_method": "exact_search"}

    if fallback_candidates:
        _, _, _, start_line, end_line, text = min(fallback_candidates)
        return {"start_line": start_line, "end_line": end_line, "text": text, "match_method": "levenshtein"}

    return None


def collect_spans(code: str, labels: list[dict], concept: str) -> list[dict]:
    code_lines = code.splitlines() or [code]

    if concept == "programming_paradigm":
        if not labels: return []
        return [{
            "start_line": 1,
            "end_line": len(code_lines),
            "text": code,
            "match_method": "full_file",
            "annotations": [normalize_annotation(label, concept) for label in labels],
        }]

    spans_by_key = {}

    for raw_label in labels:
        annotation = normalize_annotation(raw_label, concept)
        code_elements = raw_label.get("code_elements") or []
        
        # Handle both list-of-spans schema and legacy elements structures seamlessly
        if isinstance(code_elements, dict):
            code_elements = [code_elements]

        for element in code_elements:
            if isinstance(element, dict):
                start_hint = element.get("start_line") or element.get("line")
                end_hint = element.get("end_line") or start_hint
                snippet = str(element.get("snippet", ""))
            else:
                start_hint, end_hint = None, None
                snippet = str(element)

            span = match_span(code_lines, snippet, start_hint, end_hint)
            if span is None:
                continue

            key = (span["start_line"], span["end_line"], span["text"])
            if key not in spans_by_key:
                spans_by_key[key] = {
                    "start_line": span["start_line"],
                    "end_line": span["end_line"],
                    "text": span["text"],
                    "match_method": span["match_method"],
                    "annotations": [],
                }
            spans_by_key[key]["annotations"].append(annotation)

    spans = list(spans_by_key.values())
    spans.sort(key=lambda s: (s["start_line"], s["end_line"], s["text"]))
    return spans


def annotate_file(file_path: Path, prompt_parts: dict, concept: str, processor: AutoProcessor, model: AutoModelForCausalLM, max_new_tokens: int) -> dict:
    code = file_path.read_text(encoding="utf-8")
    raw_result = annotate_code(code, prompt_parts, processor, model, max_new_tokens)
    
    labels = raw_result.get("labels", []) if isinstance(raw_result, dict) else []
    spans = collect_spans(code, labels, concept)
    
    # Check no_detected boolean dynamically from registry specific keys
    config = PROMPT_REGISTRY[concept]
    no_detected = raw_result.get(config.no_detected_key, len(labels) == 0) if isinstance(raw_result, dict) else False

    return {
        "concept": concept,
        "language": get_language(file_path),
        "path": str(file_path.resolve()),
        "spans": spans,
        "no_pattern_detected": no_detected,
    }


def main() -> None:
    args = parse_arguments()
    code_dir = Path(args.code_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean factorization call utilizing our generic build_prompt function
    prompt_parts = build_prompt(args.concept)
    
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")

    files = collect_code_files(code_dir, args.limit)
    print(f"Annotating {len(files)} files from {code_dir} with concept '{args.concept}'.")

    with open(output_path, "w", encoding="utf-8") as output_file:
        for file_path in tqdm(files, desc="Annotating", unit="file"):
            try:
                record = annotate_file(file_path, prompt_parts, args.concept, processor, model, args.max_new_tokens)
                # Ensure valid single line json output formatting per JSONL standard specifications
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                tqdm.write(f"{file_path.name}: {len(record['spans'])} spans tagged")
            except Exception as error:
                error_record = {
                    "concept": args.concept,
                    "language": get_language(file_path),
                    "path": str(file_path.resolve()),
                    "spans": [],
                    "error": str(error),
                }
                output_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                output_file.flush()
                tqdm.write(f"{file_path.name}: failed ({error})")

    print(f"Wrote JSONL output directly to {output_path}")


if __name__ == "__main__":
    main()