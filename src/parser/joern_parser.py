# concept_extractor.py
# pip install cpgqls-client
#
# Start Joern first:
# joern --server --server-host localhost --server-port 8080

from __future__ import annotations

import argparse

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

from cpgqls_client import CPGQLSClient, import_code_query

from src.parser.extractors import *


JOERN_ENDPOINT = "localhost:8080"


# ----------------------------
# Joern client helpers
# ----------------------------


def import_project(client: CPGQLSClient, source_path: str, project_name: str) -> str:
    source_path = str(Path(source_path).resolve())
    query = import_code_query(source_path, project_name)
    return run_query(client, query)



# ----------------------------
# Concept extractor interface
# ----------------------------

@dataclass
class ConceptAnnotation:
    concept: str
    code: str
    file: str
    line: int
    column: int
    method: str | None = None
    extra: Dict[str, Any] | None = None

# ----------------------------
# Registry
# ----------------------------

class ConceptRegistry:
    def __init__(self):
        self.extractors: Dict[str, ConceptExtractor] = {}

    def register(self, extractor: ConceptExtractor) -> None:
        self.extractors[extractor.name] = extractor

    def run_all(self, client: CPGQLSClient) -> List[Dict[str, Any]]:
        all_annotations = []

        for name, extractor in self.extractors.items():
            print(f"[+] Extracting concept: {name}")
            raw = run_query(client, extractor.query())
            annotations = extractor.parse(raw)
            all_annotations.extend(annotations)

        return all_annotations


# ----------------------------
# Output helpers
# ----------------------------

def save_jsonl(items: List[Dict[str, Any]], output_path: str, jsonl_writing_mode: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(jsonl_writing_mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ----------------------------
# Debugging helpers
# ----------------------------

def debug_query(
    client: CPGQLSClient,
    query: str,
    title: str | None = None,
    max_chars: int = 5000,
) -> str:
    """
    Execute a Joern query and pretty-print the raw result.
    Useful for debugging Joern traversals.
    """
    if title:
        print(f"\n[DEBUG] {title}")

    print("\n[DEBUG] Query:")
    print(query)

    raw = run_query(client, query)

    print("\n[DEBUG] Result:")
    if len(raw) > max_chars:
        print(raw[:max_chars] + "\n...[truncated]...")
    else:
        print(raw)

    return raw

# ----------------------------
# Argument parsing helpers
# ----------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joern concept extractor")
    parser.add_argument("--code-dir", type=str, default="data/code/py/", help="Directory containing source code to analyze")
    parser.add_argument("--project-name", type=str, default="Python", help="Name of the project in Joern")
    parser.add_argument("--output-path", type=str, default="outputs/annotations.jsonl", help="Path to save extracted annotations")
    parser.add_argument("--jsonl-writing-mode", type=str, choices=["w", "a"], default="w", help="Whether to overwrite or append to the output JSONL file")
    return parser.parse_args()
"""
Example call:
python -m src.parser.joern_parser \
    --code-dir data/code/py/ \
    --project-name Python \
    --output-path outputs/annotations.jsonl \
    --jsonl-writing-mode a
"""

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_arguments()
    code_dir = args.code_dir
    project_name = args.project_name
    output_path = args.output_path
    jsonl_writing_mode = args.jsonl_writing_mode
    client = CPGQLSClient(JOERN_ENDPOINT)

    print("[+] Importing code into Joern...")
    print(import_project(client, code_dir, project_name))

    registry = ConceptRegistry()
    registry.register(ArithmeticExpressionExtractor())
    registry.register(LoopBlockExtractor())


    print("[+] Debugging queries...")
    print(registry.extractors)
    debug_query(client, """cpg.controlStructure
  .controlStructureType("FOR|WHILE|DO")
  .map { loop =>
    val startLine = loop.lineNumber.getOrElse(-1)
    // 1. On récupère toutes les lignes de l'AST sous forme de liste d'entiers valides
    val allLines = loop.ast.lineNumber.flatMap(_.toList).l
    // 2. On extrait la valeur maximale de cette liste
    val endLine = if (allLines.nonEmpty) allLines.max else startLine
    
    (loop.code, startLine, endLine)
  }
  .l
""", title="LoopBlockExtractor query")

    annotations = registry.run_all(client)

    print(f"[+] Extracted {len(annotations)} annotations")
    save_jsonl(annotations, output_path, jsonl_writing_mode)

    print(f"[+] Saved annotations to {output_path}")


if __name__ == "__main__":
    main()