from pathlib import Path
from typing import Any, Dict, List
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.parser.extractors import *

import re


class ConceptRegistry:
    def __init__(self) -> None:
        self.extractors: Dict[str, Neo4jConceptExtractor] = {}

    def register(self, extractor: Neo4jConceptExtractor) -> None:
        self.extractors[extractor.name] = extractor

    def run_all(
        self,
        driver: Any,
        source_dir: Path,
    ) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []

        for name, extractor in self.extractors.items():
            print(f"[+] Extracting concept: {name}")
            annotations.extend(extractor.get_annotations(driver, source_dir))

        return annotations


def main() -> None:
    load_dotenv()
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if neo4j_password is None:
        raise RuntimeError("NEO4J_PASSWORD is not set in .env")

    registry = ConceptRegistry()
    variable_extractors = [
        FutureMutatedVariable,
        FixedValue,
        Gatherer,
        Stepper,
        Walker,
        MostRecentHolder,
        MostWantedHolder,
        OneWayFlag,
        Organizer,
        Container,
        VariableDependent,
        Follower,
        FutureDependency,
        FutureReturnDependency,
        VariableUsedInFunction,
        Temporary,
        FutureBranchDependency,
        FutureRecursiveCall,
    ]
    for extractor_cls in variable_extractors:
        for decl_only in [True, False]:
            name_suffix = "_decl" if decl_only else ""
            extractor = extractor_cls(name=f"{re.sub(r'(?<!^)(?=[A-Z])', '_', extractor_cls.__name__).lower()}{name_suffix}", decl_only=decl_only)
            registry.register(extractor)
            print(f"Registered extractor: {extractor.name}")
            print(f"Extractor query: {extractor.query}\n")

    node_extractors = [
        SingleControlFlow,
        BinaryControlFlow,
        NCaseControlFlow,
        FutureLoopEntry,
        FutureBranchMerge,
    ]
    for extractor_cls in node_extractors:
        extractor = extractor_cls(name=f"{re.sub(r'(?<!^)(?=[A-Z])', '_', extractor_cls.__name__).lower()}")
        registry.register(extractor)
        print(f"Registered extractor: {extractor.name}")
        print(f"Extractor query: {extractor.query}\n")

    source_dir = Path("data/code/C++")

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", neo4j_password),
    )

    try:
        annotations = registry.run_all(driver, source_dir)

        print(f"Total annotations extracted: {len(annotations)}")
        
        output_file = Path("output/annotations/joern_annotations.jsonl")
        with output_file.open("w") as f:
            for annotation in annotations:
                json.dump(annotation, f, ensure_ascii=False)
                f.write("\n")
        


    finally:
        driver.close()


if __name__ == "__main__":
    main()
