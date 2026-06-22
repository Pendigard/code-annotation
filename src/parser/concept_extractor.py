from pathlib import Path
from typing import Any, Dict, List
import json

from neo4j import GraphDatabase

from src.parser.extractors import *
from src.parser.parser import NEO4J_PASSWORD


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
    registry = ConceptRegistry()
    registry.register(FutureMutatedVariable())
    registry.register(FixedValue())
    registry.register(Gatherer())

    source_dir = Path("data/code/C++")

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", NEO4J_PASSWORD),
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