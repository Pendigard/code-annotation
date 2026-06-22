from pathlib import Path

from src.parser.joern_utils import run_command
from dotenv import load_dotenv
import os
import shutil


BASE_DIR = Path.cwd()

GRAPH_EXPORT_DIR = BASE_DIR / "output" / "graph"
INPUT_SOURCE_DIR = BASE_DIR / "data" / "code" / "C++"


load_dotenv()
neo4j_import_dir = os.getenv("NEO4J_IMPORT_DIR")

if neo4j_import_dir is None:
    raise RuntimeError("NEO4J_IMPORT_DIR is not set in .env")

NEO4J_IMPORT_DIR = Path(neo4j_import_dir)

neo4j_password = os.getenv("NEO4J_PASSWORD")
if neo4j_password is None:
    raise RuntimeError("NEO4J_PASSWORD is not set in .env")

NEO4J_PASSWORD = neo4j_password


def run_joern_parse(source_path: Path, output_file: Path) -> None:
    """Generate a CPG from the source code."""
    run_command(
        [
            "joern-parse",
            str(source_path),
            "--output",
            str(output_file),
        ]
    )

def run_neo4jcsv_export(cpg_file: Path, csv_export_dir: Path) -> None:
    """Export the CPG to Neo4j."""
    run_command(
        [
            "joern-export",
            str(cpg_file),
            "--out",
            str(csv_export_dir),
            "--repr=all",
            "--format=neo4jcsv"
        ]
    )


def run_neo4j_import(csv_export_dir: Path, file_pattern: str) -> None:
    run_command(
        [
            "find",
            str(csv_export_dir),
            "-name",
            file_pattern,
            "-exec",
            "cypher-shell",
            "-u",
            "neo4j",
            "-p",
            NEO4J_PASSWORD,
            "--file",
            "{}",
            ";"
        ]
    )

def run_full_neo4j_import(csv_export_dir: Path, neo4j_import_dir: Path) -> None:
    neo4j_import_dir.mkdir(parents=True, exist_ok=True)

    for file in neo4j_import_dir.glob("*"):
        if file.is_file():
            file.unlink()

    for file in csv_export_dir.glob("*_data.csv"):
        shutil.copy(file, neo4j_import_dir)

    run_neo4j_import(csv_export_dir, "nodes_*_cypher.csv")
    run_neo4j_import(csv_export_dir, "edges_*_cypher.csv")

def run_joern_to_neo4j_pipeline(source_path: Path, output_file: Path, csv_export_dir: Path, neo4j_import_dir: Path) -> None:
    run_joern_parse(source_path, output_file)
    run_neo4jcsv_export(output_file, csv_export_dir)
    run_full_neo4j_import(csv_export_dir, neo4j_import_dir)


def main() -> None:
    GRAPH_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = GRAPH_EXPORT_DIR / "cpp" / "cpg.bin"
    run_joern_parse(INPUT_SOURCE_DIR, output_file)
    print(f"Graph export completed: {output_file}")

    csv_export_dir = GRAPH_EXPORT_DIR / "cpp" / "neo4j_csv"
    run_neo4jcsv_export(output_file, csv_export_dir)
    print(f"Neo4jcsv export completed: {csv_export_dir}")

    csv_export_dir = GRAPH_EXPORT_DIR / "cpp" / "neo4j_csv"

    run_full_neo4j_import(csv_export_dir, NEO4J_IMPORT_DIR)
    print("Neo4j import completed")

if __name__ == "__main__":
    main()