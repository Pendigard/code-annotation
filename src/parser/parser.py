import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.parser.joern_utils import run_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "code"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "graph"


@dataclass(frozen=True)
class LanguageConfig:
    source_directory: str
    joern_language: str


# Keys are stable CLI/output names; source_directory matches data/code exactly.
LANGUAGES: dict[str, LanguageConfig] = {
    "c": LanguageConfig("C", "C"),
    "cpp": LanguageConfig("C++", "C"),
    "csharp": LanguageConfig("C#", "CSHARPSRC"),
    "java": LanguageConfig("Java", "JAVASRC"),
    "javascript": LanguageConfig("JavaScript", "JAVASCRIPT"),
    "php": LanguageConfig("PHP", "PHP"),
    "python": LanguageConfig("Python", "PYTHONSRC"),
    "ruby": LanguageConfig("Ruby", "RUBYSRC"),
}

LANGUAGE_ALIASES = {
    "c++": "cpp",
    "c#": "csharp",
    "cs": "csharp",
    "js": "javascript",
    "py": "python",
    "rb": "ruby",
}


def normalize_language(value: str) -> str:
    """Return the canonical CLI name for a language."""
    normalized = value.strip().lower()
    return LANGUAGE_ALIASES.get(normalized, normalized)


def resolve_languages(requested: Sequence[str]) -> list[str]:
    """Validate and de-duplicate requested languages while preserving order."""
    normalized = [normalize_language(language) for language in requested]

    if "all" in normalized:
        if len(normalized) != 1:
            raise ValueError("'all' cannot be combined with individual languages")
        return list(LANGUAGES)

    unknown = sorted(set(normalized) - LANGUAGES.keys())
    if unknown:
        supported = ", ".join(LANGUAGES)
        raise ValueError(
            f"Unsupported language(s): {', '.join(unknown)}. "
            f"Supported values: {supported}, all"
        )

    return list(dict.fromkeys(normalized))


def run_joern_parse(
    source_path: Path,
    output_file: Path,
    joern_language: str,
    *,
    no_overlays: bool = False,
    enable_file_content: bool = False,
    max_num_def: int | None = None,
) -> None:
    """Generate one CPG with an explicit Joern frontend."""
    command = [
        "joern-parse",
        str(source_path),
        "--language",
        joern_language,
        "--output",
        str(output_file),
    ]

    if no_overlays:
        command.append("--nooverlays")
    if max_num_def is not None:
        command.extend(["--max-num-def", str(max_num_def)])
    if enable_file_content:
        command.extend(["--frontend-args", "--enable-file-content"])

    run_command(command)


def parse_language(
    language: str,
    source_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    no_overlays: bool = False,
    enable_file_content: bool = False,
    max_num_def: int | None = None,
) -> Path | None:
    """Parse one language directory and return its CPG path, or None if skipped."""
    config = LANGUAGES[language]
    source_path = source_root / config.source_directory
    output_file = output_root / language / "cpg.bin"

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Source directory for {language!r} does not exist: {source_path}"
        )

    if output_file.exists() and not overwrite:
        print(
            f"[skip] {language}: {output_file} already exists (use --overwrite)",
            flush=True,
        )
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(".bin.tmp")
    temporary_output.unlink(missing_ok=True)

    print(
        f"[parse] {language}: {source_path} "
        f"(Joern frontend: {config.joern_language})",
        flush=True,
    )
    started_at = time.monotonic()
    try:
        run_joern_parse(
            source_path,
            temporary_output,
            config.joern_language,
            no_overlays=no_overlays,
            enable_file_content=enable_file_content,
            max_num_def=max_num_def,
        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    elapsed = time.monotonic() - started_at

    if not temporary_output.is_file():
        raise RuntimeError(f"Joern completed without creating {temporary_output}")

    temporary_output.replace(output_file)

    size_mib = output_file.stat().st_size / (1024 * 1024)
    print(
        f"[done]  {language}: {output_file} ({size_mib:.1f} MiB, {elapsed:.1f}s)",
        flush=True,
    )
    return output_file


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one native Joern CPG per language directory for downstream "
            "concept extraction and subgraph mining."
        )
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        metavar="LANGUAGE",
        help=(
            "languages to parse (c, cpp, csharp, java, javascript, php, python, "
            "ruby), or 'all'"
        ),
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="list configured languages and their Joern frontends, then exit",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"directory containing language folders (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CPG output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing cpg.bin files; otherwise they are skipped",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first Joern failure instead of trying remaining languages",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="disable Joern's default semantic, control-flow and data-flow overlays",
    )
    parser.add_argument(
        "--enable-file-content",
        action="store_true",
        help="store source content in FILE nodes for later source-offset operations",
    )
    parser.add_argument(
        "--max-num-def",
        type=int,
        help="maximum definitions per method considered by Joern data-flow analysis",
    )
    return parser


def print_languages() -> None:
    print("Language    Source folder    Joern frontend")
    for name, config in LANGUAGES.items():
        print(f"{name:<11} {config.source_directory:<16} {config.joern_language}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.list_languages:
        print_languages()
        return 0

    if not args.languages:
        parser.error("--languages is required unless --list-languages is used")
    if args.max_num_def is not None and args.max_num_def < 1:
        parser.error("--max-num-def must be a positive integer")
    if shutil.which("joern-parse") is None:
        parser.error("joern-parse was not found in PATH")

    try:
        languages = resolve_languages(args.languages)
    except ValueError as error:
        parser.error(str(error))

    source_root = args.source_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    missing_directories = [
        source_root / LANGUAGES[language].source_directory
        for language in languages
        if not (source_root / LANGUAGES[language].source_directory).is_dir()
    ]
    if missing_directories:
        parser.error(
            "missing source directories: "
            + ", ".join(str(path) for path in missing_directories)
        )

    output_root.mkdir(parents=True, exist_ok=True)

    completed: list[Path] = []
    failures: list[tuple[str, str]] = []
    skipped = 0
    for language in languages:
        try:
            output_file = parse_language(
                language,
                source_root,
                output_root,
                overwrite=args.overwrite,
                no_overlays=args.no_overlays,
                enable_file_content=args.enable_file_content,
                max_num_def=args.max_num_def,
            )
        except (FileNotFoundError, RuntimeError) as error:
            print(f"[fail]  {language}: {error}", flush=True)
            failures.append((language, str(error)))
            if args.fail_fast:
                break
            continue

        if output_file is None:
            skipped += 1
        else:
            completed.append(output_file)

    print(
        f"CPG generation complete: {len(completed)} created, "
        f"{skipped} skipped, {len(failures)} failed, {len(languages)} selected."
    )
    if failures:
        print("Failed languages: " + ", ".join(name for name, _ in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
