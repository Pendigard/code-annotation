from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FewShotExample:
    title: str
    code: str
    expected_output: dict[str, Any]


@dataclass(frozen=True)
class PromptConfig:
    concepts_path: str
    system_prompt: str
    taxonomy_title: str
    taxonomy_intro: str
    user_prompt_title: str
    annotation_guidelines: list[str]
    json_schema_labels_item: dict[str, Any]
    no_detected_key: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)


# Unified JSON schemas & guidelines mapped out to clear data structures
_COMMON_GUIDELINES = [
    "Do not guess. If evidence is weak, return no label.",
    "Prefer the most specific label available.",
    "Provide short evidence grounded in the code structure.",
    "Provide all lines associated with the concept, not just a single line, to give a clearer picture.",
]

_COMMON_ELEMENTS_SCHEMA = [{
    "start_line": "number",
    "end_line": "number",
    "snippet": "string (the continuous block of code spanning these lines)"
}]

# ---------------------
# | Few-Shot Examples |
# ---------------------

IDIOMS_AND_STRUCTURES_EXAMPLES = [
    FewShotExample(
        title="Example 1: Guard Clauses combined with Context Managers",
        code=(
            "1: def process_log_file(file_path, min_severity):\n"
            "2:     if not file_path or min_severity is None:\n"
            "3:         return []\n"
            "4: \n"
            "5:     matching_lines = []\n"
            "6:     with open(file_path, \"r\", encoding=\"utf-8\") as stream:\n"
            "7:         for line in stream:\n"
            "8:             if min_severity in line:\n"
            "9:                 matching_lines.append(line.strip())\n"
            "10: \n"
            "11:     return matching_lines"
        ),
        expected_output={
            "labels": [
                {
                    "idiom": "Guard clauses",
                    "confidence": "high",
                    "evidence": "Handles invalid parameter states right at the top of the function to return early and keep the main code branch shallow.",
                    "code_elements": [
                        {
                            "start_line": 2,
                            "end_line": 3,
                            "snippet": "    if not file_path or min_severity is None:\n        return []"
                        }
                    ]
                },
                {
                    "idiom": "Context managers",
                    "confidence": "high",
                    "evidence": "Uses the 'with' statement wrapper to safely acquire, read, and guarantee the closing cleanup of a file resource on scope exit.",
                    "code_elements": [
                        {
                            "start_line": 6,
                            "end_line": 9,
                            "snippet": "    with open(file_path, \"r\", encoding=\"utf-8\") as stream:\n        for line in stream:\n            if min_severity in line:\n                matching_lines.append(line.strip())"
                        }
                    ]
                }
            ],
            "no_idiom_detected": False
        }
    ),
    FewShotExample(
        title="Example 2: Multiple occurrences of Pattern Matching and Destructuring",
        code=(
            "1: interface User { id: string; settings: { theme: string; active: boolean } }\n"
            "2: \n"
            "3: function updateProfile([first, last]: [string, string], user: User): void {\n"
            "4:     console.log(`Updating profile for ${first} ${last}`);\n"
            "5: \n"
            "6:     const { settings: { theme } } = user;\n"
            "7:     if (theme === \"dark\") {\n"
            "8:         applyDarkStyles();\n"
            "9:     }\n"
            "10: }"
        ),
        expected_output={
            "labels": [
                {
                    "idiom": "Pattern matching and destructuring",
                    "confidence": "high",
                    "evidence": "The code extracts variables directly from structured data formats across two distinct logic spans: first by unpacking positional array elements in the function arguments, and later by extracting deep nested fields from the user object.",
                    "code_elements": [
                        {
                            "start_line": 3,
                            "end_line": 3,
                            "snippet": "function updateProfile([first, last]: [string, string], user: User): void {"
                        },
                        {
                            "start_line": 6,
                            "end_line": 6,
                            "snippet": "    const { settings: { theme } } = user;"
                        }
                    ]
                }
            ],
            "no_idiom_detected": False
        }
    )
]

DESIGN_PATTERN_EXAMPLES = [
    FewShotExample(
        title="Example 1: Dependency Injection combined with Singleton",
        code=(
            "1: public class NotificationService {\n"
            "2:     private static NotificationService instance;\n"
            "3:     private final EmailClient emailClient;\n"
            "4: \n"
            "5:     // Dependencies are supplied from the outside\n"
            "6:     public NotificationService(EmailClient emailClient) {\n"
            "7:         this.emailClient = emailClient;\n"
            "8:     }\n"
            "9: \n"
            "10:     public static synchronized NotificationService getInstance(EmailClient client) {\n"
            "11:         if (instance == null) {\n"
            "12:             instance = new NotificationService(client);\n"
            "13:         }\n"
            "14:         return instance;\n"
            "15:     }\n"
            "16: }"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Creational",
                    "pattern": "Dependency injection",
                    "confidence": "high",
                    "evidence": "Accepts an abstract EmailClient collaborator via the constructor instead of letting the class instantiate a concrete implementation itself.",
                    "code_elements": [
                        {
                            "start_line": 6,
                            "end_line": 8,
                            "snippet": "    public NotificationService(EmailClient emailClient) {\n        this.emailClient = emailClient;\n    }"
                        }
                    ]
                },
                {
                    "category": "Creational",
                    "pattern": "Singleton",
                    "confidence": "high",
                    "evidence": "Protects class instantiation behind a lazy synchronized global access check, ensuring only one instance ever occupies runtime memory.",
                    "code_elements": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "snippet": "    private static NotificationService instance;"
                        },
                        {
                            "start_line": 10,
                            "end_line": 15,
                            "snippet": "    public static synchronized NotificationService getInstance(EmailClient client) {\n        if (instance == null) {\n            instance = new NotificationService(client);\n        }\n        return instance;\n    }"
                        }
                    ]
                }
            ],
            "no_pattern_detected": False
        }
    ),
    FewShotExample(
        title="Example 2: Multiple occurrences of Strategy",
        code=(
            "1: class PaymentProcessor:\n"
            "2:     def __init__(self, validation_strategy, processing_strategy):\n"
            "3:         self._validate = validation_strategy\n"
            "4:         self._process = processing_strategy\n"
            "5: \n"
            "6:     def execute(self, user_id, amount):\n"
            "7:         # First Strategy execution point\n"
            "8:         if not self._validate(user_id):\n"
            "9:             raise ValueError(\"Invalid User\")\n"
            "10: \n"
            "11:         # Second Strategy execution point\n"
            "12:         return self._process(amount)"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Behavioral",
                    "pattern": "Strategy",
                    "confidence": "high",
                    "evidence": "The class delegates parts of its algorithm to interchangeable, polymorphic behaviors passed at initialization. This pattern is invoked twice: once for a validation routine and again for a core processing step.",
                    "code_elements": [
                        {
                            "start_line": 8,
                            "end_line": 8,
                            "snippet": "        if not self._validate(user_id):"
                        },
                        {
                            "start_line": 12,
                            "end_line": 12,
                            "snippet": "        return self._process(amount)"
                        }
                    ]
                }
            ],
            "no_pattern_detected": False
        }
    )
]

ALGORITHMIC_PARADIGM_EXAMPLES = [
    FewShotExample(
        title="Example 1: Memoization combined with a Greedy Algorithm",
        code=(
            "1: def compute_and_select_intervals(intervals, memo={}):\n"
            "2:     # 1. Memoization layer checking for a cached structure\n"
            "3:     cache_key = tuple(intervals)\n"
            "4:     if cache_key in memo:\n"
            "5:         return memo[cache_key]\n"
            "6: \n"
            "7:     # 2. Greedy strategy: sort and pick elements strictly by earliest end time\n"
            "8:     sorted_intervals = sorted(intervals, key=lambda x: x[1])\n"
            "9:     selected = []\n"
            "10:     last_end_time = -float('inf')\n"
            "11: \n"
            "12:     for start, end in sorted_intervals:\n"
            "13:         if start >= last_end_time:\n"
            "14:             selected.append((start, end))\n"
            "15:             last_end_time = end\n"
            "16: \n"
            "17:     memo[cache_key] = selected\n"
            "18:     return selected"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Dynamic programming",
                    "paradigm": "Memoization",
                    "confidence": "high",
                    "evidence": "Implements top-down caching by checking if the computation result for the input has already been stored in a lookup dictionary to bypass redundant evaluation.",
                    "code_elements": [
                        {
                            "start_line": 3,
                            "end_line": 5,
                            "snippet": "    cache_key = tuple(intervals)\n    if cache_key in memo:\n        return memo[cache_key]"
                        }
                    ]
                },
                {
                    "category": "Greedy algorithm",
                    "paradigm": "Greedy algorithm",
                    "confidence": "high",
                    "evidence": "Sorts jobs by ending boundaries and systematically grabs the next available item that finishes earliest, making a locally optimal choice at each step without backtracking.",
                    "code_elements": [
                        {
                            "start_line": 8,
                            "end_line": 15,
                            "snippet": "    sorted_intervals = sorted(intervals, key=lambda x: x[1])\n    selected = []\n    last_end_time = -float('inf')\n\n    for start, end in sorted_intervals:\n        if start >= last_end_time:\n            selected.append((start, end))\n            last_end_time = end"
                        }
                    ]
                }
            ],
            "no_paradigm_detected": False
        }
    ),
    FewShotExample(
        title="Example 2: Multiple occurrences of Divide and conquer",
        code=(
            "1: function complexTask(arr) {\n"
            "2:     if (arr.length <= 1) return arr;\n"
            "3: \n"
            "4:     const mid = Math.floor(arr.length / 2);\n"
            "5: \n"
            "6:     // First split-and-solve execution point\n"
            "7:     const leftSolved = complexTask(arr.slice(0, mid));\n"
            "8: \n"
            "9:     // Second split-and-solve execution point\n"
            "10:     const rightSolved = complexTask(arr.slice(mid));\n"
            "11: \n"
            "12:     return merge(leftSolved, rightSolved);\n"
            "13: }"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Divide and conquer",
                    "paradigm": "Divide and conquer",
                    "confidence": "high",
                    "evidence": "The function splits a structural collection into separate sub-problems of the same type recursively. This strategy is invoked explicitly across two distinct sibling positions to handle both the left partition and right partition separately before combining them.",
                    "code_elements": [
                        {
                            "start_line": 7,
                            "end_line": 7,
                            "snippet": "    const leftSolved = complexTask(arr.slice(0, mid));"
                        },
                        {
                            "start_line": 10,
                            "end_line": 10,
                            "snippet": "    const rightSolved = complexTask(arr.slice(mid));"
                        }
                    ]
                }
            ],
            "no_paradigm_detected": False
        }
    )
]

PROGRAMMING_PARADIGM_EXAMPLES = [
    FewShotExample(
        title="Example 1: Generic Programming combined with Functional Programming",
        code=(
            "1: // 1. Generic template abstracting concrete types behind parameters\n"
            "2: interface PipelineProcessor<T> {\n"
            "3:     process: (items: T[]) => T[];\n"
            "4: }\n"
            "5: \n"
            "6: function createFilterPipeline<T>(predicate: (item: T) => boolean): PipelineProcessor<T> {\n"
            "7:     return {\n"
            "8:         // 2. Declarative, side-effect-free collection transformation\n"
            "9:         process: (items: T[]) => items.filter(predicate)\n"
            "10:     };\n"
            "11: }"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Generic programming",
                    "paradigm": "Generic programming",
                    "confidence": "high",
                    "evidence": "Uses abstract type parameters '<T>' to design data structures and interfaces completely detached from concrete domain types, enabling maximum behavioral reuse.",
                    "code_elements": [
                        {
                            "start_line": 2,
                            "end_line": 4,
                            "snippet": "interface PipelineProcessor<T> {\n    process: (items: T[]) => T[];\n}"
                        },
                        {
                            "start_line": 6,
                            "end_line": 6,
                            "snippet": "function createFilterPipeline<T>(predicate: (item: T) => boolean): PipelineProcessor<T> {"
                        }
                    ]
                },
                {
                    "category": "Declarative",
                    "paradigm": "Functional",
                    "confidence": "high",
                    "evidence": "Implements an immutable transformation using higher-order functions ('filter') without changing external state or manipulating traditional loop steps.",
                    "code_elements": [
                        {
                            "start_line": 9,
                            "end_line": 9,
                            "snippet": "        process: (items: T[]) => items.filter(predicate)"
                        }
                    ]
                }
            ],
            "no_paradigm_detected": False
        }
    ),
    FewShotExample(
        title="Example 2: Multiple occurrences of Class-based Object-oriented Programming",
        code=(
            "1: // First standalone entity boundary definition\n"
            "2: class DatabaseConnection {\n"
            "3:     connect() {\n"
            "4:         System.out.println(\"Connecting...\");\n"
            "5:     }\n"
            "6: }\n"
            "7: \n"
            "8: // Second standalone entity boundary definition\n"
            "9: class UserRepository {\n"
            "10:     private final DatabaseConnection db;\n"
            "11: \n"
            "12:     public UserRepository(DatabaseConnection db) {\n"
            "13:         this.db = db;\n"
            "14:     }\n"
            "15: }"
        ),
        expected_output={
            "labels": [
                {
                    "category": "Imperative",
                    "paradigm": "Class-based",
                    "confidence": "high",
                    "evidence": "Organizes separate program states and structures around explicit blueprinted modules ('class') that bundle internal fields and operations together. This class-based pattern appears twice to define distinct, interacting types.",
                    "code_elements": [
                        {
                            "start_line": 2,
                            "end_line": 6,
                            "snippet": "class DatabaseConnection {\n    connect() {\n        System.out.println(\"Connecting...\");\n    }\n}"
                        },
                        {
                            "start_line": 9,
                            "end_line": 15,
                            "snippet": "class UserRepository {\n    private final DatabaseConnection db;\n\n    public UserRepository(DatabaseConnection db) {\n        this.db = db;\n    }\n}"
                        }
                    ]
                }
            ],
            "no_paradigm_detected": False
        }
    )
]

PROMPT_REGISTRY = {
    "design_pattern": PromptConfig(
        concepts_path="data/concepts/level_3/design_patterns.json",
        system_prompt="You are a code annotation assistant specialized in software design patterns. Identify design patterns in code samples and return consistent, well-structured labels for dataset annotation.",
        taxonomy_title="Design Pattern Taxonomy",
        taxonomy_intro="Design patterns are grouped into the following categories.",
        user_prompt_title="Code Sample",
        no_detected_key="no_pattern_detected",
        few_shot_examples=DESIGN_PATTERN_EXAMPLES,
        annotation_guidelines=[
            "Identify all design patterns that are clearly present in the code.",
            "Label a pattern only when its core intent is implemented, not just because similar syntax appears.",
            "Multiple labels are allowed if the code contains multiple design patterns.",
            "If no design pattern is present, return an empty list.",
        ] + _COMMON_GUIDELINES,
        json_schema_labels_item={
            "category": "string",
            "pattern": "string",
            "confidence": "high | medium | low",
            "evidence": "short explanation",
            "code_elements": _COMMON_ELEMENTS_SCHEMA,
        },
    ),
    "idioms_and_structures": PromptConfig(
        concepts_path="data/concepts/level_2/idioms_and_structures.json",
        system_prompt="You are a code annotation assistant specialized in idioms and language structures. Identify idiomatic code constructs and return consistent, well-structured labels for dataset annotation.",
        taxonomy_title="Idioms and Structures Taxonomy",
        taxonomy_intro="Idioms and structures are grouped into the following labels.",
        user_prompt_title="Relevant Code Excerpt",
        no_detected_key="no_idiom_detected",
        few_shot_examples=IDIOMS_AND_STRUCTURES_EXAMPLES,
        annotation_guidelines=[
            "Identify all idioms or structures that are clearly present in the code.",
            "Label an idiom only when the construct is actually used, not just because the syntax exists.",
            "Multiple labels are allowed if the code contains multiple idioms.",
            "If no idiom or structure is present, return an empty list.",
        ] + _COMMON_GUIDELINES,
        json_schema_labels_item={
            "idiom": "string",
            "confidence": "high | medium | low",
            "evidence": "short explanation",
            "code_elements": _COMMON_ELEMENTS_SCHEMA,
        },
    ),
    "algorithmic_paradigm": PromptConfig(
        concepts_path="data/concepts/level_4/algorithmic_paradigm.json",
        system_prompt="You are a code annotation assistant specialized in algorithmic paradigms. Identify algorithmic paradigms in code samples and return consistent, well-structured labels for dataset annotation.",
        taxonomy_title="Algorithmic Paradigm Taxonomy",
        taxonomy_intro="Algorithmic paradigms are grouped into the following labels.",
        user_prompt_title="Full Code Sample",
        no_detected_key="no_paradigm_detected",
        few_shot_examples=ALGORITHMIC_PARADIGM_EXAMPLES,
        annotation_guidelines=[
            "Inspect the full sample before deciding.",
            "Identify all algorithmic paradigms that are clearly present in the code.",
            "A paradigm should be labeled only when its core algorithmic strategy is implemented.",
            "Multiple labels are allowed if the code contains multiple paradigms.",
            "If no algorithmic paradigm is present, return an empty list.",
        ] + _COMMON_GUIDELINES,
        json_schema_labels_item={
            "category": "string",
            "paradigm": "string",
            "confidence": "high | medium | low",
            "evidence": "short explanation",
            "code_elements": _COMMON_ELEMENTS_SCHEMA,
        },
    ),
    "programming_paradigm": PromptConfig(
        concepts_path="data/concepts/level_4/programming_paradigm.json",
        system_prompt="You are a code annotation assistant specialized in programming paradigms. Identify the programming paradigm of a full code sample and return consistent, well-structured labels for dataset annotation.",
        taxonomy_title="Programming Paradigm Taxonomy",
        taxonomy_intro="Programming paradigms are grouped into the following categories.",
        user_prompt_title="Full Code Sample",
        no_detected_key="no_paradigm_detected",
        few_shot_examples=PROGRAMMING_PARADIGM_EXAMPLES,
        annotation_guidelines=[
            "Inspect the whole sample before deciding.",
            "Identify the programming paradigm that is clearly present in the code.",
            "A paradigm should be labeled only when it describes the overall code style, not a tiny local fragment.",
            "If no programming paradigm is present, return an empty list.",
        ] + _COMMON_GUIDELINES,
        json_schema_labels_item={
            "category": "string",
            "paradigm": "string",
            "confidence": "high | medium | low",
            "evidence": "short explanation",
            "code_elements": _COMMON_ELEMENTS_SCHEMA,
        },
    ),
}

def add_line_numbers_to_code(code: str) -> str:
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(code.splitlines()))

def _resolve_concepts_path(concepts_path: str) -> Path:
    path = Path(concepts_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=16)
def _load_concepts(concepts_path: str) -> dict:
    resolved = _resolve_concepts_path(concepts_path)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_top_level_concepts(concepts_data: dict) -> list[dict]:
    return concepts_data.get("concepts", [concepts_data])


def _generate_taxonomy_lines(concepts: list[dict], indent: int = 0) -> list[str]:
    lines = []
    prefix = "  " * indent

    for concept in concepts:
        has_children = bool(concept.get("subconcepts"))
        label_name = "Category" if indent == 0 and has_children else "Label"

        lines.append(f"{prefix}- {label_name}: {concept['name']}")
        lines.append(f"{prefix}  Description: {concept['description']}")

        if has_children:
            lines.append(f"{prefix}  Subconcepts:")
            lines.extend(_generate_taxonomy_lines(concept["subconcepts"], indent + 2))

    return lines


def _build_annotation_instructions(config: PromptConfig) -> str:
    guidelines_str = "\n".join(f"{i+1}. {g}" for i, g in enumerate(config.annotation_guidelines))
    
    schema_dict = {
        "labels": [config.json_schema_labels_item],
        config.no_detected_key: True
    }
    schema_json = json.dumps(schema_dict, indent=4)

    # Compile the few-shot section dynamically if examples are present
    few_shot_str = ""
    if config.few_shot_examples:
        few_shot_str = "\n\n## Few-Shot Examples\n\n"
        for ex in config.few_shot_examples:
            few_shot_str += (
                f"### {ex.title}\n\n"
                f"**Input:**\n```\n{ex.code}\n```\n\n"
                f"**Expected Output JSON:**\n```json\n{json.dumps(ex.expected_output, indent=4)}\n```\n\n"
            )

    return (
        f"# Annotation Instructions\n\n"
        f"Annotate the code sample using only labels from the taxonomy above.\n\n"
        f"Guidelines:\n{guidelines_str}\n\n"
        f"Return only valid JSON with this schema:\n\n"
        f"```json\n{schema_json}\n```"
        f"{few_shot_str}"
    )


def build_prompt(prompt_type: str, custom_path: str | None = None) -> dict[str, str]:
    if prompt_type not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt type: '{prompt_type}'.")

    config = PROMPT_REGISTRY[prompt_type]
    path_to_load = custom_path or config.concepts_path

    concepts_data = _load_concepts(path_to_load)
    concepts = _get_top_level_concepts(concepts_data)

    taxonomy = [f"# {config.taxonomy_title}", config.taxonomy_intro, ""]
    taxonomy.extend(_generate_taxonomy_lines(concepts))

    return {
        "system_prompt": config.system_prompt,
        "taxonomy": "\n".join(taxonomy).strip(),
        "annotation_instructions": _build_annotation_instructions(config).strip(),
        "user_prompt_template": f"# {config.user_prompt_title}\n\n```\n{{code}}\n```",
    }

if __name__ == "__main__":

    _original_load_concepts = _load_concepts.__wrapped__

    def _mock_load_concepts_fallback(concepts_path: str) -> dict:
        try:
            return _original_load_concepts(concepts_path)
        except FileNotFoundError:
            # Returns a generic minimal taxonomy mapping if files aren't created yet
            return {
                "concepts": [
                    {
                        "name": "Core Category Template",
                        "description": "Placeholder category used to demonstrate functional compilation loop.",
                        "subconcepts": [
                            {
                                "name": "Specific Implementation Metric",
                                "description": "Leaf node demonstrating structural parsing depth."
                            }
                        ]
                    }
                ]
            }

    
    _load_concepts = _mock_load_concepts_fallback

    print("=" * 80)
    print(f"GENERATING ALL SYSTEM PROMPTS FROM PROMPT_REGISTRY ({len(PROMPT_REGISTRY)} CONFIGS FOUND)")
    print("=" * 80 + "\n")

    for prompt_key in PROMPT_REGISTRY.keys():
        print(f"Executing: build_prompt('{prompt_key}')...")
        try:
            compiled_prompt = build_prompt(prompt_key)
            title_header = f" PROMPT INTERFACE FOR TYPE: {prompt_key.upper()} "
            print("\n" + "#" * 10 + title_header + "#" * 10)
            
            print("\n[SYSTEM PROMPT]")
            print(compiled_prompt["system_prompt"])
            
            print("\n[TAXONOMY STRUCTURE]")
            print(compiled_prompt["taxonomy"])
            
            print("\n[ANNOTATION INSTRUCTIONS & FEW-SHOTS]")
            print(compiled_prompt["annotation_instructions"])
            
            print("\n[USER TEMPLATE STRUCTURE]")
            print(compiled_prompt["user_prompt_template"])
            
            print("\n" + "#" * (20 + len(title_header)))
            print("-" * 80 + "\n")

        except Exception as e:
            print(f"Failed to compile registry target '{prompt_key}': {e}", file=sys.stderr)