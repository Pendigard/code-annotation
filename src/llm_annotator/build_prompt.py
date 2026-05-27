import json
from pathlib import Path


def build_design_pattern_prompt(
    concepts_path: str = "data/concepts/level_3/design_patterns.json",
):
    with open(concepts_path, "r", encoding="utf-8") as f:
        design_patterns = json.load(f)

    system_prompt = (
        "You are a code annotation assistant specialized in software design patterns. "
        "Your goal is to identify design patterns in code samples and return consistent, "
        "well-structured labels for dataset annotation."
    )

    taxonomy = ["# Design Pattern Taxonomy\n"]
    taxonomy.append("Design patterns are grouped into the following categories.\n")

    for category in design_patterns["concepts"]:
        taxonomy.append(f"## Category: {category['name']}")
        taxonomy.append(f"Description: {category['description']}")
        taxonomy.append("Patterns:")

        for pattern in category["subconcepts"]:
            taxonomy.append(
                f"- Label: {pattern['name']}\n"
                f"  Description: {pattern['description']}"
            )

        taxonomy.append("")

    annotation_instructions = """
# Annotation Instructions

You must annotate the code sample using only labels from the taxonomy above.

Guidelines:
1. Identify all design patterns that are clearly present in the code.
2. Do not guess. If evidence is weak, mark the pattern as "uncertain".
3. A pattern should be labeled only if its core intent is implemented, not merely because the code uses similar syntax.
4. Multiple labels are allowed if the code contains multiple design patterns.
5. If no design pattern is present, return an empty list.
6. Prefer the most specific pattern label available.
7. Provide short evidence grounded in the code structure.

Return only valid JSON with this schema:

{
  "labels": [
    {
      "category": "string",
      "pattern": "string",
      "confidence": "high | medium | low",
      "evidence": "short explanation",
      "code_elements": ["class/function/method names involved"]
    }
  ],
  "no_pattern_detected": true | false
}
""".strip()

    user_prompt_template = """
# Code Sample

```
{code}
```
""".strip()
    
    return {
        "system_prompt": system_prompt,
        "taxonomy": "\n".join(taxonomy),
        "annotation_instructions": annotation_instructions,
        "user_prompt_template": user_prompt_template,
    }


if __name__ == "__main__":
    prompt_parts = build_design_pattern_prompt()
    print(prompt_parts["system_prompt"])
    print()
    print(prompt_parts["taxonomy"])
    print()
    print(prompt_parts["annotation_instructions"])