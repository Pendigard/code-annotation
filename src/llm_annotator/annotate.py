import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_annotator.build_prompt import build_design_pattern_prompt


MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

CODE_SAMPLE = """
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def log(self, message):
        print(message)
"""


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text}")

    return json.loads(match.group(0))


def build_messages(code: str):
    prompt_parts = build_design_pattern_prompt()

    user_prompt = prompt_parts["user_prompt_template"].format(code=code)

    return [
        {
            "role": "system",
            "content": (
                prompt_parts["system_prompt"]
                + "\n\n"
                + prompt_parts["taxonomy"]
                + "\n\n"
                + prompt_parts["annotation_instructions"]
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def annotate_code(code: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = build_messages(code)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_json(response)


def main():
    annotation = annotate_code(CODE_SAMPLE)
    print(json.dumps(annotation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()