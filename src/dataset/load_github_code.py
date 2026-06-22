from datasets import load_dataset
import json
import argparse
from tqdm import tqdm
import os

LANGUAGES_FILE = "data/languages.json"

with open(LANGUAGES_FILE, "r") as f:
    languages = json.load(f)

AVAILABLE_LANGUAGES = list(languages.keys())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load and filter the GitHub code dataset.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to load from the dataset.")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save the loaded dataset.")
    parser.add_argument("--languages", nargs="+", default=AVAILABLE_LANGUAGES, help="List of programming languages to include in the dataset.")
    parser.add_argument("--min-code-length", type=int, default=10, help="Minimum number of lines of code for a sample to be included.")
    parser.add_argument("--max-code-length", type=int, default=1000, help="Maximum number of lines of code for a sample to be included.")
    return parser.parse_args()

def write_code_sample(code, path, language, data_path):
    filename = path.split("/")[-1]
    num_files = len(os.listdir(f"{data_path}/{language}")) if os.path.exists(f"{data_path}/{language}") else 0
    save_path = f"{data_path}/{language}/{num_files}_{filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(code)

def main():
    args = parse_arguments()
    
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
    count = 0
    for i, code_sample in tqdm(enumerate(ds), total=args.num_samples):
        if code_sample["language"] in args.languages:
            code = code_sample["code"]
            if code.count("\n") < args.min_code_length or code.count("\n") > args.max_code_length:
                continue
            path = code_sample["path"]
            language = code_sample["language"]


            write_code_sample(code, path, language, args.data_path)
            count += 1
        if count >= args.num_samples:
            break
    
    print(f"Loaded {args.num_samples} samples from the GitHub code dataset and saved them to {args.data_path}.")


if __name__ == "__main__":
    main()
    