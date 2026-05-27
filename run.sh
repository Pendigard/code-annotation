python -m src.parser.joern_parser \
    --code-dir data/code/py/ \
    --project-name Python \
    --output-path outputs/annotations.jsonl \
    --jsonl-writing-mode a

python -m src.dataset.load_github_code \
    --num_samples 1000 \
    --data_path /work03/celian/code-samples 