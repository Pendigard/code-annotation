python -m src.parser.joern_parser \
    --code-dir data/code/test/ \
    --project-name CPP \
    --output-path outputs/annotations.jsonl \
    --jsonl-writing-mode w

python -m src.dataset.load_github_code \
    --num_samples 1000 \
    --data_path /work03/celian/code-samples \
    --min-code-length 10 \
    --max-code-length 300

python -m src.llm_annotator.annotate_data \
    --code-dir /work03/celian/code-samples \
    --output-path outputs/algorithmic_paradigm.jsonl \
    --concept "algorithmic_paradigm" \
    --max-new-tokens 4096

python -m src.SAE.encode_sae \
    --code-dir /work03/celian/code-samples \
    --output-path /work03/celian/sae_activations.parquet \
    --model-name google/gemma-2-2b \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id layer_20/width_16k/canonical \
    --layer 20 \
    --hook-point hook_resid_post \
    --method transformer_lens \
    --batch-size 32 \
    --limit 0 \
    --device cuda \
    --dtype float16

python -m src.feature_labelling.tcf_script \
    --device cuda \
    --dtype float32 \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id layer_{}/width_16k/canonical \
    --model-name google/gemma-2-2b \
    --code-dir /work03/celian/gemma-2-2b-code \
    --annotations-path outputs/ast_labels.parquet \
    --output-dir /work03/celian/code_feature_labelling