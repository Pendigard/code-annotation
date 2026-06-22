# SAE Code Auto-Labelling Pipeline

Research prototype for automatically assigning human-readable programming concepts to Sparse Autoencoder (SAE) features activated by source code.

The project studies how large language models represent code internally. It combines concept annotations, token-level SAE activations, and statistical association metrics to identify features that may encode syntax, programming idioms, design patterns, or broader algorithmic and programming paradigms.

> **Status:** active research and development. The main components are implemented and can be run independently, while the end-to-end workflow and configuration are still being consolidated.

## Research objective

Sparse Autoencoders decompose a model's hidden activations into a large set of sparse features. These features are easier to inspect than dense activations, but they do not come with semantic labels.

This repository explores an automatic labelling strategy:

1. collect source-code samples;
2. annotate code spans with known programming concepts;
3. align those annotations with the model tokenizer;
4. extract token-level SAE feature activations;
5. measure the association between concepts and features;
6. rank candidate feature labels and assess their specificity.

The current scoring pipeline uses co-occurrence statistics such as Pointwise Mutual Information (PMI), precision, recall, F1, and a TCF score computed from feature activations. AST purity is also computed as a complementary structural diagnostic.

## Pipeline overview

```text
GitHub code dataset
        |
        v
Source files grouped by language
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
   Tree-sitter AST      Joern + Neo4j      LLM annotation
   token labels         static concepts     semantic concepts
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                  Token-aligned concept labels
                            |
                            v
                 Transformer hidden activations
                            |
                            v
                     SAE feature activations
                            |
                            v
              PMI / F1 / TCF + AST-purity scores
                            |
                            v
                 Candidate labels for SAE features
```

## Implemented components

### 1. Code dataset collection

[`src/dataset/load_github_code.py`](src/dataset/load_github_code.py) streams [`codeparrot/github-code`](https://huggingface.co/datasets/codeparrot/github-code), filters samples by language and line count, and writes individual source files grouped by language.

The supported file extensions are defined in [`data/languages.json`](data/languages.json). The current map covers 21 languages, including Python, C, C++, Java, JavaScript, TypeScript, Go, Rust, and several scripting languages.

### 2. Concept annotation

Two complementary annotation routes are available.

#### Static analysis with Joern

[`src/parser/parser.py`](src/parser/parser.py) builds a Code Property Graph (CPG) with Joern, exports it as Neo4j CSV, and imports the graph into a local Neo4j instance.

[`src/parser/concept_extractor.py`](src/parser/concept_extractor.py) runs Cypher-based extractors over that graph. Three C++ concepts are currently registered:

- `fixed_value`: local variables or parameters that are not reassigned;
- `future_mutated_variable`: declarations whose value is written later;
- `gatherer`: variables that accumulate state inside a loop.

The extractors emit JSONL annotations with source paths and character spans.

#### Semantic annotation with an instruction-tuned LLM

[`src/llm_annotator/annotate_data.py`](src/llm_annotator/annotate_data.py) prompts a Hugging Face causal language model to label code using controlled taxonomies. It validates the generated JSON, locates the reported snippets in the source, and writes normalized span annotations to JSONL.

Four prompt configurations are implemented:

- `idioms_and_structures`;
- `design_pattern`;
- `algorithmic_paradigm`;
- `programming_paradigm`.

The prompt builder in [`src/llm_annotator/build_prompt.py`](src/llm_annotator/build_prompt.py) generates taxonomy-aware instructions and few-shot examples from the concept files under [`data/concepts/`](data/concepts/).

### 3. AST labelling

[`src/parser/ast_labelling.py`](src/parser/ast_labelling.py) parses Python, C++, JavaScript, and Java with Tree-sitter. Each model token is associated with the hierarchy of named AST nodes it intersects. These labels are later used to estimate whether a feature is specific to a syntactic construct.

### 4. Token alignment

[`src/feature_labelling/token_labels.py`](src/feature_labelling/token_labels.py) converts Joern or LLM span annotations into a token-level dataframe using tokenizer offset mappings. Each labelled token stores its source path, language, token index, annotations, and concept names.

### 5. SAE activation extraction

[`src/SAE/encode_sae.py`](src/SAE/encode_sae.py) loads a TransformerLens-compatible model and an SAE through `sae-lens`, captures a selected residual-stream hook, and records every positive SAE activation for every source token.

It supports:

- a pretrained SAE release and ID, or a local SAE checkpoint;
- a selected model layer and hook point;
- single-layer exports to Parquet, CSV, or pickle;
- multi-layer extraction with one model forward pass per file;
- CPU or CUDA execution.

### 6. Feature-concept scoring

[`src/feature_labelling/feature_pmi.py`](src/feature_labelling/feature_pmi.py) builds sparse token-feature and token-concept matrices and computes feature/concept association metrics.

[`src/feature_labelling/tcf_script.py`](src/feature_labelling/tcf_script.py) connects the main stages across every model layer. It caches SAE activations, selects sufficiently frequent concepts, ranks candidate feature/concept pairs, computes TCF metrics, and exports AST-purity results.

## Repository structure

```text
.
├── data/
│   ├── concepts/               # Hierarchical concept taxonomies
│   └── languages.json          # Language-to-extension mapping
├── output/                     # Example/intermediate research artifacts
├── pages/                      # Saved pages used to build taxonomies
├── src/
│   ├── SAE/                    # SAE activation extraction
│   ├── dataset/                # Code corpus collection
│   ├── feature_labelling/      # Token alignment and feature metrics
│   ├── llm_annotator/          # Prompt construction and LLM labelling
│   ├── parser/                 # Tree-sitter and Joern analysis
│   └── scrapping/              # Taxonomy collection scripts
├── .env.example                # Neo4j configuration template
└── run.sh                      # Working command examples
```

## Installation

There is not yet a pinned dependency or packaging file. A development environment currently needs Python 3.10+ and the libraries imported by the pipeline:

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  torch transformers transformer-lens sae-lens datasets \
  pandas pyarrow numpy scipy scikit-learn tqdm \
  tree-sitter tree-sitter-python tree-sitter-cpp \
  tree-sitter-javascript tree-sitter-java \
  neo4j python-dotenv beautifulsoup4 requests
```

Model and dataset downloads require access to Hugging Face. Some models may also require accepting their licence and authenticating with a Hugging Face token. GPU execution is strongly recommended for LLM annotation and multi-layer SAE encoding.

### Optional Joern/Neo4j setup

The static-analysis route additionally requires Java, [Joern](https://docs.joern.io/installation/), and [Neo4j](https://neo4j.com/docs/operations-manual/current/installation/).

On macOS:

```bash
brew install joern neo4j
brew services start neo4j
```

Create the local configuration file:

```bash
cp .env.example .env
```

Then set:

```dotenv
NEO4J_PASSWORD="your-password"
NEO4J_IMPORT_DIR="/path/to/neo4j/import"
```

The Neo4j database is expected at `bolt://localhost:7687` with the user `neo4j`.

## Usage

Run commands from the repository root so that `src` imports and relative data paths resolve correctly.

### Build a code corpus

```bash
python -m src.dataset.load_github_code \
  --num_samples 1000 \
  --data_path data/code \
  --languages Python C++ Java JavaScript \
  --min-code-length 10 \
  --max-code-length 300
```

This creates language directories such as `data/code/Python/` and `data/code/C++/`.

### Annotate code with an LLM

```bash
python -m src.llm_annotator.annotate_data \
  --code-dir data/code \
  --output-path output/annotations/idioms.jsonl \
  --concept idioms_and_structures \
  --model-name google/gemma-4-E4B-it \
  --max-new-tokens 2048
```

Use `--limit N` for a small experimental run. Annotation errors are preserved as JSONL records instead of stopping the full dataset job.

### Run the Joern annotation route

After configuring `.env` and starting Neo4j:

```bash
python -m src.parser.parser
python -m src.parser.concept_extractor
```

The current entry points use `data/code/C++/` and write graph data under `output/graph/cpp/` and annotations under `output/annotations/`.

### Align annotations with tokens

```bash
python -m src.feature_labelling.token_labels \
  --annotations-path output/annotations/idioms.jsonl \
  --annotation-type llm \
  --tokenizer-name google/gemma-2-2b \
  --output-path output/idiom_token_labels.parquet
```

Set `--annotation-type joern` for the static-analysis JSONL format.

### Extract SAE activations for one layer

```bash
python -m src.SAE.encode_sae \
  --code-dir data/code \
  --output-path output/sae_activations.parquet \
  --model-name google/gemma-2-2b \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_20/width_16k/canonical \
  --layer 20 \
  --hook-point hook_resid_post \
  --method transformer_lens \
  --device cuda \
  --dtype float16
```

### Compute labels across layers

The integrated experiment consumes concept annotations and AST labels, encodes missing SAE layers, and writes per-layer metrics:

```bash
python -m src.feature_labelling.tcf_script \
  --device cuda \
  --dtype float32 \
  --model-name google/gemma-2-2b \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id 'layer_{}/width_16k/canonical' \
  --code-dir data/code \
  --annotations-path output/annotations/idioms.jsonl \
  --annotation-type llm \
  --ast-path output/ast_labels.parquet \
  --num-act-threshold 10 \
  --output-dir output/experiments
```

The script maintains a per-layer SAE cache under `SAE/` inside the selected output directory. This avoids recomputing layers that completed successfully in an earlier run.

## Data formats

### Span annotations (`.jsonl`)

LLM records contain one source file and its labelled spans:

```json
{
  "concept": "idioms_and_structures",
  "language": "Python",
  "path": "/absolute/path/to/sample.py",
  "spans": [
    {
      "start_line": 2,
      "end_line": 3,
      "text": "...",
      "match_method": "exact_bounds",
      "annotations": [
        {"label": "Guard clauses", "confidence": "high"}
      ]
    }
  ]
}
```

Joern records instead contain a concept name and an exact character range (`span_start`, `span_end`).

### Token labels (`.parquet`)

Important columns include `path`, `language`, `token_index`, `token_id`, `token`, `annotations`, and `concepts`.

### SAE activations (`.parquet`)

Each row represents one token. `feature_ids` contains the active SAE feature indices, sorted by decreasing activation, and `feature_activations` contains the corresponding values.

### Feature metrics (`.parquet`)

The integrated pipeline writes one TCF result and one AST-purity result per model layer under the experiment output directory.

## Research use

This repository is an experimental research artifact rather than a production labelling service. When reporting results, record the code corpus, tokenizer, base model, layer and hook point, SAE release and ID, annotation source, taxonomy version, and scoring thresholds. These choices directly affect the resulting feature labels.
