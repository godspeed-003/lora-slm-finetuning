# LoRA Fine-tuning Documentation

## Technical Overview

This repository implements **Low-Rank Adaptation (LoRA)** for fine-tuning Small Language Models (SLMs), specifically GPT-2, on a diverse set of text corpora. LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, significantly reducing the number of trainable parameters for downstream tasks.

### Key Concepts

- **Rank (r)**: The dimension of the low-rank matrices. We use `r=8`.
- **Alpha (lora_alpha)**: Scaling factor for the LoRA weights. We use `alpha=16`.
- **Target Modules**: The modules to apply LoRA to. We target `c_attn` (attention projection) and `c_proj` (output projection).

## Dataset Details

We utilize a curated mix of 10 datasets to ensure satisfactory performance across various domains:

| Dataset | Split | Samples Used | Description |
| :--- | :--- | :--- | :--- |
| **wikitext-103** | Train | ~10k | Encyclopedia articles |
| **TinyStories** | Train | ~10k | Simple narratives |
| **ag_news** | Train | ~10k | News classification |
| **yelp_review_full** | Train | ~10k | Sentiment/Reviews |
| **cnn_dailymail** | Train | ~10k | Summarization |
| **xsum** | Train | ~10k | Extreme summarization |
| **squad** | Train | ~10k | QA contexts |
| **imdb** | Train | ~10k | Movie reviews |

*Note: Datasets are sampled to `max_samples_per_dataset` (default 10,000) to ensure manageable training times on consumer hardware or Colab.*

## Script Usage Guide

### 1. Training (`src/train.py`)
Standard training script.
```bash
python src/train.py
```
- Uses `configs/training_config.yaml` and `configs/datasets.yaml`.
- Logs to Weights & Biases.

### 2. Checkpoint Training (`src/checkpoint_train.py`)
Resumes training from a specific checkpoint with enhanced metrics.
```bash
python src/checkpoint_train.py
```
- You may need to edit the `resume_checkpoint_path` default argument or calling code.

### 3. Evaluation (`src/evaluate.py`)
Calculates perplexity on the test set.
```bash
python src/evaluate.py --lora_path outputs/final_model --max_length 512
```

### 4. Inference (`src/inference.py`)
Generates text using the fine-tuned model.

**Interactive Mode:**
```bash
python src/inference.py --lora_path outputs/final_model --mode interactive
```

**Batch Mode:**
```bash
python src/inference.py --lora_path outputs/final_model --mode batch --input_file prompts.txt --output_file results.txt
```

## Reproducibility

To reproduce the results:
1. Install dependencies: `pip install -r requirements.txt`
2. Set the random seed to `42` (default in config).
3. Run `src/train.py`.
4. Run `src/evaluate.py` on the output.

The `src/utils.py` module handles `set_seed` to ensure `torch`, `numpy`, and `random` are deterministic.
