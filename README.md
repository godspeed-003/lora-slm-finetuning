# LoRA Fine-tuning of Small Language Models

Production-grade implementation of LoRA (Low-Rank Adaptation) fine-tuning for GPT-2 on multiple text datasets.

## Overview

This project demonstrates parameter-efficient fine-tuning using PEFT (Parameter Efficient Fine-Tuning) library with LoRA adapters.

**Model:** GPT-2 (124M parameters)  
**Method:** LoRA with rank=8, alpha=16  
**Trainable params:** 811,008 (0.65% of total)  
**Datasets:** 10 diverse text corpora  
**Training samples:** ~415k (after sampling)

## Datasets

We curated 10 text-only, non-reasoning datasets from Hugging Face:

1. **wikitext-103** - Wikipedia articles (encyclopedia knowledge)
2. **wikitext-2** - Smaller Wikipedia subset
3. **TinyStories** - Short fictional narratives
4. **ag_news** - News classification dataset
5. **yelp_review_full** - Restaurant reviews
6. **amazon_polarity** - Product reviews
7. **cnn_dailymail** - News summarization
8. **xsum** - Extreme summarization
9. **squad** - Question-answering contexts
10. **imdb** - Movie reviews

**Total:** ~415k training samples, ~46k validation samples

## Repository Structure
```
lora-slm-finetuning/
├── configs/
│   ├── datasets.yaml         # Dataset configuration
│   └── training_config.yaml  # Training hyperparameters
├── src/
│   ├── train.py             # Basic training script
│   ├── train_enhanced.py    # Enhanced with detailed W&B metrics
│   ├── data_loader.py       # Dataset loading utilities
│   └── utils.py             # Helper functions
├── outputs/                 # Training checkpoints
├── README.md
├── documentation.md
└── requirements.txt
```

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Weights & Biases
```bash
wandb login
```

## Training

### Basic Training
```bash
python src/train.py
```

### Enhanced Training (with detailed metrics)
```bash
python src/train_enhanced.py
```

## Configuration

### LoRA Parameters
```yaml
lora:
  r: 8                    # LoRA rank
  lora_alpha: 16          # Scaling factor
  lora_dropout: 0.1       # Dropout
  target_modules: ["c_attn", "c_proj"]
```

### Training Hyperparameters
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  fp16: true
  max_length: 512
```

## Tracked Metrics

All experiments logged to W&B:

**Training:**
- Token loss, perplexity (PPL), bits-per-token (BPT)
- Learning rate, gradient norm
- EMA token loss
- Throughput (tokens/sec)

**Validation:**
- Token loss, sequence loss
- Perplexity, bits-per-token
- Token/sequence counts

## Results

See W&B dashboard: [link]

## Hardware

- **GPU:** NVIDIA T4 (15GB VRAM)
- **Training time:** ~4 hours for 3 epochs
- **Platform:** Google Colab

## References

- PEFT Library: https://github.com/huggingface/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685