# LoRA Fine-tuning of Small Language Models

Production-grade implementation of LoRA (Low-Rank Adaptation) fine-tuning for GPT-2 on multiple text datasets.

## Overview

This project demonstrates parameter-efficient fine-tuning using PEFT (Parameter Efficient Fine-Tuning) library with LoRA adapters.

- **Model:** GPT-2 (124M parameters)
- **Method:** LoRA (r=8, alpha=16)
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, WandB

## Repository Structure
```
lora-slm-finetuning/
├── configs/             # Configuration files
│   ├── datasets.yaml    # Dataset selection and split info
│   └── training_config.yaml # Hyperparameters
├── src/                 # Source code
│   ├── train.py         # Main training script
│   ├── checkpoint_train.py # Resume training with advanced metrics
│   ├── inference.py     # Text generation (CLI & Batch)
│   ├── evaluate.py      # Perplexity calculation
│   ├── data_loader.py   # Dataset processing
│   └── utils.py         # Logging, seeding, metrics
├── notebooks/           # Jupyter notebook with my experimentation
│   └── finetune.ipynb   # finetuned on colab
├── docs/                # Project documentation and reports
│   ├── report.tex       # Latex research report
│   └── gpt_finetine.pdf # Compiled report PDF
├── outputs/             # Saved weights, metrics and checkpoints
├── README.md            # Project overview
├── documentation.md     # Technical details
└── requirements.txt     # Dependencies
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd lora-slm-finetuning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Optional: Install as a package*
   ```bash
   pip install -e .
   ```

3. **Weights & Biases Login:**
   ```bash
   wandb login
   ```

## Usage

### Training
Run the standard training loop:
```bash
python src/train_enhanced.py
```

### Inference
Run interactive chat-like inference:
```bash
python src/inference.py --lora_path outputs/final_model
```

Run batch inference from a file:
```bash
python src/inference.py --lora_path outputs/final_model --mode batch --input_file prompts.txt --output_file results.txt
```

### Evaluation
Calculate perplexity on the held-out test set:
```bash
python src/evaluate.py --lora_path outputs/final_model
```

## Configuration

Modify `configs/training_config.yaml` to change hyperparameters like `learning_rate`, `epochs`, or LoRA rank.
Modify `configs/datasets.yaml` to add/remove datasets or change sampling sizes.

## Results & Reproducibility

See `documentation.md` for detailed reproduction steps and technical explanations.
All random seeds are set to `42` by default.