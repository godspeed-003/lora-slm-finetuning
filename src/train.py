"""
Main training script for LoRA fine-tuning
"""
import os
import yaml
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from data_loader import load_config, prepare_datasets, get_tokenizer
import numpy as np

def compute_metrics(eval_preds):
    """Compute perplexity and other metrics"""
    logits, labels = eval_preds
    
    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "token_loss": loss.item()
    }

def train(
    model_config_path='configs/training_config.yaml',
    dataset_config_path='configs/datasets.yaml'
):
    """Main training function"""
    
    # Load configurations
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    
    # Initialize W&B
    wandb.init(
        project=model_config['wandb']['project'],
        entity=model_config['wandb'].get('entity'),
        tags=model_config['wandb']['tags'],
        config={**model_config, **dataset_config}
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model_config['model']['name'])
    
    # Load datasets
    print("\nPreparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        dataset_config,
        tokenizer,
        max_length=model_config['data']['max_length']
    )
    
    # Load base model
    print(f"\nLoading model: {model_config['model']['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model']['name'],
        cache_dir=model_config['model'].get('cache_dir'),
        torch_dtype=torch.float16 if model_config['training']['fp16'] else torch.float32
    )
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=model_config['lora']['r'],
        lora_alpha=model_config['lora']['lora_alpha'],
        lora_dropout=model_config['lora']['lora_dropout'],
        target_modules=model_config['lora']['target_modules'],
        bias=model_config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_config['training']['output_dir'],
        num_train_epochs=model_config['training']['num_train_epochs'],
        per_device_train_batch_size=model_config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=model_config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=model_config['training']['gradient_accumulation_steps'],
        learning_rate=model_config['training']['learning_rate'],
        weight_decay=model_config['training']['weight_decay'],
        warmup_steps=model_config['training']['warmup_steps'],
        logging_steps=model_config['training']['logging_steps'],
        save_steps=model_config['training']['save_steps'],
        eval_steps=model_config['training']['eval_steps'],
        save_total_limit=model_config['training']['save_total_limit'],
        fp16=model_config['training']['fp16'],
        gradient_checkpointing=model_config['training']['gradient_checkpointing'],
        max_grad_norm=model_config['training']['max_grad_norm'],
        seed=model_config['training']['seed'],
        report_to="wandb",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,  # Add if needed
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(os.path.join(model_config['training']['output_dir'], 'final_model'))
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    wandb.finish()
    print("\n✅ Training complete!")

if __name__ == "__main__":
    train()