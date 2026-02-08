"""
Enhanced training script with comprehensive W&B metrics logging
"""
import os
import yaml
import torch
import wandb
import time
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from data_loader import load_config, prepare_datasets, get_tokenizer
import numpy as np

class DetailedMetricsCallback(TrainerCallback):
    """Custom callback for detailed metrics logging"""
    
    def __init__(self):
        self.step_start_time = None
        self.ema_loss = None
        self.ema_alpha = 0.9
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # Calculate step time
            step_time = time.time() - self.step_start_time
            
            # Get current loss
            current_loss = state.log_history[-1].get('loss', 0)
            
            # Calculate EMA loss
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * current_loss
            
            # Calculate perplexity
            ppl = math.exp(current_loss) if current_loss < 100 else float('inf')
            
            # Calculate bits per token (bpt = loss / ln(2))
            bpt = current_loss / math.log(2)
            
            # Calculate throughput
            # Effective batch size = per_device_batch_size * gradient_accumulation_steps * num_gpus
            effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            tokens_per_batch = effective_batch_size * args.max_steps  # Approximate
            throughput = tokens_per_batch / step_time if step_time > 0 else 0
            
            # Log additional metrics
            wandb.log({
                'train/token_loss': current_loss,
                'train/sequence_loss': current_loss,  # Same for causal LM
                'train/ppl': ppl,
                'train/bpt': bpt,
                'train/step_time': step_time,
                'train/ema_token_loss': self.ema_loss,
                'train/throughput_tokens_per_sec': throughput,
                'step': state.global_step,
            })

def compute_detailed_metrics(eval_preds):
    """
    Compute comprehensive evaluation metrics
    """
    predictions, labels = eval_preds
    
    # Handle tuple output from model (logits, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Remove padding tokens (usually -100)
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    # Calculate token-level loss
    loss_fct = torch.nn.CrossEntropyLoss()
    token_loss = loss_fct(
        torch.tensor(predictions),
        torch.tensor(labels)
    ).item()
    
    # Calculate perplexity
    ppl = math.exp(token_loss) if token_loss < 100 else float('inf')
    
    # Calculate bits per token
    bpt = token_loss / math.log(2)
    
    # Token count
    n_tokens = len(labels)
    
    return {
        'token_loss': token_loss,
        'sequence_loss': token_loss,  # Same for causal LM
        'ppl': ppl,
        'bpt': bpt,
        'n_tokens': n_tokens,
    }

def train_enhanced(
    model_config_path='configs/training_config.yaml',
    dataset_config_path='configs/datasets.yaml'
):
    """Enhanced training function with comprehensive metrics"""
    
    # Load configurations
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    
    # Initialize W&B with detailed config
    run_config = {
        'model_name': model_config['model']['name'],
        'lora_r': model_config['lora']['r'],
        'lora_alpha': model_config['lora']['lora_alpha'],
        'learning_rate': model_config['training']['learning_rate'],
        'batch_size': model_config['training']['per_device_train_batch_size'],
        'grad_accum_steps': model_config['training']['gradient_accumulation_steps'],
        'epochs': model_config['training']['num_train_epochs'],
        'max_length': model_config['data']['max_length'],
        'fp16': model_config['training']['fp16'],
        'seed': model_config['training']['seed'],
        'num_datasets': len(dataset_config['datasets']),
    }
    
    wandb.init(
        project=model_config['wandb']['project'],
        entity=model_config['wandb'].get('entity'),
        tags=model_config['wandb']['tags'],
        config=run_config
    )
    
    # Log important hyperparameters
    wandb.config.update({
        'train/optimizer': 'AdamW',
        'train/precision': 'fp16' if model_config['training']['fp16'] else 'fp32',
        'train/lora_rank': model_config['lora']['r'],
        'train/grad_accum_steps': model_config['training']['gradient_accumulation_steps'],
        'train/weight_decay': model_config['training']['weight_decay'],
    })
    
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
    
    # Log dataset info
    wandb.log({
        'dataset/train_samples': len(train_dataset),
        'dataset/eval_samples': len(eval_dataset),
        'dataset/num_datasets': len(dataset_config['datasets']),
    })
    
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
    trainable_params = model.print_trainable_parameters()
    
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
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_first_step=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize custom callback
    metrics_callback = DetailedMetricsCallback()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[metrics_callback],
        # compute_metrics=compute_detailed_metrics,  # Can add but slows training
    )
    
    # Train
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    total_time = time.time() - start_time
    
    # Log final training metrics
    wandb.log({
        'train/total_time_hours': total_time / 3600,
        'train/samples_per_sec': len(train_dataset) * training_args.num_train_epochs / total_time,
    })
    
    # Save final model
    print("\n💾 Saving final model...")
    output_path = os.path.join(model_config['training']['output_dir'], 'final_model')
    trainer.save_model(output_path)
    
    # Save to W&B (optional)
    # wandb.save(f"{output_path}/*")
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log evaluation metrics with val/ prefix
    val_metrics = {
        'val/token_loss': eval_results['eval_loss'],
        'val/ppl': math.exp(eval_results['eval_loss']),
        'val/bpt': eval_results['eval_loss'] / math.log(2),
        'val/n_tokens': len(eval_dataset) * model_config['data']['max_length'],
        'val/n_sequences': len(eval_dataset),
    }
    wandb.log(val_metrics)
    
    print("\n✅ Final Results:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    wandb.finish()
    print("\n✅ Training complete!")
    
    return trainer, eval_results

if __name__ == "__main__":
    train_enhanced()