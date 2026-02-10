"""
Evaluation script for LoRA fine-tuned models.
Calculates perplexity and loss on a test dataset.
"""
import argparse
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import PeftModel
from torch.utils.data import DataLoader
from data_loader import load_config, prepare_datasets, get_tokenizer
from utils import setup_logging, set_seed, compute_perplexity

logger = setup_logging()

def evaluate(model, dataloader, device):
    """
    Evaluate model on dataloader.
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    perplexity = compute_perplexity(avg_loss)
    
    return avg_loss, perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA Model")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yaml", help="Path to dataset config")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = get_tokenizer(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device
    )
    
    logger.info(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.to(device)
    
    # Load datasets
    logger.info("Loading evaluation dataset...")
    dataset_config = load_config(args.dataset_config)
    # We only care about the test set here
    _, eval_dataset = prepare_datasets(dataset_config, tokenizer, max_length=args.max_length)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Evaluate
    logger.info(f"Starting evaluation on {len(eval_dataset)} samples...")
    avg_loss, perplexity = evaluate(model, dataloader, device)
    
    # Report
    logger.info("=" * 30)
    logger.info(f"RESULTS for {args.lora_path}")
    logger.info(f"Test Loss:       {avg_loss:.4f}")
    logger.info(f"Test Perplexity: {perplexity:.4f}")
    logger.info("=" * 30)

if __name__ == "__main__":
    main()
