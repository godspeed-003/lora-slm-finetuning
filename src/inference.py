"""
Inference script for LoRA fine-tuned models.
Supports interactive mode and batch processing from a file.
"""
import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils import setup_logging, set_seed

logger = setup_logging()

def load_model(base_model_path, lora_adapter_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load base model and apply LoRA adapter.
    
    Args:
        base_model_path (str): Path to base model.
        lora_adapter_path (str): Path to LoRA adapter contents.
        device (str): Device to load model on.
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device
    )
    
    logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    """
    Generate text from a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def interactive_mode(model, tokenizer, args):
    """Run interactive CLI for generation."""
    print("\n--- Interactive Inference Mode ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        prompt = input("Prompt > ")
        if prompt.lower() in ['exit', 'quit']:
            break
            
        try:
            generated = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                device=model.device
            )
            print(f"\nResponse:\n{generated}\n")
            print("-" * 50)
        except Exception as e:
            logger.error(f"Generation failed: {e}")

def batch_mode(model, tokenizer, args):
    """Run batch inference from a file."""
    if not args.input_file or not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} not found.")
        return

    logger.info(f"Processing prompts from {args.input_file}...")
    
    with open(args.input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    for prompt in prompts:
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=model.device
        )
        results.append(f"Prompt: {prompt}\nResponse: {generated}\n{'-'*50}\n")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.writelines(results)
        logger.info(f"Results saved to {args.output_file}")
    else:
        for res in results:
            print(res)

def main():
    parser = argparse.ArgumentParser(description="LoRA Inference Script")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name or path")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--mode", type=str, choices=['interactive', 'batch'], default='interactive', help="Inference mode")
    parser.add_argument("--input_file", type=str, help="Input file for batch mode (one prompt per line)")
    parser.add_argument("--output_file", type=str, help="Output file for batch mode results")
    parser.add_argument("--max_length", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        model, tokenizer = load_model(args.base_model, args.lora_path, device)
        
        if args.mode == 'interactive':
            interactive_mode(model, tokenizer, args)
        else:
            batch_mode(model, tokenizer, args)
            
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")

if __name__ == "__main__":
    main()
