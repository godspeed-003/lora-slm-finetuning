"""
Data loading and preprocessing utilities
"""
import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_datasets(datasets_config, tokenizer, max_length=512):
    """
    Load and prepare multiple datasets for training
    
    Args:
        datasets_config: Dict with dataset configurations
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        train_dataset, eval_dataset
    """
    all_datasets = []
    max_samples = datasets_config.get('max_samples_per_dataset', 50000)
    
    for dataset_info in datasets_config['datasets']:
        try:
            print(f"\nLoading {dataset_info['name']}...")
            
            # Load dataset
            if 'config' in dataset_info and dataset_info['config'] != 'default':
                dataset = load_dataset(
                    dataset_info['name'],
                    dataset_info['config'],
                    split=dataset_info['split']
                )
            else:
                dataset = load_dataset(
                    dataset_info['name'],
                    split=dataset_info['split']
                )
            
            # Sample if too large
            if len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
            
            # Extract text field
            text_field = dataset_info.get('text_field', 'text')
            
            # Handle different dataset structures
            if text_field == 'dialog':  # daily_dialog special case
                dataset = dataset.map(
                    lambda x: {'text': ' '.join(x['dialog'])},
                    remove_columns=dataset.column_names
                )
            elif text_field in dataset.column_names:
                dataset = dataset.map(
                    lambda x: {'text': x[text_field]},
                    remove_columns=dataset.column_names
                )
            
            # Tokenize
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors=None
                )
            
            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                desc=f"Tokenizing {dataset_info['name']}"
            )
            
            # Add labels (for causal LM, labels = input_ids)
            tokenized = tokenized.map(
                lambda x: {'labels': x['input_ids']},
                batched=True
            )
            
            all_datasets.append(tokenized)
            print(f"✓ Loaded {len(tokenized)} samples from {dataset_info['name']}")
            
        except Exception as e:
            print(f"✗ Error loading {dataset_info['name']}: {e}")
            continue
    
    # Concatenate all datasets
    if not all_datasets:
        raise ValueError("No datasets loaded successfully!")
    
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"\n✓ Total samples: {len(combined_dataset)}")
    
    # Train/validation split
    val_split = datasets_config.get('validation_split', 0.1)
    split_dataset = combined_dataset.train_test_split(
        test_size=val_split,
        seed=42
    )
    
    return split_dataset['train'], split_dataset['test']

def get_tokenizer(model_name):
    """Load tokenizer and set padding token"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer