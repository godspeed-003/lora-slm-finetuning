"""
Data loading and preprocessing utilities for LoRA fine-tuning.
"""
import yaml
import logging
from typing import Dict, Tuple, Any, List
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer

# Try to import from utils if available, otherwise setup basic logging
try:
    from utils import setup_logging
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
        
    Returns:
        Dict: Parsed configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def prepare_datasets(
    datasets_config: Dict[str, Any], 
    tokenizer: PreTrainedTokenizer, 
    max_length: int = 512
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare multiple datasets for training.
    
    Args:
        datasets_config (Dict): Configuration dictionary for datasets.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        max_length (int): Maximum sequence length for tokenization.
    
    Returns:
        Tuple[Dataset, Dataset]: Train and validation datasets.
    """
    all_datasets = []
    max_samples = datasets_config.get('max_samples_per_dataset', 50000)
    
    for dataset_info in datasets_config.get('datasets', []):
        dataset_name = dataset_info['name']
        try:
            logger.info(f"Loading {dataset_name}...")
            
            # Load dataset
            load_args = {
                'path': dataset_name,
                'split': dataset_info['split']
            }
            if 'config' in dataset_info and dataset_info['config'] != 'default':
                load_args['name'] = dataset_info['config']
                
            dataset = load_dataset(**load_args)
            
            # Sample if too large
            if len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
                logger.info(f"Sampled {max_samples} from {dataset_name}")
            
            # Extract text field
            text_field = dataset_info.get('text_field', 'text')
            
            # Handle different dataset structures
            if text_field == 'dialog':  # daily_dialog special case
                dataset = dataset.map(
                    lambda x: {'text': ' '.join(x['dialog'])},
                    remove_columns=dataset.column_names,
                    desc=f"Processing {dataset_name}"
                )
            elif text_field in dataset.column_names:
                dataset = dataset.map(
                    lambda x: {'text': x[text_field]},
                    remove_columns=dataset.column_names,
                    desc=f"Processing {dataset_name}"
                )
            else:
                logger.warning(f"Field '{text_field}' not found in {dataset_name}. Skipping...")
                continue
            
            # Tokenize function
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors=None
                )
            
            # Apply tokenization
            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                desc=f"Tokenizing {dataset_name}"
            )
            
            # Add labels (for causal LM, labels = input_ids)
            tokenized = tokenized.map(
                lambda x: {'labels': x['input_ids']},
                batched=True,
                desc=f"Adding labels for {dataset_name}"
            )
            
            all_datasets.append(tokenized)
            logger.info(f"✓ Loaded {len(tokenized)} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"✗ Error loading {dataset_name}: {e}")
            continue
    
    # Concatenate all datasets
    if not all_datasets:
        raise ValueError("No datasets loaded successfully!")
    
    combined_dataset = concatenate_datasets(all_datasets)
    logger.info(f"✓ Total combined samples: {len(combined_dataset)}")
    
    # Train/validation split
    val_split = datasets_config.get('validation_split', 0.1)
    split_dataset = combined_dataset.train_test_split(
        test_size=val_split,
        seed=42
    )
    
    logger.info(f"Train set: {len(split_dataset['train'])} / Eval set: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']

def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer and set padding token.
    
    Args:
        model_name (str): Name or path of the model.
        
    Returns:
        PreTrainedTokenizer: Initialized tokenizer.
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer