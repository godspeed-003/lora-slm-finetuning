"""
Utility functions for LoRA fine-tuning project.
Includes logging setup, seeding, and metric calculations.
"""
import os
import random
import logging
import sys
import numpy as np
import torch
import math

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configure logging to both console and file.
    
    Args:
        log_file (str, optional): Path to the log file.
        level (int): Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss safely.
    
    Args:
        loss (float): Cross-entropy loss value.
        
    Returns:
        float: Calculated perplexity.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def format_time(elapsed_time: float) -> str:
    """
    Format time in seconds to HH:MM:SS string.
    
    Args:
        elapsed_time (float): Time in seconds.
        
    Returns:
        str: Formatted time string.
    """
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        tuple: (trainable_params, all_params, trainable_percentage)
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    return trainable_params, all_param, 100 * trainable_params / all_param
