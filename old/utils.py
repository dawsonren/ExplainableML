"""
Utility functions for training and evaluation.
"""
import torch

def set_seed(seed: int = 42):
    """Ensure reproducibility as much as possible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False