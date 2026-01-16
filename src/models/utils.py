"""
Model utility functions.
"""
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


def initialize_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize network weights using orthogonal initialization.
    
    Args:
        module: Network module to initialize
        gain: Gain for orthogonal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        step: Current step
        path: Path to save checkpoint
        metrics: Optional metrics to include
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
        
    Returns:
        Tuple of (epoch, step, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    metrics = checkpoint.get('metrics', None)
    
    return epoch, step, metrics


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String summary
    """
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(f"Total parameters: {count_parameters(model):,}")
    lines.append("")
    lines.append("Layers:")
    for name, module in model.named_modules():
        if name:
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                lines.append(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    return "\n".join(lines)


class EarlyStopping:
    """
    Early stopping to stop training when a monitored quantity has stopped improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether to look for decreasing or increasing metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class GradientClipper:
    """
    Gradient clipping utility.
    """
    
    def __init__(self, max_norm: float = 0.5):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum norm for gradient clipping
        """
        self.max_norm = max_norm
    
    def __call__(self, model: nn.Module) -> float:
        """
        Clip gradients and return the total norm.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm
        ).item()
