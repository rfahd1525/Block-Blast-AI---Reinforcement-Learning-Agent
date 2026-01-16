"""
Base Agent class for Block Blast AI.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize base agent.
        
        Args:
            device: Torch device to use
        """
        self.device = device
        self.training = True
    
    @abstractmethod
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action given an observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to select action deterministically
            
        Returns:
            Tuple of (action, info_dict)
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent given a batch of experience.
        
        Args:
            batch: Batch of experience
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load from
        """
        pass
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.training = True
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False
    
    def to(self, device: torch.device) -> "BaseAgent":
        """Move agent to device."""
        self.device = device
        return self
