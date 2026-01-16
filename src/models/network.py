"""
Neural Network Architectures for Block Blast AI.

This module provides the neural network models for policy and value estimation.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class BlockBlastNetwork(nn.Module):
    """
    Neural network for Block Blast that processes board and piece information.
    
    Architecture:
    1. Convolutional encoder for the board + piece masks
    2. Fully connected layers
    3. Separate policy and value heads
    """
    
    def __init__(
        self,
        board_size: int = 8,
        num_pieces: int = 3,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        fc_hidden: Tuple[int, ...] = (512, 256),
        action_space_size: int = 192,
        use_residual: bool = True,
        use_batch_norm: bool = True,
    ):
        """
        Initialize the network.
        
        Args:
            board_size: Size of the game board (8x8)
            num_pieces: Number of pieces per turn (3)
            conv_channels: Channels for each conv layer
            fc_hidden: Hidden layer sizes for FC layers
            action_space_size: Total number of actions (3 * 8 * 8 = 192)
            use_residual: Whether to use residual connections
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.board_size = board_size
        self.num_pieces = num_pieces
        self.action_space_size = action_space_size
        
        # Input: board (1 channel) + piece masks (3 channels) = 4 channels
        input_channels = 1 + num_pieces
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        for i, out_channels in enumerate(conv_channels):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            
            # Add residual block after first conv
            if use_residual and i > 0:
                conv_layers.append(ResidualBlock(out_channels))
            
            in_channels = out_channels
        
        self.conv_encoder = nn.Sequential(*conv_layers)
        
        # Calculate size after conv layers
        conv_output_size = conv_channels[-1] * board_size * board_size
        
        # Build fully connected layers
        fc_layers = []
        fc_input = conv_output_size
        for hidden_size in fc_hidden:
            fc_layers.append(nn.Linear(fc_input, hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.1))
            fc_input = hidden_size
        
        self.fc_encoder = nn.Sequential(*fc_layers)
        
        # Policy head - outputs logits for each action
        self.policy_head = nn.Sequential(
            nn.Linear(fc_hidden[-1], 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size),
        )
        
        # Value head - outputs single scalar value estimate
        self.value_head = nn.Sequential(
            nn.Linear(fc_hidden[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            # Xavier uniform for linear layers (works well for ReLU)
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Kaiming for conv layers
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            board: Board state, shape (batch, 8, 8) or (batch, 1, 8, 8)
            pieces: Piece masks, shape (batch, 3, 8, 8)
            action_mask: Valid actions mask, shape (batch, 192)
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # Ensure board has channel dimension
        if board.dim() == 3:
            board = board.unsqueeze(1)  # (batch, 1, 8, 8)
        
        # Concatenate board and pieces
        x = torch.cat([board, pieces], dim=1)  # (batch, 4, 8, 8)
        
        # Convolutional encoding
        x = self.conv_encoder(x)  # (batch, channels, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, channels * 64)
        
        # Fully connected encoding
        x = self.fc_encoder(x)  # (batch, fc_hidden[-1])
        
        # Policy and value heads
        action_logits = self.policy_head(x)  # (batch, 192)
        value = self.value_head(x)  # (batch, 1)
        
        # Apply action mask (set invalid actions to large negative value)
        if action_mask is not None:
            # Convert mask to float and invert (True = valid -> 0, False = invalid -> -inf)
            mask_value = torch.where(
                action_mask.bool(),
                torch.zeros_like(action_logits),
                torch.full_like(action_logits, float('-inf'))
            )
            action_logits = action_logits + mask_value
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Properly masks invalid actions:
        - Invalid actions set to -inf before softmax (zero probability)
        - Entropy computed ONLY over valid actions for better gradients
        - 100% of sampled actions are guaranteed valid

        Args:
            board: Board state
            pieces: Piece masks
            action_mask: Valid actions mask (1=valid, 0=invalid)
            action: Optional action to evaluate (if None, sample new action)
            deterministic: Whether to select action deterministically

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(board, pieces, action_mask)

        # Compute probabilities (invalid actions have -inf logits -> 0 probability)
        probs = F.softmax(logits, dim=-1)

        # Create distribution - only valid actions can be sampled
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        # Log probability of selected action
        log_prob = dist.log_prob(action)

        # Compute entropy ONLY over valid actions for cleaner gradient signal
        entropy = self._masked_entropy(probs, action_mask)

        return action, log_prob, entropy, value

    def _masked_entropy(
        self,
        probs: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy only over valid actions.

        Standard entropy includes near-zero probabilities from masked actions,
        which adds noise to gradients. This computes entropy only over the
        valid action distribution for cleaner learning signal.

        Args:
            probs: Action probabilities after softmax (batch, num_actions)
            action_mask: Valid action mask (batch, num_actions), 1=valid

        Returns:
            Entropy tensor (batch,)
        """
        mask = action_mask.bool()

        # Mask out invalid action probabilities and renormalize
        masked_probs = probs * mask.float()
        prob_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        normalized_probs = masked_probs / prob_sum

        # Entropy: -sum(p * log(p)) only where mask is True
        log_probs = torch.log(normalized_probs.clamp(min=1e-10))
        entropy = -(normalized_probs * log_probs * mask.float()).sum(dim=-1)

        return entropy

    def get_value(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
    ) -> torch.Tensor:
        """Get only the value estimate."""
        _, value = self.forward(board, pieces)
        return value


class ActorCritic(nn.Module):
    """
    Separate actor and critic networks for PPO.
    
    This can sometimes work better than a shared network.
    """
    
    def __init__(
        self,
        board_size: int = 8,
        num_pieces: int = 3,
        conv_channels: Tuple[int, ...] = (32, 64, 64),
        fc_hidden: int = 256,
        action_space_size: int = 192,
    ):
        """Initialize actor-critic networks."""
        super().__init__()
        
        self.board_size = board_size
        self.num_pieces = num_pieces
        self.action_space_size = action_space_size
        
        input_channels = 1 + num_pieces
        
        # Shared feature extractor (lighter weight)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        conv_output_size = conv_channels[-1] * board_size * board_size
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(conv_output_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, action_space_size),
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _encode(self, board: torch.Tensor, pieces: torch.Tensor) -> torch.Tensor:
        """Encode observation."""
        if board.dim() == 3:
            board = board.unsqueeze(1)
        x = torch.cat([board, pieces], dim=1)
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self._encode(board, pieces)
        
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        if action_mask is not None:
            mask_value = torch.where(
                action_mask.bool(),
                torch.zeros_like(logits),
                torch.full_like(logits, float('-inf'))
            )
            logits = logits + mask_value
        
        return logits, value
    
    def get_action_and_value(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value with proper masking."""
        logits, value = self.forward(board, pieces, action_mask)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)

        # Compute entropy only over valid actions
        entropy = self._masked_entropy(probs, action_mask)

        return action, log_prob, entropy, value

    def _masked_entropy(
        self,
        probs: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy only over valid actions."""
        mask = action_mask.bool()
        masked_probs = probs * mask.float()
        prob_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        normalized_probs = masked_probs / prob_sum
        log_probs = torch.log(normalized_probs.clamp(min=1e-10))
        entropy = -(normalized_probs * log_probs * mask.float()).sum(dim=-1)
        return entropy

    def get_value(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
    ) -> torch.Tensor:
        """Get value estimate."""
        features = self._encode(board, pieces)
        return self.critic(features).squeeze(-1)


if __name__ == "__main__":
    # Test the networks
    print("Testing BlockBlastNetwork...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    network = BlockBlastNetwork().to(device)
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create sample inputs
    batch_size = 4
    board = torch.rand(batch_size, 8, 8).to(device)
    pieces = torch.rand(batch_size, 3, 8, 8).to(device)
    action_mask = torch.ones(batch_size, 192).to(device)
    
    # Forward pass
    logits, value = network(board, pieces, action_mask)
    print(f"Logits shape: {logits.shape}")  # (4, 192)
    print(f"Value shape: {value.shape}")  # (4,)
    
    # Get action and value
    action, log_prob, entropy, value = network.get_action_and_value(
        board, pieces, action_mask
    )
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Entropy: {entropy}")
    print(f"Value: {value}")
    
    print("\nTesting ActorCritic...")
    ac = ActorCritic().to(device)
    print(f"ActorCritic parameters: {sum(p.numel() for p in ac.parameters()):,}")
    
    logits, value = ac(board, pieces, action_mask)
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
