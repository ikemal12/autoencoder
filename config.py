import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class TrainingConfig:
    # Data configuration
    data_dir: str = "data"
    cache_tensors: bool = True
    input_size: Tuple[int, int] = (144, 224)
    
    # Model hyperparameters
    latent_dim: int = 96
    use_attention: bool = True
    dropout_rate: float = 0.5
    residual_layers: int = 2
    filters: Tuple[int, ...] = (32, 64, 128)
    
    # Training configuration
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 2e-3
    weight_decay: float = 0.0025
    optimizer: str = 'Lion'
    
    # Loss function parameters
    alpha: float = 0.6  # MSE weight
    beta: float = 0.35  # SSIM weight
    gamma: float = 0.05  # Latent regularization weight
    
    # Logging and checkpointing
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    save_best_model: bool = True
    
    # Early stopping
    patience: int = 5
    
    def __post_init__(self):
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self, filename: str = "autoencoder_checkpoint.pth") -> str:
        """Generate full path for checkpoint file"""
        return os.path.join(self.checkpoint_dir, filename)
    
    def to_dict(self):
        """Convert configuration to a dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Create a global configuration instance
config = TrainingConfig()