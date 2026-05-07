"""
Configuration for WAF Model Training

Centralized configuration management for hyperparameters, paths, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data paths
    data_dir: str = "data/processed"
    model_save_dir: str = "models/checkpoints"
    
    # Model selection
    model_name: str = "distilbert-base-uncased"  # Can be distilbert, bert-base, or MiniLM
    num_labels: int = 5
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 8
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Optimization
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    
    # Hardware
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    use_fp16: bool = False  # Mixed precision training
    
    # Label mapping
    label_map: dict = field(default_factory=lambda: {
        "SAFE": 0,
        "SQLI": 1,
        "XSS": 2,
        "PATH_TRAVERSAL": 3,
        "COMMAND_INJECTION": 4
    })
    
    id2label: dict = field(default_factory=lambda: {
        0: "SAFE",
        1: "SQLI",
        2: "XSS",
        3: "PATH_TRAVERSAL",
        4: "COMMAND_INJECTION"
    })
    
    def __post_init__(self):
        # Create directories if they don't exist
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default configuration instance
config = TrainingConfig()


if __name__ == "__main__":
    print("Training Configuration:")
    print("="*60)
    for field_name, field_value in config.__dict__.items():
        print(f"{field_name}: {field_value}")