"""
WAF Configuration Module

Centralized configuration for the WAF engine.
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class WAFConfig:
    """Configuration for WAF engine."""
    
    # Model settings
    model_path: str = "models/checkpoints/best_model"
    
    # Decision thresholds
    high_confidence_threshold: float = 0.85  # Block immediately
    medium_confidence_threshold: float = 0.60  # Log with warning
    
    # Logging settings
    enable_logging: bool = True
    log_file: str = "logs/waf.jsonl"
    log_level: str = "INFO"
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_url_length: int = 2048
    max_header_size: int = 8192
    
    # Security settings
    block_on_high_confidence: bool = True
    allow_on_low_confidence: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_window: int = 300  # 5 minutes
    
    def __post_init__(self):
        # Create log directory
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)


# Default configuration
waf_config = WAFConfig()