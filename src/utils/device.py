"""Device utilities and configuration management."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device preference ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        torch.mps.manual_seed(seed)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": 1,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
    
    return info


def print_device_info() -> None:
    """Print information about available devices."""
    info = get_device_info()
    
    print("Device Information:")
    print("-" * 30)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    
    if info["cuda_available"]:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"Current CUDA Device: {info['cuda_current_device']}")
        print(f"CUDA Device Name: {info['cuda_device_name']}")
        print(f"CUDA Memory Allocated: {info['cuda_memory_allocated'] / 1024**3:.2f} GB")
        print(f"CUDA Memory Reserved: {info['cuda_memory_reserved'] / 1024**3:.2f} GB")
    
    print(f"Recommended Device: {get_device()}")


class ConfigManager:
    """Configuration manager using OmegaConf."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from file or create default."""
        if self.config_path and self.config_path.exists():
            config = OmegaConf.load(self.config_path)
        else:
            config = self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        default_config = {
            # Model configuration
            "model": {
                "type": "autoencoder",
                "in_channels": 3,
                "out_channels": 3,
                "hidden_channels": 64,
                "base_channels": 64,
                "depth": 4,
                "num_frames": 5,
            },
            
            # Data configuration
            "data": {
                "train_path": "data/raw/train",
                "val_path": "data/raw/val",
                "test_path": "data/raw/test",
                "batch_size": 8,
                "num_workers": 4,
                "frame_size": [256, 256],
                "temporal_window": 1,
                "noise_type": "gaussian",
                "noise_level": 0.1,
            },
            
            # Training configuration
            "training": {
                "num_epochs": 100,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "patience": 10,
                "save_best": True,
                "device": "auto",
            },
            
            # Loss configuration
            "loss": {
                "l1_weight": 1.0,
                "charbonnier_weight": 1.0,
                "perceptual_weight": 0.1,
                "temporal_weight": 0.1,
                "use_perceptual": True,
                "use_temporal": True,
            },
            
            # Paths
            "paths": {
                "save_dir": "checkpoints",
                "log_dir": "logs",
                "results_dir": "results",
                "assets_dir": "assets",
            },
            
            # Logging
            "logging": {
                "log_level": "INFO",
                "log_interval": 10,
                "save_interval": 5,
            },
            
            # Reproducibility
            "seed": 42,
        }
        
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key)
        except KeyError:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        OmegaConf.set(self.config, key, value)
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration (uses default if None)
        """
        save_path = Path(path) if path else self.config_path
        if save_path:
            OmegaConf.save(self.config, save_path)
    
    def merge(self, other_config: Union[Dict, DictConfig]) -> None:
        """Merge another configuration.
        
        Args:
            other_config: Configuration to merge
        """
        if isinstance(other_config, dict):
            other_config = OmegaConf.create(other_config)
        
        self.config = OmegaConf.merge(self.config, other_config)
    
    def print_config(self) -> None:
        """Print current configuration."""
        print(OmegaConf.to_yaml(self.config))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return OmegaConf.to_container(self.config, resolve=True)


def load_config(config_path: Union[str, Path]) -> ConfigManager:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration manager
    """
    return ConfigManager(config_path)


def create_config(
    model_type: str = "autoencoder",
    data_path: str = "data/raw",
    output_dir: str = "outputs",
    **kwargs
) -> ConfigManager:
    """Create configuration with custom parameters.
    
    Args:
        model_type: Type of model to use
        data_path: Path to data directory
        output_dir: Output directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration manager
    """
    config_manager = ConfigManager()
    
    # Set basic parameters
    config_manager.set("model.type", model_type)
    config_manager.set("data.train_path", f"{data_path}/train")
    config_manager.set("data.val_path", f"{data_path}/val")
    config_manager.set("data.test_path", f"{data_path}/test")
    config_manager.set("paths.save_dir", f"{output_dir}/checkpoints")
    config_manager.set("paths.log_dir", f"{output_dir}/logs")
    config_manager.set("paths.results_dir", f"{output_dir}/results")
    
    # Set additional parameters
    for key, value in kwargs.items():
        config_manager.set(key, value)
    
    return config_manager


def setup_experiment(
    experiment_name: str,
    config: Optional[ConfigManager] = None,
    base_dir: str = "experiments",
) -> Path:
    """Setup experiment directory structure.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration manager
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    experiment_dir = Path(base_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "assets").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    
    # Save configuration
    if config:
        config.save(experiment_dir / "configs" / "config.yaml")
    
    return experiment_dir


# Environment variables for configuration
def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables.
    
    Returns:
        Configuration dictionary
    """
    env_config = {}
    
    # Common environment variables
    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            env_config[var.lower()] = value
    
    return env_config


def setup_environment(seed: int = 42) -> None:
    """Setup environment for reproducible experiments.
    
    Args:
        seed: Random seed
    """
    # Set random seeds
    set_seed(seed)
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Print device information
    print_device_info()
    
    # Print environment configuration
    env_config = get_env_config()
    if env_config:
        print("\nEnvironment Configuration:")
        print("-" * 30)
        for key, value in env_config.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Example usage
    setup_environment(seed=42)
    
    # Create and save configuration
    config = create_config(
        model_type="unet",
        data_path="data/raw",
        output_dir="outputs",
        training={"num_epochs": 50, "learning_rate": 2e-3},
    )
    
    config.print_config()
    config.save("config.yaml")
