"""Main training script for video denoising."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from src.models.models import VideoDenoisingModel
from src.data.dataset import create_data_loaders, VideoTransform
from src.train.trainer import train_model
from src.utils.device import setup_environment, create_config, ConfigManager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video denoising model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-type", type=str, default="autoencoder", 
                       choices=["autoencoder", "unet", "temporal"],
                       help="Type of model to train")
    parser.add_argument("--data-path", type=str, default="data/raw",
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment(seed=args.seed)
    
    # Load or create configuration
    if args.config:
        config = ConfigManager(args.config)
    else:
        config = create_config(
            model_type=args.model_type,
            data_path=args.data_path,
            output_dir=args.output_dir,
            training={
                "num_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "device": args.device,
            },
            data={
                "batch_size": args.batch_size,
            }
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(output_dir / "config.yaml")
    
    print("Configuration:")
    config.print_config()
    
    # Create model
    model_config = config.get("model", {})
    model = VideoDenoisingModel(**model_config)
    
    print(f"\nModel: {model.get_model_info()}")
    
    # Create data loaders
    data_config = config.get("data", {})
    
    # Create transforms
    train_transform = VideoTransform(
        resize=tuple(data_config.get("frame_size", [256, 256])),
        horizontal_flip=True,
        brightness_contrast=True,
    )
    
    val_transform = VideoTransform(
        resize=tuple(data_config.get("frame_size", [256, 256])),
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_path=data_config.get("train_path", "data/raw/train"),
        val_path=data_config.get("val_path", "data/raw/val"),
        test_path=data_config.get("test_path", "data/raw/test"),
        batch_size=data_config.get("batch_size", 8),
        num_workers=data_config.get("num_workers", 4),
        noise_type=data_config.get("noise_type", "gaussian"),
        noise_level=data_config.get("noise_level", 0.1),
        temporal_window=data_config.get("temporal_window", 1),
        frame_size=tuple(data_config.get("frame_size", [256, 256])),
        train_transform=train_transform,
        val_transform=val_transform,
    )
    
    print(f"\nData loaders created:")
    print(f"Train batches: {len(data_loaders['train'])}")
    if "val" in data_loaders:
        print(f"Val batches: {len(data_loaders['val'])}")
    if "test" in data_loaders:
        print(f"Test batches: {len(data_loaders['test'])}")
    
    # Training configuration
    training_config = config.get("training", {})
    
    # Train model
    trainer = train_model(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders.get("val"),
        num_epochs=training_config.get("num_epochs", 100),
        learning_rate=training_config.get("learning_rate", 1e-3),
        device=training_config.get("device", "auto"),
        save_dir=output_dir / "checkpoints",
        log_dir=output_dir / "logs",
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
