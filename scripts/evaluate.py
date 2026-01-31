"""Main evaluation script for video denoising."""

import argparse
from pathlib import Path

import torch

from src.models.models import VideoDenoisingModel
from src.data.dataset import create_data_loaders, VideoTransform
from src.train.trainer import evaluate_model
from src.utils.device import setup_environment, ConfigManager


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate video denoising model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Create model
    model_config = config.get("model", {})
    model = VideoDenoisingModel(**model_config)
    
    print(f"Model: {model.get_model_info()}")
    
    # Create test data loader
    data_config = config.get("data", {})
    
    # Override data path if provided
    if args.data_path:
        test_path = args.data_path
    else:
        test_path = data_config.get("test_path", "data/raw/test")
    
    # Create transform
    val_transform = VideoTransform(
        resize=tuple(data_config.get("frame_size", [256, 256])),
    )
    
    # Create test data loader
    test_loaders = create_data_loaders(
        train_path="dummy",  # Not used
        test_path=test_path,
        batch_size=args.batch_size,
        num_workers=data_config.get("num_workers", 4),
        noise_type=data_config.get("noise_type", "gaussian"),
        noise_level=data_config.get("noise_level", 0.1),
        temporal_window=data_config.get("temporal_window", 1),
        frame_size=tuple(data_config.get("frame_size", [256, 256])),
        val_transform=val_transform,
    )
    
    test_loader = test_loaders["test"]
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        checkpoint_path=args.checkpoint,
        device=args.device,
        results_dir=args.output_dir,
    )
    
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.6f}")


if __name__ == "__main__":
    main()
