#!/usr/bin/env python3
"""Test script to verify the video denoising project setup."""

import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.models.models import VideoDenoisingModel, DenoisingAutoencoder, UNetDenoiser, TemporalDenoiser
        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from src.utils.device import get_device, set_seed, ConfigManager
        print("✓ Device utilities imports successful")
    except ImportError as e:
        print(f"✗ Device utilities imports failed: {e}")
        return False
    
    try:
        from src.utils.losses import CombinedLoss, CharbonnierLoss, MetricsCalculator
        print("✓ Loss functions imports successful")
    except ImportError as e:
        print(f"✗ Loss functions imports failed: {e}")
        return False
    
    try:
        from src.data.dataset import VideoDataset, VideoTransform, create_synthetic_dataset
        print("✓ Data utilities imports successful")
    except ImportError as e:
        print(f"✗ Data utilities imports failed: {e}")
        return False
    
    try:
        from src.train.trainer import Trainer, Evaluator
        print("✓ Training utilities imports successful")
    except ImportError as e:
        print(f"✗ Training utilities imports failed: {e}")
        return False
    
    return True


def test_models():
    """Test model creation and forward passes."""
    print("\nTesting models...")
    
    try:
        from src.models.models import VideoDenoisingModel
        
        # Test autoencoder
        model = VideoDenoisingModel(model_type="autoencoder")
        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert output.shape == x.shape
        print("✓ Autoencoder model test passed")
        
        # Test UNet
        model = VideoDenoisingModel(model_type="unet")
        output = model(x)
        assert output.shape == x.shape
        print("✓ UNet model test passed")
        
        # Test temporal
        model = VideoDenoisingModel(model_type="temporal")
        x_temporal = torch.randn(1, 3, 5, 64, 64)
        output = model(x_temporal)
        assert output.shape == (1, 3, 64, 64)
        print("✓ Temporal model test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model tests failed: {e}")
        return False


def test_losses():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    try:
        from src.utils.losses import CombinedLoss, CharbonnierLoss
        
        # Test Charbonnier loss
        loss_fn = CharbonnierLoss()
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        print("✓ Charbonnier loss test passed")
        
        # Test combined loss
        loss_fn = CombinedLoss()
        losses = loss_fn(pred, target)
        assert "total" in losses
        assert losses["total"].item() >= 0
        print("✓ Combined loss test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function tests failed: {e}")
        return False


def test_data_utilities():
    """Test data utilities."""
    print("\nTesting data utilities...")
    
    try:
        from src.data.dataset import VideoTransform, create_synthetic_dataset
        
        # Test transform
        transform = VideoTransform(resize=(128, 128))
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = transform(image)
        assert transformed.shape == (3, 128, 128)
        print("✓ VideoTransform test passed")
        
        # Test synthetic dataset creation
        create_synthetic_dataset(
            output_dir="test_data",
            num_videos=1,
            frames_per_video=5,
            frame_size=(64, 64)
        )
        print("✓ Synthetic dataset creation test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Data utilities tests failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.device import create_config, ConfigManager
        
        # Test config creation
        config = create_config(model_type="unet")
        assert config.get("model.type") == "unet"
        print("✓ Configuration creation test passed")
        
        # Test config manager
        config_manager = ConfigManager()
        assert config_manager.get("model.type") == "autoencoder"
        print("✓ Configuration manager test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration tests failed: {e}")
        return False


def test_device_utilities():
    """Test device utilities."""
    print("\nTesting device utilities...")
    
    try:
        from src.utils.device import get_device, set_seed
        
        # Test device detection
        device = get_device("auto")
        assert isinstance(device, torch.device)
        print("✓ Device detection test passed")
        
        # Test seed setting
        set_seed(42)
        torch_rand = torch.rand(1)
        set_seed(42)
        torch_rand2 = torch.rand(1)
        assert torch.allclose(torch_rand, torch_rand2)
        print("✓ Seed setting test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Device utilities tests failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Video Denoising Project - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_losses,
        test_data_utilities,
        test_configuration,
        test_device_utilities,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
