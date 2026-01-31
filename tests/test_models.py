"""Unit tests for video denoising models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.models import DenoisingAutoencoder, UNetDenoiser, TemporalDenoiser, VideoDenoisingModel
from src.utils.losses import CharbonnierLoss, PerceptualLoss, TemporalConsistencyLoss, CombinedLoss
from src.utils.device import get_device, set_seed
from src.data.dataset import VideoDataset, VideoTransform


class TestModels:
    """Test model implementations."""
    
    def test_denoising_autoencoder(self):
        """Test DenoisingAutoencoder model."""
        model = DenoisingAutoencoder(in_channels=3, hidden_channels=64, out_channels=3)
        
        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        assert output.shape == x.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_unet_denoiser(self):
        """Test UNetDenoiser model."""
        model = UNetDenoiser(in_channels=3, out_channels=3, base_channels=32, depth=3)
        
        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        assert output.shape == x.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_temporal_denoiser(self):
        """Test TemporalDenoiser model."""
        model = TemporalDenoiser(in_channels=3, out_channels=3, num_frames=5, base_channels=32)
        
        # Test forward pass
        x = torch.randn(2, 3, 5, 64, 64)
        output = model(x)
        
        assert output.shape == (2, 3, 64, 64)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_video_denoising_model(self):
        """Test VideoDenoisingModel wrapper."""
        # Test autoencoder
        model = VideoDenoisingModel(model_type="autoencoder")
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == x.shape
        
        # Test UNet
        model = VideoDenoisingModel(model_type="unet")
        output = model(x)
        assert output.shape == x.shape
        
        # Test temporal
        model = VideoDenoisingModel(model_type="temporal")
        x_temporal = torch.randn(2, 3, 5, 64, 64)
        output = model(x_temporal)
        assert output.shape == (2, 3, 64, 64)
    
    def test_model_info(self):
        """Test model information retrieval."""
        model = VideoDenoisingModel(model_type="autoencoder")
        info = model.get_model_info()
        
        assert "model_type" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["model_type"] == "autoencoder"


class TestLosses:
    """Test loss functions."""
    
    def test_charbonnier_loss(self):
        """Test CharbonnierLoss."""
        loss_fn = CharbonnierLoss()
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0
        assert loss.shape == torch.Size([])
    
    def test_perceptual_loss(self):
        """Test PerceptualLoss."""
        loss_fn = PerceptualLoss()
        pred = torch.randn(2, 3, 224, 224)
        target = torch.randn(2, 3, 224, 224)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0
        assert loss.shape == torch.Size([])
    
    def test_temporal_consistency_loss(self):
        """Test TemporalConsistencyLoss."""
        loss_fn = TemporalConsistencyLoss()
        pred_frames = torch.randn(2, 5, 3, 64, 64)
        
        loss = loss_fn(pred_frames)
        assert loss.item() >= 0.0
        assert loss.shape == torch.Size([])
    
    def test_combined_loss(self):
        """Test CombinedLoss."""
        loss_fn = CombinedLoss()
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        pred_frames = torch.randn(2, 5, 3, 64, 64)
        
        losses = loss_fn(pred, target, pred_frames)
        
        assert "l1" in losses
        assert "charbonnier" in losses
        assert "perceptual" in losses
        assert "temporal" in losses
        assert "total" in losses
        
        for loss_name, loss_value in losses.items():
            assert loss_value.item() >= 0.0


class TestData:
    """Test data handling."""
    
    def test_video_transform(self):
        """Test VideoTransform."""
        transform = VideoTransform(
            resize=(128, 128),
            horizontal_flip=True,
            brightness_contrast=True,
        )
        
        # Create test image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply transform
        transformed = transform(image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 128, 128)
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0


class TestDevice:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Reset seed and test reproducibility
        set_seed(42)
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        # Create model
        model = VideoDenoisingModel(model_type="autoencoder")
        
        # Create loss function
        criterion = CombinedLoss()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy data
        batch_size = 2
        noisy = torch.randn(batch_size, 3, 64, 64)
        clean = torch.randn(batch_size, 3, 64, 64)
        
        # Training step
        optimizer.zero_grad()
        pred = model(noisy)
        losses = criterion(pred, clean)
        loss = losses["total"]
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(loss)
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        model = VideoDenoisingModel(model_type="unet")
        
        # Save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_info": model.get_model_info(),
        }
        
        # Load model
        new_model = VideoDenoisingModel(model_type="unet")
        new_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Test that models produce same output
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__])
