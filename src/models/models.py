"""Video denoising models for advanced computer vision applications."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DenoisingAutoencoder(nn.Module):
    """Simple CNN-based autoencoder for video denoising.
    
    This is a basic implementation suitable for educational purposes and
    as a baseline for more advanced video denoising models.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        hidden_channels: Number of hidden channels in the encoder/decoder
        out_channels: Number of output channels (default: 3 for RGB)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Ensure output is in range [0, 1]
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Denoised tensor of shape (B, C, H, W)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UNetDenoiser(nn.Module):
    """UNet-based video denoiser with skip connections.
    
    A more advanced architecture that preserves fine details through
    skip connections between encoder and decoder layers.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Number of base channels in the first layer
        depth: Depth of the UNet (number of downsampling layers)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        
        self.depth = depth
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # First encoder layer
        self.encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        
        # Downsampling layers
        for i in range(depth - 1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            
            self.encoder_layers.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Upsampling layers
        for i in range(depth - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),
                    nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),  # +out_ch for skip connection
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Final decoder layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the UNet.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Denoised tensor of shape (B, C, H, W)
        """
        # Encoder forward pass
        encoder_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                x = layer(x)
            else:
                x = layer(x)
            encoder_outputs.append(x)
        
        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder_layers):
            skip_idx = self.depth - 2 - i
            skip_connection = encoder_outputs[skip_idx]
            
            # Concatenate skip connection
            x = torch.cat([x, skip_connection], dim=1)
            x = layer(x)
        
        x = self.final_layer(x)
        return x


class TemporalDenoiser(nn.Module):
    """Temporal-aware video denoiser using 3D convolutions.
    
    This model processes multiple frames simultaneously to leverage
    temporal information for better denoising.
    
    Args:
        in_channels: Number of input channels per frame
        out_channels: Number of output channels per frame
        num_frames: Number of frames to process together
        base_channels: Number of base channels
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_frames: int = 5,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        
        # 3D convolution layers for temporal processing
        self.temporal_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        
        self.temporal_conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        
        # Final 2D convolution for output
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the temporal denoiser.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W) where T is num_frames
            
        Returns:
            Denoised tensor of shape (B, C, H, W) - center frame
        """
        # Process temporal dimension
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        x = self.temporal_conv3(x)
        
        # Extract center frame
        center_idx = self.num_frames // 2
        center_frame = x[:, :, center_idx, :, :]
        
        # Final 2D processing
        output = self.final_conv(center_frame)
        return output


class VideoDenoisingModel(nn.Module):
    """Main video denoising model that can switch between different architectures.
    
    Args:
        model_type: Type of model to use ('autoencoder', 'unet', 'temporal')
        **kwargs: Additional arguments passed to the specific model
    """
    
    def __init__(self, model_type: str = "autoencoder", **kwargs) -> None:
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == "autoencoder":
            self.model = DenoisingAutoencoder(**kwargs)
        elif model_type == "unet":
            self.model = UNetDenoiser(**kwargs)
        elif model_type == "temporal":
            self.model = TemporalDenoiser(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the selected model.
        
        Args:
            x: Input tensor
            
        Returns:
            Denoised tensor
        """
        return self.model(x)
    
    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
