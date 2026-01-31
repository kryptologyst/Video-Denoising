"""Loss functions and metrics for video denoising."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import numpy as np
from typing import Dict, Optional, Tuple


class CharbonnierLoss(nn.Module):
    """Charbonnier loss for robust video denoising.
    
    More robust to outliers than L2 loss, commonly used in video restoration.
    
    Args:
        eps: Small constant for numerical stability
    """
    
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Charbonnier loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Charbonnier loss
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features.
    
    Computes loss in feature space rather than pixel space,
    leading to more perceptually pleasing results.
    
    Args:
        feature_layers: Which VGG layers to use for features
        weights: Weights for each feature layer
    """
    
    def __init__(
        self,
        feature_layers: Tuple[int, ...] = (3, 8, 15, 22),
        weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.feature_extractor = nn.ModuleList()
        
        for i in range(max(feature_layers) + 1):
            self.feature_extractor.append(vgg[i])
        
        self.feature_layers = feature_layers
        self.weights = weights or (1.0,) * len(feature_layers)
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute perceptual loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            
        Returns:
            Perceptual loss
        """
        pred_features = []
        target_features = []
        
        # Extract features
        for i, layer in enumerate(self.feature_extractor):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_features.append(pred)
                target_features.append(target)
        
        # Compute loss for each feature layer
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video denoising.
    
    Encourages smooth temporal transitions in denoised videos.
    
    Args:
        alpha: Weight for temporal consistency loss
    """
    
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred_frames: Tensor) -> Tensor:
        """Compute temporal consistency loss.
        
        Args:
            pred_frames: Predicted frames tensor (B, T, C, H, W)
            
        Returns:
            Temporal consistency loss
        """
        if pred_frames.size(1) < 2:
            return torch.tensor(0.0, device=pred_frames.device)
        
        # Compute frame differences
        frame_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
        
        # L2 loss on frame differences
        temporal_loss = torch.mean(frame_diff ** 2)
        
        return self.alpha * temporal_loss


class CombinedLoss(nn.Module):
    """Combined loss function for video denoising.
    
    Combines multiple loss terms for better denoising results.
    
    Args:
        l1_weight: Weight for L1 loss
        charbonnier_weight: Weight for Charbonnier loss
        perceptual_weight: Weight for perceptual loss
        temporal_weight: Weight for temporal consistency loss
        use_perceptual: Whether to use perceptual loss
        use_temporal: Whether to use temporal consistency loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        charbonnier_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.1,
        use_perceptual: bool = True,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
        
        if use_temporal:
            self.temporal_loss = TemporalConsistencyLoss()
        else:
            self.temporal_loss = None
        
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_frames: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute combined loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            pred_frames: Predicted frames for temporal loss (B, T, C, H, W)
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # L1 loss
        l1_loss = self.l1_loss(pred, target)
        losses["l1"] = l1_loss
        
        # Charbonnier loss
        charbonnier_loss = self.charbonnier_loss(pred, target)
        losses["charbonnier"] = charbonnier_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses["perceptual"] = perceptual_loss
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)
        
        # Temporal consistency loss
        if self.temporal_loss is not None and pred_frames is not None:
            temporal_loss = self.temporal_loss(pred_frames)
            losses["temporal"] = temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.charbonnier_weight * charbonnier_loss +
            self.perceptual_weight * perceptual_loss +
            self.temporal_weight * temporal_loss
        )
        losses["total"] = total_loss
        
        return losses


def calculate_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> Tensor:
    """Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        max_val: Maximum possible value
        
    Returns:
        PSNR value
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def calculate_ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        
    Returns:
        SSIM value
    """
    # Convert to grayscale if needed
    if pred.size(1) == 3:
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
    else:
        pred_gray = pred.squeeze(1)
        target_gray = target.squeeze(1)
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D window
    window = g[:, None] * g[None, :]
    window = window / window.sum()
    
    # Pad images
    pad = window_size // 2
    pred_padded = F.pad(pred_gray, (pad, pad, pad, pad), mode='reflect')
    target_padded = F.pad(target_gray, (pad, pad, pad, pad), mode='reflect')
    
    # Calculate means
    mu_pred = F.conv2d(pred_padded, window.unsqueeze(0).unsqueeze(0), padding=0)
    mu_target = F.conv2d(target_padded, window.unsqueeze(0).unsqueeze(0), padding=0)
    
    # Calculate variances and covariance
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.conv2d(pred_padded ** 2, window.unsqueeze(0).unsqueeze(0), padding=0) - mu_pred_sq
    sigma_target_sq = F.conv2d(target_padded ** 2, window.unsqueeze(0).unsqueeze(0), padding=0) - mu_target_sq
    sigma_pred_target = F.conv2d(pred_padded * target_padded, window.unsqueeze(0).unsqueeze(0), padding=0) - mu_pred_target
    
    # SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    # Calculate SSIM
    ssim = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / \
           ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    
    return ssim.mean()


def calculate_lpips(pred: Tensor, target: Tensor) -> Tensor:
    """Calculate Learned Perceptual Image Patch Similarity (LPIPS).
    
    Note: This is a simplified implementation. For production use,
    consider using the official LPIPS implementation.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        LPIPS value
    """
    # Simple implementation using VGG features
    # In practice, you would use a pre-trained LPIPS model
    vgg = torchvision.models.vgg16(pretrained=True).features
    vgg.eval()
    
    with torch.no_grad():
        pred_features = vgg(pred)
        target_features = vgg(target)
    
    # Compute L2 distance in feature space
    lpips = F.mse_loss(pred_features, target_features)
    return lpips


class MetricsCalculator:
    """Calculate various metrics for video denoising evaluation."""
    
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.metrics = {}
    
    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
        """
        with torch.no_grad():
            psnr = calculate_psnr(pred, target)
            ssim = calculate_ssim(pred, target)
            lpips = calculate_lpips(pred, target)
            
            if "psnr" not in self.metrics:
                self.metrics["psnr"] = []
                self.metrics["ssim"] = []
                self.metrics["lpips"] = []
            
            self.metrics["psnr"].append(psnr.item())
            self.metrics["ssim"].append(ssim.item())
            self.metrics["lpips"].append(lpips.item())
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics.
        
        Returns:
            Dictionary containing average metrics
        """
        if not self.metrics:
            return {}
        
        return {
            "psnr": np.mean(self.metrics["psnr"]),
            "ssim": np.mean(self.metrics["ssim"]),
            "lpips": np.mean(self.metrics["lpips"]),
        }
    
    def reset(self) -> None:
        """Reset metrics."""
        self.metrics = {}


