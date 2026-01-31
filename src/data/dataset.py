"""Data loading and processing utilities for video denoising."""

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class VideoDataset(Dataset):
    """Dataset for video denoising tasks.
    
    Supports both single video files and directories of video frames.
    Can add synthetic noise for training or use pre-noised videos.
    
    Args:
        data_path: Path to video file or directory containing frames
        noise_type: Type of noise to add ('gaussian', 'poisson', 'salt_pepper', 'none')
        noise_level: Intensity of noise (0.0 to 1.0)
        transform: Optional transform to apply to frames
        temporal_window: Number of frames to process together (for temporal models)
        frame_size: Target frame size (height, width)
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        transform: Optional[Callable] = None,
        temporal_window: int = 1,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.transform = transform
        self.temporal_window = temporal_window
        self.frame_size = frame_size
        
        # Load video frames
        self.frames = self._load_frames()
        
        if len(self.frames) == 0:
            raise ValueError(f"No frames found in {data_path}")
    
    def _load_frames(self) -> List[np.ndarray]:
        """Load frames from video file or directory."""
        frames = []
        
        if self.data_path.is_file():
            # Load from video file
            cap = cv2.VideoCapture(str(self.data_path))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if self.frame_size:
                    frame_rgb = cv2.resize(frame_rgb, (self.frame_size[1], self.frame_size[0]))
                
                frames.append(frame_rgb)
            
            cap.release()
        
        elif self.data_path.is_dir():
            # Load from directory of images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = sorted([
                f for f in self.data_path.iterdir() 
                if f.suffix.lower() in image_extensions
            ])
            
            for img_file in image_files:
                frame = cv2.imread(str(img_file))
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if self.frame_size:
                        frame_rgb = cv2.resize(frame_rgb, (self.frame_size[1], self.frame_size[0]))
                    
                    frames.append(frame_rgb)
        
        return frames
    
    def _add_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add synthetic noise to frame."""
        if self.noise_type == "none":
            return frame
        
        frame_float = frame.astype(np.float32) / 255.0
        
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_level, frame_float.shape)
            noisy_frame = frame_float + noise
        
        elif self.noise_type == "poisson":
            noisy_frame = np.random.poisson(frame_float * 255) / 255.0
        
        elif self.noise_type == "salt_pepper":
            noisy_frame = frame_float.copy()
            salt_pepper = np.random.random(frame_float.shape)
            noisy_frame[salt_pepper < self.noise_level / 2] = 0
            noisy_frame[salt_pepper > 1 - self.noise_level / 2] = 1
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Clip values to valid range
        noisy_frame = np.clip(noisy_frame, 0, 1)
        return (noisy_frame * 255).astype(np.uint8)
    
    def __len__(self) -> int:
        """Return number of samples."""
        if self.temporal_window > 1:
            return max(1, len(self.frames) - self.temporal_window + 1)
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a sample from the dataset."""
        if self.temporal_window > 1:
            # Get temporal window of frames
            start_idx = idx
            end_idx = min(start_idx + self.temporal_window, len(self.frames))
            frame_window = self.frames[start_idx:end_idx]
            
            # Pad if necessary
            while len(frame_window) < self.temporal_window:
                frame_window.append(frame_window[-1])  # Repeat last frame
            
            # Process each frame
            clean_frames = []
            noisy_frames = []
            
            for frame in frame_window:
                clean_frame = frame.copy()
                noisy_frame = self._add_noise(frame)
                
                if self.transform:
                    clean_frame = self.transform(clean_frame)
                    noisy_frame = self.transform(noisy_frame)
                else:
                    # Convert to tensor and normalize
                    clean_frame = torch.from_numpy(clean_frame).permute(2, 0, 1).float() / 255.0
                    noisy_frame = torch.from_numpy(noisy_frame).permute(2, 0, 1).float() / 255.0
                
                clean_frames.append(clean_frame)
                noisy_frames.append(noisy_frame)
            
            # Stack frames: (T, C, H, W)
            clean_tensor = torch.stack(clean_frames)
            noisy_tensor = torch.stack(noisy_frames)
            
            return {
                "clean": clean_tensor,
                "noisy": noisy_tensor,
                "frame_indices": torch.tensor([start_idx + i for i in range(self.temporal_window)])
            }
        
        else:
            # Single frame processing
            clean_frame = self.frames[idx].copy()
            noisy_frame = self._add_noise(clean_frame)
            
            if self.transform:
                clean_frame = self.transform(clean_frame)
                noisy_frame = self.transform(noisy_frame)
            else:
                # Convert to tensor and normalize
                clean_frame = torch.from_numpy(clean_frame).permute(2, 0, 1).float() / 255.0
                noisy_frame = torch.from_numpy(noisy_frame).permute(2, 0, 1).float() / 255.0
            
            return {
                "clean": clean_frame,
                "noisy": noisy_frame,
                "frame_indices": torch.tensor([idx])
            }


class VideoTransform:
    """Custom transform for video frames."""
    
    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = None,
        crop: Optional[Tuple[int, int]] = None,
        horizontal_flip: bool = False,
        rotation: bool = False,
        brightness_contrast: bool = False,
    ) -> None:
        self.resize = resize
        self.crop = crop
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.brightness_contrast = brightness_contrast
    
    def __call__(self, frame: np.ndarray) -> Tensor:
        """Apply transforms to frame."""
        # Convert to PIL Image for transforms
        from PIL import Image
        pil_image = Image.fromarray(frame)
        
        # Resize
        if self.resize:
            pil_image = F.resize(pil_image, self.resize)
        
        # Random crop
        if self.crop:
            pil_image = F.center_crop(pil_image, self.crop)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            pil_image = F.hflip(pil_image)
        
        # Random rotation
        if self.rotation and random.random() > 0.5:
            angle = random.uniform(-15, 15)
            pil_image = F.rotate(pil_image, angle)
        
        # Random brightness and contrast
        if self.brightness_contrast and random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            pil_image = F.adjust_brightness(pil_image, brightness_factor)
            pil_image = F.adjust_contrast(pil_image, contrast_factor)
        
        # Convert back to tensor
        tensor = F.to_tensor(pil_image)
        return tensor


def create_data_loaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    noise_type: str = "gaussian",
    noise_level: float = 0.1,
    temporal_window: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        noise_type: Type of noise to add
        noise_level: Intensity of noise
        temporal_window: Number of frames to process together
        frame_size: Target frame size
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        
    Returns:
        Dictionary containing data loaders
    """
    data_loaders = {}
    
    # Training data loader
    train_dataset = VideoDataset(
        data_path=train_path,
        noise_type=noise_type,
        noise_level=noise_level,
        transform=train_transform,
        temporal_window=temporal_window,
        frame_size=frame_size,
    )
    
    data_loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Validation data loader
    if val_path:
        val_dataset = VideoDataset(
            data_path=val_path,
            noise_type=noise_type,
            noise_level=noise_level,
            transform=val_transform,
            temporal_window=temporal_window,
            frame_size=frame_size,
        )
        
        data_loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    # Test data loader
    if test_path:
        test_dataset = VideoDataset(
            data_path=test_path,
            noise_type=noise_type,
            noise_level=noise_level,
            transform=val_transform,
            temporal_window=temporal_window,
            frame_size=frame_size,
        )
        
        data_loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return data_loaders


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_videos: int = 10,
    frames_per_video: int = 100,
    frame_size: Tuple[int, int] = (256, 256),
    video_types: List[str] = ["moving_objects", "static_scene", "texture_patterns"],
) -> None:
    """Create a synthetic dataset for testing video denoising.
    
    Args:
        output_dir: Directory to save synthetic videos
        num_videos: Number of videos to generate
        frames_per_video: Number of frames per video
        frame_size: Size of each frame
        video_types: Types of synthetic videos to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_idx in range(num_videos):
        video_type = random.choice(video_types)
        video_frames = []
        
        for frame_idx in range(frames_per_video):
            if video_type == "moving_objects":
                # Generate frames with moving colored circles
                frame = np.zeros((*frame_size, 3), dtype=np.uint8)
                
                # Background gradient
                for i in range(frame_size[0]):
                    for j in range(frame_size[1]):
                        frame[i, j] = [i // 4, j // 4, (i + j) // 8]
                
                # Moving circles
                t = frame_idx / frames_per_video
                for i in range(3):
                    x = int(frame_size[1] * (0.2 + 0.6 * np.sin(2 * np.pi * t + i * np.pi / 3)))
                    y = int(frame_size[0] * (0.2 + 0.6 * np.cos(2 * np.pi * t + i * np.pi / 3)))
                    cv2.circle(frame, (x, y), 20, (255, 100 + i * 50, 100), -1)
            
            elif video_type == "static_scene":
                # Generate static scene with some variation
                frame = np.random.randint(50, 200, (*frame_size, 3), dtype=np.uint8)
                
                # Add some geometric shapes
                cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
                cv2.circle(frame, (200, 200), 50, (0, 255, 0), -1)
                
                # Add slight temporal variation
                noise = np.random.normal(0, 5, frame.shape)
                frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            
            elif video_type == "texture_patterns":
                # Generate textured patterns
                frame = np.zeros((*frame_size, 3), dtype=np.uint8)
                
                # Create wave patterns
                for i in range(frame_size[0]):
                    for j in range(frame_size[1]):
                        wave1 = 127 + 127 * np.sin(2 * np.pi * i / 50 + frame_idx * 0.1)
                        wave2 = 127 + 127 * np.cos(2 * np.pi * j / 50 + frame_idx * 0.1)
                        frame[i, j] = [wave1, wave2, (wave1 + wave2) / 2]
            
            video_frames.append(frame)
        
        # Save video
        video_path = output_dir / f"synthetic_video_{video_idx:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, frame_size)
        
        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Created synthetic video: {video_path}")


if __name__ == "__main__":
    # Create a small synthetic dataset for testing
    create_synthetic_dataset(
        output_dir="data/raw/synthetic",
        num_videos=3,
        frames_per_video=50,
        frame_size=(128, 128),
    )
