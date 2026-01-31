# Video Denoising - Advanced Computer Vision Project

A production-ready implementation of video denoising using deep learning techniques. This project provides multiple state-of-the-art models for removing noise from video sequences, with comprehensive training, evaluation, and demo capabilities.

## Features

- **Multiple Model Architectures**: Autoencoder, UNet, and Temporal-aware models
- **Advanced Loss Functions**: L1, Charbonnier, Perceptual, and Temporal consistency losses
- **Comprehensive Metrics**: PSNR, SSIM, LPIPS, and temporal consistency evaluation
- **Modern Training Pipeline**: With TensorBoard logging, checkpointing, and early stopping
- **Interactive Demo**: Streamlit-based web application for real-time denoising
- **Production Ready**: Type hints, comprehensive documentation, and reproducible experiments

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Video-Denoising.git
cd Video-Denoising
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create synthetic data for testing:
```bash
python -c "from src.data.dataset import create_synthetic_dataset; create_synthetic_dataset('data/raw/synthetic')"
```

### Training

Train a model using the default configuration:

```bash
python scripts/train.py --model-type autoencoder --data-path data/raw/synthetic --epochs 50
```

Train with a specific configuration:

```bash
python scripts/train.py --config configs/unet.yaml --epochs 100
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

### Demo

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
0598_Video_Denoising/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   └── models.py            # DenoisingAutoencoder, UNetDenoiser, TemporalDenoiser
│   ├── data/                    # Data handling
│   │   └── dataset.py          # VideoDataset, data loaders, transforms
│   ├── utils/                   # Utilities
│   │   ├── device.py           # Device management, configuration
│   │   └── losses.py           # Loss functions and metrics
│   ├── train/                   # Training utilities
│   │   └── trainer.py          # Trainer, Evaluator classes
│   └── eval/                    # Evaluation utilities
├── configs/                     # Configuration files
│   ├── default.yaml            # Default configuration
│   ├── unet.yaml              # UNet-specific configuration
│   └── temporal.yaml          # Temporal model configuration
├── scripts/                    # Training and evaluation scripts
│   ├── train.py               # Main training script
│   └── evaluate.py           # Evaluation script
├── demo/                      # Demo application
│   └── app.py                # Streamlit demo
├── data/                     # Data directory
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── tests/                   # Unit tests
├── assets/                  # Assets and visualizations
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md             # This file
```

## Models

### 1. DenoisingAutoencoder
A simple CNN-based autoencoder suitable for educational purposes and as a baseline.

**Architecture**: Encoder-Decoder with ReLU activations and Sigmoid output
**Parameters**: ~200K parameters
**Use Case**: Quick prototyping and baseline comparisons

### 2. UNetDenoiser
UNet architecture with skip connections for preserving fine details.

**Architecture**: U-shaped network with skip connections
**Parameters**: ~1M parameters
**Use Case**: High-quality denoising with detail preservation

### 3. TemporalDenoiser
3D CNN-based model that processes multiple frames simultaneously.

**Architecture**: 3D convolutions with temporal processing
**Parameters**: ~500K parameters
**Use Case**: Video sequences with temporal consistency requirements

## Data Format

The project supports multiple data formats:

### Video Files
- **Supported formats**: MP4, AVI, MOV, MKV
- **Structure**: Single video files or directories containing video files
- **Processing**: Automatic frame extraction and temporal windowing

### Image Sequences
- **Supported formats**: JPG, PNG, BMP, TIFF
- **Structure**: Directories containing sequential image files
- **Processing**: Automatic sorting and temporal windowing

### Synthetic Data
The project includes utilities to generate synthetic test data:

```python
from src.data.dataset import create_synthetic_dataset

create_synthetic_dataset(
    output_dir="data/raw/synthetic",
    num_videos=10,
    frames_per_video=100,
    frame_size=(256, 256),
    video_types=["moving_objects", "static_scene", "texture_patterns"]
)
```

## Configuration

The project uses OmegaConf for flexible configuration management. Key configuration sections:

### Model Configuration
```yaml
model:
  type: autoencoder  # autoencoder, unet, temporal
  in_channels: 3
  out_channels: 3
  hidden_channels: 64
  base_channels: 64
  depth: 4
  num_frames: 5
```

### Data Configuration
```yaml
data:
  train_path: data/raw/train
  val_path: data/raw/val
  test_path: data/raw/test
  batch_size: 8
  num_workers: 4
  frame_size: [256, 256]
  temporal_window: 1
  noise_type: gaussian  # gaussian, poisson, salt_pepper
  noise_level: 0.1
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  learning_rate: 1e-3
  weight_decay: 1e-4
  patience: 10
  save_best: true
  device: auto
```

## Loss Functions

### Combined Loss
The project uses a combination of multiple loss functions:

- **L1 Loss**: Basic pixel-wise loss
- **Charbonnier Loss**: Robust to outliers
- **Perceptual Loss**: VGG-based feature loss
- **Temporal Consistency Loss**: Encourages smooth temporal transitions

```python
loss = l1_weight * L1_loss + 
       charbonnier_weight * Charbonnier_loss + 
       perceptual_weight * Perceptual_loss + 
       temporal_weight * Temporal_loss
```

## Metrics

### Image Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Temporal Metrics
- **Temporal Consistency**: Frame-to-frame smoothness
- **Temporal PSNR**: PSNR across temporal dimension

## Training

### Basic Training
```bash
python scripts/train.py \
    --model-type unet \
    --data-path data/raw \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-3
```

### Advanced Training
```bash
python scripts/train.py \
    --config configs/unet.yaml \
    --epochs 150 \
    --device cuda \
    --resume checkpoints/latest.pth
```

### Training Features
- **Automatic device detection**: CUDA → MPS → CPU
- **Mixed precision training**: Automatic mixed precision support
- **Gradient accumulation**: For large batch sizes
- **Early stopping**: Prevents overfitting
- **Checkpointing**: Saves best and latest models
- **TensorBoard logging**: Real-time training visualization

## Evaluation

### Model Evaluation
```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --output-dir results/evaluation
```

### Evaluation Output
- **Metrics file**: `metrics.txt` with numerical results
- **Predictions**: `predictions.pth` with model outputs
- **Targets**: `targets.pth` with ground truth
- **Visualizations**: Side-by-side comparisons

## Demo Application

The Streamlit demo provides an interactive interface for:

### Features
- **File Upload**: Support for video and image files
- **Real-time Processing**: Instant denoising with progress bars
- **Noise Configuration**: Adjustable noise types and levels
- **Model Selection**: Choose between different architectures
- **Download Results**: Save processed content

### Usage
1. Launch the demo: `streamlit run demo/app.py`
2. Upload a video or image file
3. Configure noise parameters
4. Process and download results

## Performance

### Model Comparison

| Model | Parameters | Memory | Speed | Quality |
|-------|------------|--------|-------|---------|
| Autoencoder | 200K | Low | Fast | Good |
| UNet | 1M | Medium | Medium | Excellent |
| Temporal | 500K | High | Slow | Excellent |

### Hardware Requirements

**Minimum Requirements**:
- CPU: 4 cores, 8GB RAM
- GPU: 4GB VRAM (optional)
- Storage: 2GB free space

**Recommended Requirements**:
- CPU: 8 cores, 16GB RAM
- GPU: 8GB VRAM (RTX 3070 or better)
- Storage: 10GB free space

## Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

**Slow Training**:
- Increase number of workers
- Use faster storage (SSD)
- Enable CUDA optimizations

**Poor Quality Results**:
- Increase training epochs
- Adjust learning rate
- Use perceptual loss
- Add data augmentation

### Performance Optimization

**Training Speed**:
- Use multiple GPUs with DataParallel
- Enable mixed precision training
- Optimize data loading with more workers
- Use faster storage for datasets

**Inference Speed**:
- Use TensorRT optimization
- Enable ONNX export
- Use smaller model variants
- Implement model quantization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{video_denoising_2024,
  title={Video Denoising - Advanced Computer Vision Project},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Video-Denoising}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- OpenCV community for computer vision tools
- Streamlit team for the demo framework
- OmegaConf team for configuration management
# Video-Denoising
