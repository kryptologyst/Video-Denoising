"""Streamlit demo application for video denoising."""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
import io

from src.models.models import VideoDenoisingModel
from src.utils.device import get_device, setup_environment


def load_model(model_type: str = "autoencoder", checkpoint_path: str = None):
    """Load the denoising model."""
    model = VideoDenoisingModel(model_type=model_type)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        st.success(f"Model loaded from {checkpoint_path}")
    else:
        st.warning("No checkpoint provided, using random weights")
    
    model.eval()
    return model


def add_noise(image: np.ndarray, noise_type: str = "gaussian", noise_level: float = 0.1) -> np.ndarray:
    """Add noise to image."""
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    elif noise_type == "poisson":
        noisy_image = np.random.poisson(image).astype(np.uint8)
    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        salt_pepper = np.random.random(image.shape[:2])
        noisy_image[salt_pepper < noise_level / 2] = 0
        noisy_image[salt_pepper > 1 - noise_level / 2] = 255
    else:
        noisy_image = image
    
    return noisy_image


def denoise_image(model, image: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Denoise image using the model."""
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Denoise
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    
    # Convert back to numpy
    denoised_image = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)
    
    return denoised_image


def process_video(model, video_file, noise_type: str, noise_level: float, device: str = "cpu"):
    """Process video file for denoising."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add noise
            noisy_frame = add_noise(frame_rgb, noise_type, noise_level)
            
            # Denoise
            denoised_frame = denoise_image(model, noisy_frame, device)
            
            # Convert RGB to BGR for output
            denoised_bgr = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2BGR)
            out.write(denoised_bgr)
            
            frame_count += 1
            progress_bar.progress(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        cap.release()
        out.release()
        
        return output_path
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Video Denoising Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Denoising Demo")
    st.markdown("Upload a video or image to see the denoising in action!")
    
    # Setup environment
    setup_environment()
    device = get_device()
    st.info(f"Using device: {device}")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["autoencoder", "unet", "temporal"],
        help="Choose the type of denoising model"
    )
    
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        help="Path to model checkpoint (optional)"
    )
    
    # Load model
    model = load_model(model_type, checkpoint_path)
    
    # Noise configuration
    st.sidebar.header("Noise Configuration")
    
    noise_type = st.sidebar.selectbox(
        "Noise Type",
        ["gaussian", "poisson", "salt_pepper"],
        help="Type of noise to add"
    )
    
    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Intensity of noise"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a video or image",
            type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
            help="Upload a video file or image to denoise"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Process based on file type
            if uploaded_file.type.startswith('video/'):
                st.subheader("Video Processing")
                
                if st.button("Process Video"):
                    with st.spinner("Processing video..."):
                        output_path = process_video(
                            model, uploaded_file, noise_type, noise_level, device
                        )
                        
                        # Display processed video
                        st.success("Video processed successfully!")
                        
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Denoised Video",
                                data=f.read(),
                                file_name=f"denoised_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        
                        # Clean up
                        os.unlink(output_path)
            
            elif uploaded_file.type.startswith('image/'):
                st.subheader("Image Processing")
                
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Display original image
                st.image(image_array, caption="Original Image", use_column_width=True)
                
                if st.button("Denoise Image"):
                    with st.spinner("Processing image..."):
                        # Add noise
                        noisy_image = add_noise(image_array, noise_type, noise_level)
                        
                        # Denoise
                        denoised_image = denoise_image(model, noisy_image, device)
                        
                        # Display results
                        col_noisy, col_denoised = st.columns(2)
                        
                        with col_noisy:
                            st.image(noisy_image, caption="Noisy Image", use_column_width=True)
                        
                        with col_denoised:
                            st.image(denoised_image, caption="Denoised Image", use_column_width=True)
    
    with col2:
        st.header("Model Information")
        
        # Display model info
        model_info = model.get_model_info()
        
        st.metric("Model Type", model_info["model_type"])
        st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
        st.metric("Trainable Parameters", f"{model_info['trainable_parameters']:,}")
        
        # Display model architecture
        st.subheader("Model Architecture")
        st.code(str(model), language="python")
        
        # Instructions
        st.header("Instructions")
        st.markdown("""
        1. **Upload a file**: Choose a video (.mp4, .avi, .mov) or image (.jpg, .png)
        2. **Configure noise**: Select noise type and level in the sidebar
        3. **Process**: Click the process button to denoise your content
        4. **Download**: Download the denoised result
        
        **Note**: For best results, use a trained model checkpoint. Without a checkpoint,
        the model uses random weights and results will not be meaningful.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit | Video Denoising using Deep Learning | "
        "Advanced Computer Vision Project"
    )


if __name__ == "__main__":
    main()
