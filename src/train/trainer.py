"""Training and evaluation utilities for video denoising."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.models.models import VideoDenoisingModel
from src.utils.losses import CombinedLoss, MetricsCalculator
from src.utils.device import get_device, set_seed


class Trainer:
    """Trainer class for video denoising models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        save_dir: Union[str, Path] = "checkpoints",
        log_dir: Union[str, Path] = "logs",
        save_best: bool = True,
        patience: int = 10,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            save_best: Whether to save best model
            patience: Early stopping patience
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = criterion or CombinedLoss()
        
        # Optimizer
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        
        # Scheduler
        self.scheduler = scheduler
        
        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.save_best = save_best
        self.patience = patience
        
        # Metrics
        self.train_metrics = MetricsCalculator(self.device)
        self.val_metrics = MetricsCalculator(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if noisy.dim() == 5:  # Temporal data (B, T, C, H, W)
                # Reshape for temporal models
                B, T, C, H, W = noisy.shape
                noisy_flat = noisy.view(B * T, C, H, W)
                clean_flat = clean.view(B * T, C, H, W)
                
                pred_flat = self.model(noisy_flat)
                pred = pred_flat.view(B, T, C, H, W)
                
                # Compute loss
                losses = self.criterion(pred_flat, clean_flat, pred)
            else:  # Single frame data (B, C, H, W)
                pred = self.model(noisy)
                losses = self.criterion(pred, clean)
            
            loss = losses["total"]
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(pred, clean)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            })
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        metrics["loss"] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move data to device
                noisy = batch["noisy"].to(self.device)
                clean = batch["clean"].to(self.device)
                
                # Forward pass
                if noisy.dim() == 5:  # Temporal data
                    B, T, C, H, W = noisy.shape
                    noisy_flat = noisy.view(B * T, C, H, W)
                    clean_flat = clean.view(B * T, C, H, W)
                    
                    pred_flat = self.model(noisy_flat)
                    pred = pred_flat.view(B, T, C, H, W)
                    
                    losses = self.criterion(pred_flat, clean_flat, pred)
                else:  # Single frame data
                    pred = self.model(noisy)
                    losses = self.criterion(pred, clean)
                
                loss = losses["total"]
                total_loss += loss.item()
                
                # Update metrics
                self.val_metrics.update(pred, clean)
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        metrics["loss"] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "latest.pth")
        
        # Save best checkpoint
        if is_best and self.save_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def train(self, num_epochs: int, resume_from: Optional[Union[str, Path]] = None) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.epoch}")
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                else:
                    self.scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, val_metrics)
            
            # Check for best model
            val_loss = val_metrics.get("loss", float('inf'))
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print epoch summary
            print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val PSNR: {val_metrics.get('psnr', 0):.2f}, "
                  f"Val SSIM: {val_metrics.get('ssim', 0):.4f}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Close tensorboard writer
        self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log metrics to tensorboard.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.epoch)
        
        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, self.epoch)
        
        # Learning rate
        if self.scheduler is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", lr, self.epoch)


class Evaluator:
    """Evaluator class for video denoising models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "auto",
        save_results: bool = True,
        results_dir: Union[str, Path] = "results",
    ) -> None:
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use
            save_results: Whether to save results
            results_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = get_device(device)
        self.save_results = save_results
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Results directory
        if self.save_results:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(self.device)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Starting evaluation...")
        
        self.metrics_calculator.reset()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                noisy = batch["noisy"].to(self.device)
                clean = batch["clean"].to(self.device)
                
                # Forward pass
                if noisy.dim() == 5:  # Temporal data
                    B, T, C, H, W = noisy.shape
                    noisy_flat = noisy.view(B * T, C, H, W)
                    clean_flat = clean.view(B * T, C, H, W)
                    
                    pred_flat = self.model(noisy_flat)
                    pred = pred_flat.view(B, T, C, H, W)
                else:  # Single frame data
                    pred = self.model(noisy)
                
                # Update metrics
                self.metrics_calculator.update(pred, clean)
                
                # Store predictions and targets for analysis
                if self.save_results:
                    all_predictions.append(pred.cpu())
                    all_targets.append(clean.cpu())
                
                # Update progress bar
                pbar.set_postfix({"batch": batch_idx + 1})
        
        # Compute final metrics
        metrics = self.metrics_calculator.compute()
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in metrics.items():
            print(f"{key.upper()}: {value:.4f}")
        
        # Save results
        if self.save_results:
            self._save_results(metrics, all_predictions, all_targets)
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict[str, float],
        predictions: List[Tensor],
        targets: List[Tensor],
    ) -> None:
        """Save evaluation results.
        
        Args:
            metrics: Evaluation metrics
            predictions: Model predictions
            targets: Ground truth targets
        """
        # Save metrics
        metrics_path = self.results_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        
        # Save predictions and targets
        pred_tensor = torch.cat(predictions, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        
        torch.save(pred_tensor, self.results_dir / "predictions.pth")
        torch.save(target_tensor, self.results_dir / "targets.pth")
        
        print(f"Results saved to {self.results_dir}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "auto",
    save_dir: Union[str, Path] = "checkpoints",
    log_dir: Union[str, Path] = "logs",
) -> Trainer:
    """Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Trained trainer object
    """
    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
    )
    
    # Train model
    trainer.train(num_epochs)
    
    return trainer


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
    results_dir: Union[str, Path] = "results",
) -> Dict[str, float]:
    """Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        checkpoint_path: Path to model checkpoint
        device: Device to use
        results_dir: Directory to save results
        
    Returns:
        Evaluation metrics
    """
    # Load checkpoint if specified
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        results_dir=results_dir,
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    return metrics
