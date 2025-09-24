"""
Training loop for FFN models.

This module provides the main training loop with support for mixed precision,
gradient clipping, learning rate scheduling, and logging.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    # Create a dummy SummaryWriter class
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

from .losses import create_loss_function


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute accuracy for binary segmentation."""
    predictions = torch.sigmoid(logits) > threshold
    correct = (predictions == targets).float()
    return correct.mean().item()


def compute_f1_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute F1 score for binary segmentation."""
    predictions = torch.sigmoid(logits) > threshold
    
    # Convert to binary
    pred_binary = predictions.float()
    target_binary = targets.float()
    
    # Compute true positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1.item()


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(
        self,
        # Model parameters
        model_config: Dict,
        
        # Training parameters
        batch_size: int = 4,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        
        # Optimization parameters
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        warmup_epochs: int = 10,
        gradient_clip_norm: float = 1.0,
        
        # Mixed precision
        use_amp: bool = True,
        
        # Loss parameters
        loss_config: Dict = None,
        
        # Logging parameters
        log_interval: int = 10,
        save_interval: int = 1000,
        val_interval: int = 500,
        
        # Checkpointing
        save_dir: str = "checkpoints",
        resume_from: Optional[str] = None,
        
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        
        # Random seed
        seed: int = 42,
    ):
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = use_amp
        self.loss_config = loss_config or {'type': 'ffn', 'bce_weight': 1.0, 'dice_weight': 0.0}
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.save_dir = save_dir
        self.resume_from = resume_from
        self.device = device
        self.seed = seed
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(**config.get('training', {}))


class TrainingLoop:
    """
    Main training loop for FFN models.
    
    Handles training with mixed precision, gradient clipping, learning rate
    scheduling, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Union[TrainingConfig, str] = None,
    ):
        """
        Args:
            model: FFN model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if isinstance(config, str):
            self.config = TrainingConfig.from_yaml(config)
        else:
            self.config = config or TrainingConfig({})
        
        # Set device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Set random seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Initialize training components
        self._setup_training()
        
        # Initialize logging
        self._setup_logging()
        
        # Load checkpoint if resuming
        self.start_epoch = 0
        self.global_step = 0
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
    
    def _setup_training(self):
        """Setup training components (optimizer, scheduler, loss, etc.)."""
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Create scheduler
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Create loss function
        self.criterion = create_loss_function(self.config.loss_config)
        
        # Mixed precision scaler
        if self.config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def _setup_logging(self):
        """Setup logging and checkpointing."""
        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.save_dir / "logs")
        
        # Training metrics
        self.train_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'lr': 0.0,
            'epoch': 0,
            'step': 0,
        }
        
        self.val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'epoch': 0,
        }
    
    def train(self):
        """Run the training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            if self.val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self._validate_epoch(epoch)
                self.val_metrics.update(val_metrics)
                self.val_metrics['epoch'] = epoch
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, train_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch, train_metrics['loss'])
        
        # Save final checkpoint
        self._save_checkpoint(self.config.num_epochs - 1, train_metrics['loss'], is_final=True)
        
        print("Training completed!")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_tensor = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    logits = self.model(input_tensor)
                    loss = self.criterion(logits, target)
            else:
                logits = self.model(input_tensor)
                loss = self.criterion(logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                accuracy = compute_accuracy(logits, target)
                f1 = compute_f1_score(logits, target)
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            total_f1 += f1
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.6f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, LR: {current_lr:.2e}")
                
                # Log to TensorBoard
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy, self.global_step)
                self.writer.add_scalar('train/f1_score', f1, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'f1_score': total_f1 / num_batches
        }
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_tensor = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        logits = self.model(input_tensor)
                        loss = self.criterion(logits, target)
                else:
                    logits = self.model(input_tensor)
                    loss = self.criterion(logits, target)
                
                # Compute metrics
                accuracy = compute_accuracy(logits, target)
                f1 = compute_f1_score(logits, target)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_f1 += f1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_f1 = total_f1 / num_batches
        
        # Log validation metrics
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', avg_accuracy, epoch)
        self.writer.add_scalar('val/f1_score', avg_f1, epoch)
        
        print(f"Validation - Epoch {epoch}, Loss: {avg_loss:.6f}, Acc: {avg_accuracy:.4f}, F1: {avg_f1:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'f1_score': avg_f1
        }
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.train_metrics.update({
            'loss': train_metrics['loss'],
            'accuracy': train_metrics['accuracy'],
            'f1_score': train_metrics['f1_score'],
            'lr': current_lr,
            'epoch': epoch,
            'step': self.global_step,
        })
        
        # Log to TensorBoard
        self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
        self.writer.add_scalar('epoch/train_f1_score', train_metrics['f1_score'], epoch)
        self.writer.add_scalar('epoch/lr', current_lr, epoch)
    
    def _save_checkpoint(self, epoch: int, loss: float, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'global_step': self.global_step,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_final:
            checkpoint_path = self.save_dir / "final_checkpoint.pth"
        else:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if not hasattr(self, 'best_loss') or loss < self.best_loss:
            self.best_loss = loss
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")
    
    def close(self):
        """Close training loop and cleanup resources."""
        self.writer.close()
