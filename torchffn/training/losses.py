"""
Loss functions for FFN training.

This module provides various loss functions commonly used in FFN training,
including BCE with logits, Dice loss, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNLoss(nn.Module):
    """
    Combined loss function for FFN training.
    
    Combines BCE with logits loss and optional Dice loss for better
    segmentation performance.
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.0,
        pos_weight: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            pos_weight: Positive class weight for BCE
            reduction: Reduction method for losses
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),
            reduction=reduction
        )
        
        if dice_weight > 0:
            self.dice_loss = DiceLoss(reduction=reduction)
        else:
            self.dice_loss = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions [B, 1, D, H, W]
            targets: Ground truth targets [B, 1, D, H, W]
            
        Returns:
            Combined loss value
        """
        # BCE loss
        bce_loss = self.bce_loss(logits, targets)
        
        total_loss = self.bce_weight * bce_loss
        
        # Dice loss (if enabled)
        if self.dice_loss is not None:
            dice_loss = self.dice_loss(logits, targets)
            total_loss += self.dice_weight * dice_loss
        
        return total_loss


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    
    Computes the Dice coefficient loss, which is useful for handling
    class imbalance in segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Model predictions [B, 1, D, H, W]
            targets: Ground truth targets [B, 1, D, H, W]
            
        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Convert to loss (1 - dice)
        dice_loss = 1.0 - dice
        
        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    which can be useful for segmentation tasks with severe class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions [B, 1, D, H, W]
            targets: Ground truth targets [B, 1, D, H, W]
            
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss for segmentation.
    
    Tversky loss is a generalization of Dice loss that allows for
    different weights for false positives and false negatives.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            logits: Model predictions [B, 1, D, H, W]
            targets: Ground truth targets [B, 1, D, H, W]
            
        Returns:
            Tversky loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute true positives, false positives, and false negatives
        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)
        
        # Compute Tversky coefficient
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Convert to loss (1 - tversky)
        tversky_loss = 1.0 - tversky
        
        # Apply reduction
        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


def create_loss_function(config: dict) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss function instance
    """
    loss_type = config.get('type', 'ffn')
    
    if loss_type == 'ffn':
        return FFNLoss(
            bce_weight=config.get('bce_weight', 1.0),
            dice_weight=config.get('dice_weight', 0.0),
            pos_weight=config.get('pos_weight', 1.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.get('pos_weight', 1.0)),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=config.get('smooth', 1e-6),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', 1.0),
            gamma=config.get('gamma', 2.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=config.get('alpha', 0.3),
            beta=config.get('beta', 0.7),
            smooth=config.get('smooth', 1e-6),
            reduction=config.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
