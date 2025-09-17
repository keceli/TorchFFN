"""
Segmentation metrics for evaluating FFN performance.

This module provides various metrics commonly used in EM connectomics,
including VI (Variation of Information), RAND, Dice, IoU, and object-wise statistics.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice score between prediction and target.
    
    Args:
        pred: Prediction tensor [B, 1, D, H, W] or [D, H, W]
        target: Target tensor [B, 1, D, H, W] or [D, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        Dice score tensor
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(pred_binary.size(0), -1) if pred_binary.dim() > 3 else pred_binary.flatten()
    target_flat = target.view(target.size(0), -1) if target.dim() > 3 else target.flatten()
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
    
    # Compute Dice score
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def compute_iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) score.
    
    Args:
        pred: Prediction tensor [B, 1, D, H, W] or [D, H, W]
        target: Target tensor [B, 1, D, H, W] or [D, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        IoU score tensor
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(pred_binary.size(0), -1) if pred_binary.dim() > 3 else pred_binary.flatten()
    target_flat = target.view(target.size(0), -1) if target.dim() > 3 else target.flatten()
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
    
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute precision and recall.
    
    Args:
        pred: Prediction tensor [B, 1, D, H, W] or [D, H, W]
        target: Target tensor [B, 1, D, H, W] or [D, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        Tuple of (precision, recall) tensors
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(pred_binary.size(0), -1) if pred_binary.dim() > 3 else pred_binary.flatten()
    target_flat = target.view(target.size(0), -1) if target.dim() > 3 else target.flatten()
    
    # Compute true positives, false positives, false negatives
    tp = (pred_flat * target_flat).sum(dim=-1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=-1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=-1)
    
    # Compute precision and recall
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision, recall


def compute_vi_score(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute Variation of Information (VI) score.
    
    VI measures the amount of information lost and gained when changing
    from one clustering to another. Lower values indicate better agreement.
    
    Args:
        pred_labels: Predicted labels [D, H, W]
        target_labels: Target labels [D, H, W]
        
    Returns:
        Tuple of (VI, split VI, merge VI)
    """
    # Convert to numpy
    pred_np = pred_labels.detach().cpu().numpy()
    target_np = target_labels.detach().cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Get unique labels
    pred_unique = np.unique(pred_flat)
    target_unique = np.unique(target_flat)
    
    # Remove background (0)
    pred_unique = pred_unique[pred_unique > 0]
    target_unique = target_unique[target_unique > 0]
    
    if len(pred_unique) == 0 and len(target_unique) == 0:
        return 0.0, 0.0, 0.0
    
    if len(pred_unique) == 0 or len(target_unique) == 0:
        return float('inf'), float('inf'), float('inf')
    
    # Compute contingency table
    n = len(pred_flat)
    contingency = np.zeros((len(pred_unique), len(target_unique)))
    
    for i, p_label in enumerate(pred_unique):
        for j, t_label in enumerate(target_unique):
            contingency[i, j] = np.sum((pred_flat == p_label) & (target_flat == t_label))
    
    # Compute marginal sums
    pred_sums = np.sum(contingency, axis=1)
    target_sums = np.sum(contingency, axis=0)
    
    # Compute entropies
    pred_entropy = 0.0
    for p_sum in pred_sums:
        if p_sum > 0:
            pred_entropy -= (p_sum / n) * np.log2(p_sum / n)
    
    target_entropy = 0.0
    for t_sum in target_sums:
        if t_sum > 0:
            target_entropy -= (t_sum / n) * np.log2(t_sum / n)
    
    # Compute mutual information
    mutual_info = 0.0
    for i in range(len(pred_unique)):
        for j in range(len(target_unique)):
            if contingency[i, j] > 0:
                mutual_info += (contingency[i, j] / n) * np.log2(
                    (contingency[i, j] * n) / (pred_sums[i] * target_sums[j])
                )
    
    # Compute VI
    vi = pred_entropy + target_entropy - 2 * mutual_info
    
    # Compute split and merge VI
    split_vi = pred_entropy - mutual_info
    merge_vi = target_entropy - mutual_info
    
    return vi, split_vi, merge_vi


def compute_rand_score(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor
) -> float:
    """
    Compute RAND score (adapted for segmentation).
    
    RAND score measures the similarity between two clusterings.
    Higher values indicate better agreement.
    
    Args:
        pred_labels: Predicted labels [D, H, W]
        target_labels: Target labels [D, H, W]
        
    Returns:
        RAND score
    """
    # Convert to numpy
    pred_np = pred_labels.detach().cpu().numpy()
    target_np = target_labels.detach().cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Get unique labels
    pred_unique = np.unique(pred_flat)
    target_unique = np.unique(target_flat)
    
    # Remove background (0)
    pred_unique = pred_unique[pred_unique > 0]
    target_unique = target_unique[target_unique > 0]
    
    if len(pred_unique) == 0 and len(target_unique) == 0:
        return 1.0
    
    if len(pred_unique) == 0 or len(target_unique) == 0:
        return 0.0
    
    # Compute contingency table
    contingency = np.zeros((len(pred_unique), len(target_unique)))
    
    for i, p_label in enumerate(pred_unique):
        for j, t_label in enumerate(target_unique):
            contingency[i, j] = np.sum((pred_flat == p_label) & (target_flat == t_label))
    
    # Compute RAND score
    n = len(pred_flat)
    n_choose_2 = n * (n - 1) / 2
    
    # Count pairs that agree
    agree_same = 0
    agree_diff = 0
    
    # Pairs that are in the same cluster in both
    for i in range(len(pred_unique)):
        for j in range(len(target_unique)):
            if contingency[i, j] > 1:
                agree_same += contingency[i, j] * (contingency[i, j] - 1) / 2
    
    # Pairs that are in different clusters in both
    pred_sums = np.sum(contingency, axis=1)
    target_sums = np.sum(contingency, axis=0)
    
    total_pairs = 0
    for p_sum in pred_sums:
        if p_sum > 1:
            total_pairs += p_sum * (p_sum - 1) / 2
    
    for t_sum in target_sums:
        if t_sum > 1:
            total_pairs += t_sum * (t_sum - 1) / 2
    
    agree_diff = n_choose_2 - total_pairs + agree_same
    
    # RAND score
    rand_score = (agree_same + agree_diff) / n_choose_2
    
    return rand_score


def compute_object_stats(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute object-wise statistics.
    
    Args:
        pred_labels: Predicted labels [D, H, W]
        target_labels: Target labels [D, H, W]
        
    Returns:
        Dictionary of object statistics
    """
    # Convert to numpy
    pred_np = pred_labels.detach().cpu().numpy()
    target_np = target_labels.detach().cpu().numpy()
    
    # Get unique labels
    pred_unique = np.unique(pred_np)
    target_unique = np.unique(target_np)
    
    # Remove background (0)
    pred_unique = pred_unique[pred_unique > 0]
    target_unique = target_unique[target_unique > 0]
    
    stats = {
        'num_pred_objects': len(pred_unique),
        'num_target_objects': len(target_unique),
        'num_objects': max(len(pred_unique), len(target_unique)),
    }
    
    if len(pred_unique) == 0 or len(target_unique) == 0:
        stats.update({
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'split_count': 0,
            'merge_count': 0,
        })
        return stats
    
    # Compute object-wise precision and recall
    # This is a simplified implementation
    # In practice, you'd use more sophisticated object matching
    
    # Count objects that have significant overlap
    threshold = 0.5  # IoU threshold for object matching
    
    matched_pred = set()
    matched_target = set()
    
    for p_label in pred_unique:
        for t_label in target_unique:
            if p_label in matched_pred or t_label in matched_target:
                continue
            
            # Compute IoU between objects
            pred_mask = (pred_np == p_label)
            target_mask = (target_np == t_label)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            if union > 0:
                iou = intersection / union
                if iou >= threshold:
                    matched_pred.add(p_label)
                    matched_target.add(t_label)
    
    # Compute statistics
    precision = len(matched_pred) / len(pred_unique) if len(pred_unique) > 0 else 0.0
    recall = len(matched_target) / len(target_unique) if len(target_unique) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    split_count = len(pred_unique) - len(matched_pred)
    merge_count = len(target_unique) - len(matched_target)
    
    stats.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'split_count': split_count,
        'merge_count': merge_count,
    })
    
    return stats


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    
    Computes multiple metrics for evaluating segmentation performance.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for binarizing predictions
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.vi_scores = []
        self.rand_scores = []
        self.object_stats = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Prediction probabilities [B, 1, D, H, W] or [D, H, W]
            target: Target binary mask [B, 1, D, H, W] or [D, H, W]
            pred_labels: Predicted labels [D, H, W] (optional)
            target_labels: Target labels [D, H, W] (optional)
        """
        # Compute basic metrics
        dice = compute_dice_score(pred, target, self.threshold)
        iou = compute_iou_score(pred, target, self.threshold)
        precision, recall = compute_precision_recall(pred, target, self.threshold)
        
        self.dice_scores.append(dice.mean().item())
        self.iou_scores.append(iou.mean().item())
        self.precision_scores.append(precision.mean().item())
        self.recall_scores.append(recall.mean().item())
        
        # Compute advanced metrics if labels are provided
        if pred_labels is not None and target_labels is not None:
            vi, split_vi, merge_vi = compute_vi_score(pred_labels, target_labels)
            rand_score = compute_rand_score(pred_labels, target_labels)
            obj_stats = compute_object_stats(pred_labels, target_labels)
            
            self.vi_scores.append(vi)
            self.rand_scores.append(rand_score)
            self.object_stats.append(obj_stats)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        if self.dice_scores:
            metrics['dice_mean'] = np.mean(self.dice_scores)
            metrics['dice_std'] = np.std(self.dice_scores)
        
        if self.iou_scores:
            metrics['iou_mean'] = np.mean(self.iou_scores)
            metrics['iou_std'] = np.std(self.iou_scores)
        
        if self.precision_scores:
            metrics['precision_mean'] = np.mean(self.precision_scores)
            metrics['precision_std'] = np.std(self.precision_scores)
        
        if self.recall_scores:
            metrics['recall_mean'] = np.mean(self.recall_scores)
            metrics['recall_std'] = np.std(self.recall_scores)
        
        if self.vi_scores:
            metrics['vi_mean'] = np.mean(self.vi_scores)
            metrics['vi_std'] = np.std(self.vi_scores)
        
        if self.rand_scores:
            metrics['rand_mean'] = np.mean(self.rand_scores)
            metrics['rand_std'] = np.std(self.rand_scores)
        
        if self.object_stats:
            # Aggregate object statistics
            all_precisions = [stats['precision'] for stats in self.object_stats]
            all_recalls = [stats['recall'] for stats in self.object_stats]
            all_f1_scores = [stats['f1_score'] for stats in self.object_stats]
            
            metrics['object_precision_mean'] = np.mean(all_precisions)
            metrics['object_recall_mean'] = np.mean(all_recalls)
            metrics['object_f1_mean'] = np.mean(all_f1_scores)
            
            metrics['object_precision_std'] = np.std(all_precisions)
            metrics['object_recall_std'] = np.std(all_recalls)
            metrics['object_f1_std'] = np.std(all_f1_scores)
        
        return metrics
