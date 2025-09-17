#!/usr/bin/env python3
"""
Benchmarking script for FFN models.

This script provides benchmarking capabilities for evaluating FFN model
performance on various metrics and datasets.
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import numpy as np
import yaml

# Add torchffn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchffn.models import FFN3D
from torchffn.inference import FloodFillEngine, FloodFillConfig
from torchffn.data import load_volume, SyntheticDataset
from torchffn.metrics import SegmentationMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark FFN model")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    
    # Data options
    parser.add_argument("--gt", type=str, default=None,
                       help="Path to ground truth volume")
    parser.add_argument("--pred", type=str, default=None,
                       help="Path to predicted volume")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data for benchmarking")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of synthetic samples")
    
    # Model options
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    
    # Benchmark options
    parser.add_argument("--metrics", type=str, nargs="+", 
                       default=["dice", "iou", "vi", "rand"],
                       help="Metrics to compute")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binary metrics")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference")
    
    # Performance options
    parser.add_argument("--warmup-runs", type=int, default=5,
                       help="Number of warmup runs")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                       help="Number of benchmark runs")
    
    # Output options
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for benchmark results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def load_model(model_path: str, device: str) -> FFN3D:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint:
        model_config = checkpoint['config'].get('model', {})
    else:
        # Default configuration
        model_config = {
            'input_channels': 3,
            'stem_channels': 32,
            'num_blocks': 12,
            'block_channels': [32, 64, 96],
            'kernel_size': 3,
            'center_crop_size': 17,
            'use_dilation': False,
            'dropout': 0.0,
        }
    
    # Create model
    model = FFN3D(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def benchmark_model_forward(model: FFN3D, input_shape: tuple, device: str, 
                          warmup_runs: int, benchmark_runs: int) -> dict:
    """Benchmark model forward pass."""
    print("Benchmarking model forward pass...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, *input_shape, device=device)
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark runs
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(benchmark_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / benchmark_runs
    throughput = 1.0 / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput_fps': throughput,
        'input_shape': input_shape,
        'batch_size': 1,
    }


def benchmark_flood_fill(engine: FloodFillEngine, raw_volume: torch.Tensor, 
                        seeds: list, warmup_runs: int, benchmark_runs: int) -> dict:
    """Benchmark flood-fill inference."""
    print("Benchmarking flood-fill inference...")
    
    # Warmup runs
    for _ in range(warmup_runs):
        for seed in seeds[:5]:  # Use first 5 seeds for warmup
            with torch.no_grad():
                _, _ = engine.flood_fill_from_seed(raw_volume, seed)
    
    # Benchmark runs
    torch.cuda.synchronize() if engine.device == "cuda" else None
    start_time = time.time()
    
    for _ in range(benchmark_runs):
        for seed in seeds[:5]:  # Use first 5 seeds for benchmark
            with torch.no_grad():
                _, _ = engine.flood_fill_from_seed(raw_volume, seed)
    
    torch.cuda.synchronize() if engine.device == "cuda" else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / (benchmark_runs * 5)
    throughput = 1.0 / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput_fps': throughput,
        'num_seeds': 5,
        'volume_shape': raw_volume.shape,
    }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, 
                   metrics: list, threshold: float) -> dict:
    """Compute specified metrics."""
    results = {}
    
    metrics_calculator = SegmentationMetrics(threshold=threshold)
    
    if 'dice' in metrics:
        dice_scores = []
        for i in range(pred.shape[0]):
            dice = compute_dice_score(pred[i:i+1], target[i:i+1], threshold)
            dice_scores.append(dice.item())
        results['dice_mean'] = np.mean(dice_scores)
        results['dice_std'] = np.std(dice_scores)
    
    if 'iou' in metrics:
        iou_scores = []
        for i in range(pred.shape[0]):
            iou = compute_iou_score(pred[i:i+1], target[i:i+1], threshold)
            iou_scores.append(iou.item())
        results['iou_mean'] = np.mean(iou_scores)
        results['iou_std'] = np.std(iou_scores)
    
    if 'precision' in metrics or 'recall' in metrics:
        precision_scores = []
        recall_scores = []
        for i in range(pred.shape[0]):
            precision, recall = compute_precision_recall(pred[i:i+1], target[i:i+1], threshold)
            precision_scores.append(precision.item())
            recall_scores.append(recall.item())
        
        if 'precision' in metrics:
            results['precision_mean'] = np.mean(precision_scores)
            results['precision_std'] = np.std(precision_scores)
        
        if 'recall' in metrics:
            results['recall_mean'] = np.mean(recall_scores)
            results['recall_std'] = np.std(recall_scores)
    
    return results


def benchmark_synthetic_data(model: FFN3D, device: str, num_samples: int, 
                           batch_size: int, metrics: list, threshold: float) -> dict:
    """Benchmark on synthetic data."""
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_samples=num_samples,
        volume_size=(64, 64, 32),
        fov_size=(33, 33, 17),
        center_crop_size=(17, 17, 9),
        num_objects_range=(1, 3),
        object_size_range=(3.0, 8.0),
        noise_level=0.1,
        seed=42
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Run inference
    all_predictions = []
    all_targets = []
    
    print("Running inference on synthetic data...")
    with torch.no_grad():
        for batch in dataloader:
            input_tensor = batch['input'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
            all_predictions.append(probs.cpu())
            all_targets.append(target.cpu())
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    print("Computing metrics...")
    metrics_results = compute_metrics(predictions, targets, metrics, threshold)
    
    return metrics_results


def benchmark_real_data(model: FFN3D, gt_path: str, pred_path: str, 
                       device: str, metrics: list, threshold: float) -> dict:
    """Benchmark on real data."""
    print(f"Loading ground truth from {gt_path}")
    gt_volume = load_volume(gt_path, device=device)
    
    print(f"Loading predictions from {pred_path}")
    pred_volume = load_volume(pred_path, device=device)
    
    # Ensure same shape
    if gt_volume.shape != pred_volume.shape:
        raise ValueError(f"Shape mismatch: GT {gt_volume.shape} vs Pred {pred_volume.shape}")
    
    # Convert to batch format for metrics computation
    gt_batch = gt_volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    pred_batch = pred_volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # Compute metrics
    print("Computing metrics...")
    metrics_results = compute_metrics(pred_batch, gt_batch, metrics, threshold)
    
    return metrics_results


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    model = load_model(args.model, device)
    
    # Benchmark results
    results = {
        'model_path': args.model,
        'device': device,
        'metrics': args.metrics,
        'threshold': args.threshold,
    }
    
    # Benchmark model forward pass
    forward_results = benchmark_model_forward(
        model, (33, 33, 17), device, args.warmup_runs, args.benchmark_runs
    )
    results['forward_pass'] = forward_results
    
    # Benchmark flood-fill inference
    if args.synthetic:
        # Create synthetic volume for flood-fill benchmark
        raw_volume = torch.randn(64, 64, 32, device=device)
        seeds = [(32, 32, 16)]  # Center seed
        
        config = FloodFillConfig()
        engine = FloodFillEngine(model, config, device)
        
        flood_fill_results = benchmark_flood_fill(
            engine, raw_volume, seeds, args.warmup_runs, args.benchmark_runs
        )
        results['flood_fill'] = flood_fill_results
    
    # Benchmark metrics
    if args.synthetic:
        metrics_results = benchmark_synthetic_data(
            model, device, args.num_samples, args.batch_size, args.metrics, args.threshold
        )
    elif args.gt and args.pred:
        metrics_results = benchmark_real_data(
            model, args.gt, args.pred, device, args.metrics, args.threshold
        )
    else:
        print("No data provided for metrics benchmarking")
        metrics_results = {}
    
    results['metrics'] = metrics_results
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Metrics: {args.metrics}")
    print(f"Threshold: {args.threshold}")
    
    print("\nForward Pass Performance:")
    print(f"  Average time: {forward_results['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {forward_results['throughput_fps']:.2f} FPS")
    
    if 'flood_fill' in results:
        print("\nFlood-Fill Performance:")
        print(f"  Average time: {results['flood_fill']['avg_time_ms']:.2f} ms")
        print(f"  Throughput: {results['flood_fill']['throughput_fps']:.2f} FPS")
    
    if metrics_results:
        print("\nMetrics:")
        for metric, value in metrics_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}")
        with open(args.output, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
