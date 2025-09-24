#!/usr/bin/env python3
"""
Training script for FFN models.

This script provides a command-line interface for training FFN models
on EM volume data with various configuration options.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add torchffn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchffn.models import FFN3D, create_ffn3d_baseline, create_ffn3d_small
from torchffn.data import SyntheticDataset, EMVolumeDataset
from torchffn.training import TrainingLoop, TrainingConfig
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FFN model")
    
    # Configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to data directory (overrides config)")
    parser.add_argument("--out-dir", type=str, default=None,
                       help="Output directory (overrides config)")
    
    # Model options
    parser.add_argument("--model-type", type=str, default="baseline",
                       choices=["baseline", "small", "custom"],
                       help="Model type to use")
    
    # Training options
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Data options
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data instead of real data")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of synthetic samples (overrides config)")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=None,
                       help="Logging interval (overrides config)")
    parser.add_argument("--save-interval", type=int, default=None,
                       help="Save interval (overrides config)")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, model_type: str) -> FFN3D:
    """Create FFN model based on configuration and type."""
    if model_type == "baseline":
        return create_ffn3d_baseline(
            input_channels=config['model']['input_channels'],
            center_crop_size=config['model']['center_crop_size']
        )
    elif model_type == "small":
        return create_ffn3d_small(
            input_channels=config['model']['input_channels'],
            center_crop_size=config['model']['center_crop_size']
        )
    else:  # custom
        return FFN3D(**config['model'])


def create_dataset(config: dict, args) -> torch.utils.data.Dataset:
    """Create dataset based on configuration and arguments."""
    # Check if we should use synthetic data
    use_synthetic = args.synthetic
    
    if not use_synthetic:
        # Check if real data directory exists and has data
        data_dir = args.data_dir or config.get('data', {}).get('data_dir', './data')
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"Warning: No real data found in {data_dir}, falling back to synthetic data")
            use_synthetic = True
    
    if use_synthetic:
        # Use synthetic dataset
        dataset_config = config.get('data', {})
        return SyntheticDataset(
            num_samples=args.num_samples or dataset_config.get('num_samples', 1000),
            volume_size=tuple(dataset_config.get('volume_size', [64, 64, 32])),
            fov_size=tuple(dataset_config.get('fov_size', [33, 33, 17])),
            center_crop_size=tuple(dataset_config.get('center_crop_size', [17, 17, 9])),
            num_objects_range=tuple(dataset_config.get('num_objects_range', [1, 3])),
            object_size_range=tuple(dataset_config.get('object_size_range', [3.0, 8.0])),
            noise_level=dataset_config.get('noise_level', 0.1),
            seed=config.get('training', {}).get('seed', 42)
        )
    else:
        # Use real EM dataset
        data_dir = args.data_dir or config.get('data', {}).get('data_dir', './data')
        return EMVolumeDataset(
            data_dir=data_dir,
            split="train",
            fov_size=tuple(config.get('data', {}).get('fov_size', [33, 33, 17])),
            center_crop_size=tuple(config.get('data', {}).get('center_crop_size', [17, 17, 9])),
            samples_per_volume=config.get('data', {}).get('samples_per_volume', 100),
            augmentation_config=config.get('data', {}).get('augmentation'),
            normalize_method=config.get('data', {}).get('normalize_method', 'per_volume'),
            seed=config.get('training', {}).get('seed', 42)
        )


def create_dataloader(dataset: torch.utils.data.Dataset, config: dict, args) -> DataLoader:
    """Create data loader."""
    batch_size = args.batch_size or config.get('training', {}).get('batch_size', 4)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )


def update_config_with_args(config: dict, args):
    """Update configuration with command line arguments."""
    if args.out_dir:
        config.setdefault('training', {})['save_dir'] = args.out_dir
    
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.epochs:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    
    if args.resume:
        config.setdefault('training', {})['resume_from'] = args.resume
    
    if args.log_interval:
        config.setdefault('training', {})['log_interval'] = args.log_interval
    
    if args.save_interval:
        config.setdefault('training', {})['save_interval'] = args.save_interval


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    update_config_with_args(config, args)
    
    # Create model
    print("Creating model...")
    model = create_model(config, args.model_type)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(config, args)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create data loader
    dataloader = create_dataloader(dataset, config, args)
    print(f"Data loader created with batch size {dataloader.batch_size}")
    
    # Create training configuration
    training_config_dict = config.get('training', {}).copy()
    
    # Convert string values to appropriate types
    if 'learning_rate' in training_config_dict:
        training_config_dict['learning_rate'] = float(training_config_dict['learning_rate'])
    if 'weight_decay' in training_config_dict:
        training_config_dict['weight_decay'] = float(training_config_dict['weight_decay'])
    
    # Override device if CUDA is not available
    if args.device:
        training_config_dict['device'] = args.device
    elif training_config_dict.get('device', 'cuda') == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        training_config_dict['device'] = 'cpu'
    
    training_config = TrainingConfig(
        model_config=config.get('model', {}),
        **training_config_dict
    )
    
    # Create training loop
    print("Creating training loop...")
    training_loop = TrainingLoop(
        model=model,
        train_loader=dataloader,
        val_loader=None,  # No validation for now
        config=training_config
    )
    
    # Start training
    print("Starting training...")
    try:
        training_loop.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        training_loop.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
