#!/usr/bin/env python3
"""
Inference script for FFN models.

This script provides a command-line interface for running FFN inference
on EM volumes with various configuration options.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import yaml

# Add torchffn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchffn.models import FFN3D
from torchffn.inference import FloodFillEngine, FloodFillConfig
from torchffn.data import load_volume, save_volume
from torchffn.inference.tiler import create_tiler_for_ffn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run FFN inference")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--raw", type=str, required=True,
                       help="Path to raw volume file")
    parser.add_argument("--out", type=str, required=True,
                       help="Output path for segmented volume")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                       help="Path to inference configuration file")
    
    # Model options
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    
    # Seeding options
    parser.add_argument("--seeds", type=str, default=None,
                       help="Path to seed coordinates file (CSV format)")
    parser.add_argument("--grid-seeding", action="store_true",
                       help="Use grid-based seeding")
    parser.add_argument("--grid-spacing", type=int, nargs=3, default=[32, 32, 16],
                       help="Grid spacing for seeding [d, h, w]")
    
    # Inference parameters
    parser.add_argument("--fov-size", type=int, nargs=3, default=None,
                       help="Field of view size [d, h, w]")
    parser.add_argument("--move-threshold", type=float, default=None,
                       help="Movement threshold")
    parser.add_argument("--accept-threshold", type=float, default=None,
                       help="Object acceptance threshold")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum flood-fill steps")
    
    # Tiling options
    parser.add_argument("--tile-size", type=int, nargs=3, default=None,
                       help="Tile size for large volumes [d, h, w]")
    parser.add_argument("--memory-limit", type=float, default=None,
                       help="Memory limit in GB")
    
    # Output options
    parser.add_argument("--output-format", type=str, default=None,
                       choices=["zarr", "hdf5", "n5", "nifti"],
                       help="Output format")
    parser.add_argument("--compression", type=str, default=None,
                       help="Compression method")
    
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


def load_inference_config(config_path: str, args) -> FloodFillConfig:
    """Load inference configuration."""
    if config_path:
        config = FloodFillConfig.from_yaml(config_path)
    else:
        # Default configuration
        config = FloodFillConfig()
    
    # Override with command line arguments
    if args.fov_size:
        config.fov_size = tuple(args.fov_size)
    
    if args.move_threshold is not None:
        config.move_threshold = args.move_threshold
    
    if args.accept_threshold is not None:
        config.accept_threshold = args.accept_threshold
    
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    
    return config


def load_seeds(seeds_path: str) -> list:
    """Load seed coordinates from file."""
    seeds = []
    
    with open(seeds_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse CSV format: d,h,w or d h w
                if ',' in line:
                    coords = [int(x.strip()) for x in line.split(',')]
                else:
                    coords = [int(x.strip()) for x in line.split()]
                
                if len(coords) == 3:
                    seeds.append(tuple(coords))
    
    return seeds


def generate_grid_seeds(volume_shape: tuple, grid_spacing: tuple) -> list:
    """Generate grid-based seed coordinates."""
    seeds = []
    
    for d in range(0, volume_shape[0], grid_spacing[0]):
        for h in range(0, volume_shape[1], grid_spacing[1]):
            for w in range(0, volume_shape[2], grid_spacing[2]):
                seeds.append((d, h, w))
    
    return seeds


def run_inference_small_volume(
    model: FFN3D,
    raw_volume: torch.Tensor,
    seeds: list,
    config: FloodFillConfig,
    device: str
) -> torch.Tensor:
    """Run inference on a small volume that fits in memory."""
    print("Running inference on small volume...")
    
    # Create flood-fill engine
    engine = FloodFillEngine(model, config, device)
    
    # Segment volume
    labeled_volume = engine.segment_volume(raw_volume, seeds)
    
    return labeled_volume


def run_inference_large_volume(
    model: FFN3D,
    raw_volume: torch.Tensor,
    seeds: list,
    config: FloodFillConfig,
    device: str,
    tile_size: tuple,
    memory_limit: float
) -> torch.Tensor:
    """Run inference on a large volume using tiling."""
    print("Running inference on large volume with tiling...")
    
    # Create tiler
    tiler = create_tiler_for_ffn(
        volume_shape=raw_volume.shape,
        fov_size=config.fov_size,
        memory_limit_gb=memory_limit
    )
    
    # Override tile size if specified
    if tile_size:
        tiler.tile_size = tile_size
    
    # Create flood-fill engine
    engine = FloodFillEngine(model, config, device)
    
    # Initialize output volume
    labeled_volume = torch.zeros_like(raw_volume, dtype=torch.int32)
    
    # Process each tile
    for tile_idx in tiler.get_tile_coords():
        print(f"Processing tile {tile_idx}")
        
        # Extract tile
        tile = tiler.extract_tile(raw_volume, tile_idx, with_halo=True)
        
        # Find seeds within this tile
        tile_seeds = []
        (d_start, d_end), (h_start, h_end), (w_start, w_end) = tiler.get_halo_bounds(tile_idx)
        
        for seed in seeds:
            seed_d, seed_h, seed_w = seed
            if (d_start <= seed_d < d_end and 
                h_start <= seed_h < h_end and 
                w_start <= seed_w < w_end):
                # Convert to tile coordinates
                tile_seed = (seed_d - d_start, seed_h - h_start, seed_w - w_start)
                tile_seeds.append(tile_seed)
        
        if tile_seeds:
            # Run inference on tile
            tile_labels = engine.segment_volume(tile, tile_seeds)
            
            # Place tile back in output volume
            tiler.place_tile(labeled_volume, tile_labels, tile_idx, with_halo=True, blend_method="max")
    
    return labeled_volume


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    model = load_model(args.model, device)
    
    # Load inference configuration
    config = load_inference_config(args.config, args)
    
    # Load raw volume
    print(f"Loading raw volume from {args.raw}")
    raw_volume = load_volume(args.raw, device=device)
    print(f"Raw volume shape: {raw_volume.shape}")
    
    # Load or generate seeds
    if args.seeds:
        seeds = load_seeds(args.seeds)
        print(f"Loaded {len(seeds)} seeds from file")
    elif args.grid_seeding:
        seeds = generate_grid_seeds(raw_volume.shape, tuple(args.grid_spacing))
        print(f"Generated {len(seeds)} grid seeds")
    else:
        # Default: use center of volume as seed
        center = (raw_volume.shape[0] // 2, raw_volume.shape[1] // 2, raw_volume.shape[2] // 2)
        seeds = [center]
        print(f"Using center seed: {center}")
    
    # Determine if we need tiling
    volume_size_gb = raw_volume.numel() * 4 / (1024**3)  # Assuming float32
    use_tiling = volume_size_gb > 2.0  # Use tiling for volumes > 2GB
    
    if use_tiling:
        print("Volume is large, using tiling")
        labeled_volume = run_inference_large_volume(
            model, raw_volume, seeds, config, device,
            tile_size=tuple(args.tile_size) if args.tile_size else None,
            memory_limit=args.memory_limit or 8.0
        )
    else:
        print("Volume is small, processing in memory")
        labeled_volume = run_inference_small_volume(
            model, raw_volume, seeds, config, device
        )
    
    # Save results
    print(f"Saving labeled volume to {args.out}")
    
    # Determine output format
    output_format = args.output_format
    if not output_format:
        # Infer from file extension
        suffix = Path(args.out).suffix.lower()
        if suffix in ['.zarr']:
            output_format = 'zarr'
        elif suffix in ['.h5', '.hdf5']:
            output_format = 'hdf5'
        elif suffix in ['.n5']:
            output_format = 'n5'
        elif suffix in ['.nii', '.nii.gz']:
            output_format = 'nifti'
        else:
            output_format = 'zarr'  # Default
    
    # Save volume
    save_volume(labeled_volume, args.out, compression=args.compression)
    
    # Print statistics
    unique_labels = torch.unique(labeled_volume)
    num_objects = len(unique_labels[unique_labels > 0])
    print(f"Segmentation completed!")
    print(f"Number of objects: {num_objects}")
    print(f"Output format: {output_format}")


if __name__ == "__main__":
    main()
