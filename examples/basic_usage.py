#!/usr/bin/env python3
"""
Basic usage example for TorchFFN.

This script demonstrates how to use TorchFFN for 3D volumetric segmentation
with synthetic data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import torchffn
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchffn import FFN3D, FloodFillEngine, FloodFillConfig
from torchffn.data import SyntheticDataset


def create_synthetic_volume():
    """Create a synthetic 3D volume with known objects."""
    print("🔬 Creating synthetic 3D volume...")

    # Create dataset
    dataset = SyntheticDataset(
        volume_size=(64, 64, 32),
        num_objects_range=(3, 8),
        object_size_range=(3.0, 8.0),
        noise_level=0.1,
    )

    sample = dataset[0]
    volume = sample['input'][0]  # Raw volume is the first channel
    labels = sample['target'][0]  # Target is the object mask
    print(f"   Volume shape: {volume.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Unique objects: {len(torch.unique(labels)) - 1}")  # -1 for background

    return volume, labels


def setup_model_and_engine():
    """Set up the FFN model and inference engine."""
    print("🧠 Setting up FFN model and inference engine...")

    # Create model
    model = FFN3D(
        input_channels=3,  # raw + mask + seed
        stem_channels=16,
        num_blocks=4,
        block_channels=[16, 32],
        center_crop_size=9,
        use_dilation=False,
        dropout=0.0,
    )

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create inference configuration
    config = FloodFillConfig(
        fov_size=(17, 17, 9),
        center_crop_size=(9, 9, 5),
        move_delta=(4, 4, 2),
        move_threshold=0.5,
        accept_threshold=0.5,
        stop_threshold=0.1,
        max_steps=100,
        max_voxels=1000,
    )

    # Create inference engine
    engine = FloodFillEngine(model, config, device="cpu")

    return model, engine


def run_single_seed_segmentation(engine, volume, seed_position):
    """Run flood-fill segmentation from a single seed."""
    print(f"🎯 Running segmentation from seed {seed_position}...")

    # Run flood-fill
    object_mask, stats = engine.flood_fill_from_seed(volume, seed_position)

    print(f"   Steps taken: {stats['steps']}")
    print(f"   Object size: {stats['object_size']} voxels")
    print(f"   Max probability: {stats['max_prob']:.3f}")
    print(f"   Mean probability: {stats['mean_prob']:.3f}")

    return object_mask, stats


def run_multi_seed_segmentation(engine, volume, seed_positions):
    """Run segmentation from multiple seeds."""
    print(f"🎯 Running multi-seed segmentation with {len(seed_positions)} seeds...")

    # Run segmentation
    labeled_volume = engine.segment_volume(volume, seed_positions)

    # Count objects
    unique_labels = torch.unique(labeled_volume)
    num_objects = len(unique_labels) - 1  # -1 for background

    print(f"   Found {num_objects} objects")
    print(
        f"   Label range: {unique_labels.min().item()} - {unique_labels.max().item()}"
    )

    return labeled_volume


def visualize_results(volume, labels, object_mask, labeled_volume, seed_position):
    """Visualize the segmentation results."""
    print("📊 Creating visualization...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("TorchFFN Segmentation Results", fontsize=16)

    # Get middle slice for visualization
    mid_slice = volume.shape[2] // 2

    # Original volume
    axes[0, 0].imshow(volume[:, :, mid_slice], cmap="gray")
    axes[0, 0].set_title("Original Volume")
    axes[0, 0].axis("off")

    # Ground truth labels
    axes[0, 1].imshow(labels[:, :, mid_slice], cmap="tab10")
    axes[0, 1].set_title("Ground Truth Labels")
    axes[0, 1].axis("off")

    # Single seed segmentation
    axes[0, 2].imshow(object_mask[:, :, mid_slice], cmap="Reds")
    axes[0, 2].set_title(f"Single Seed Segmentation\nSeed: {seed_position}")
    axes[0, 2].axis("off")

    # Multi-seed segmentation
    axes[1, 0].imshow(labeled_volume[:, :, mid_slice], cmap="tab10")
    axes[1, 0].set_title("Multi-Seed Segmentation")
    axes[1, 0].axis("off")

    # Overlay comparison
    overlay = volume[:, :, mid_slice].clone()
    overlay[object_mask[:, :, mid_slice] > 0] = 1.0
    axes[1, 1].imshow(overlay, cmap="gray")
    axes[1, 1].set_title("Overlay: Volume + Segmentation")
    axes[1, 1].axis("off")

    # Statistics
    axes[1, 2].text(
        0.1,
        0.9,
        "Segmentation Statistics:",
        transform=axes[1, 2].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 2].text(
        0.1,
        0.8,
        f"Volume shape: {volume.shape}",
        transform=axes[1, 2].transAxes,
        fontsize=10,
    )
    axes[1, 2].text(
        0.1,
        0.7,
        f"Ground truth objects: {len(torch.unique(labels)) - 1}",
        transform=axes[1, 2].transAxes,
        fontsize=10,
    )
    axes[1, 2].text(
        0.1,
        0.6,
        f"Segmented objects: {len(torch.unique(labeled_volume)) - 1}",
        transform=axes[1, 2].transAxes,
        fontsize=10,
    )
    axes[1, 2].text(
        0.1,
        0.5,
        f"Seed position: {seed_position}",
        transform=axes[1, 2].transAxes,
        fontsize=10,
    )
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis("off")

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "segmentation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Visualization saved to: {output_path}")

    # Show plot
    plt.show()


def main():
    """Main example function."""
    print("🚀 TorchFFN Basic Usage Example")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Create synthetic data
        volume, labels = create_synthetic_volume()

        # Setup model and engine
        model, engine = setup_model_and_engine()

        # Define seed positions
        seed_positions = [
            (32, 32, 16),  # Center
            (16, 16, 8),  # Corner
            (48, 48, 24),  # Another corner
        ]

        # Run single seed segmentation
        object_mask, stats = run_single_seed_segmentation(
            engine, volume, seed_positions[0]
        )

        # Run multi-seed segmentation
        labeled_volume = run_multi_seed_segmentation(engine, volume, seed_positions)

        # Visualize results
        visualize_results(
            volume, labels, object_mask, labeled_volume, seed_positions[0]
        )

        print("=" * 50)
        print("✅ Example completed successfully!")
        print()
        print("Next steps:")
        print("- Try different seed positions")
        print("- Experiment with different model architectures")
        print("- Use your own 3D volume data")
        print("- Check out the configuration files in configs/")

    except Exception as e:
        print(f"❌ Error running example: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
