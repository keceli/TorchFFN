"""
Unit tests for FFN inference engine.

This module contains tests for the flood-fill inference engine and related components.
"""

import pytest
import torch
import numpy as np

from torchffn.models import FFN3D
from torchffn.inference import FloodFillEngine, FloodFillConfig


class TestFloodFillConfig:
    """Test FloodFillConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FloodFillConfig()

        assert config.fov_size == (33, 33, 17)
        assert config.center_crop_size == (17, 17, 9)
        assert config.move_delta == (8, 8, 4)
        assert config.move_threshold == 0.5
        assert config.accept_threshold == 0.5
        assert config.stop_threshold == 0.1
        assert config.revisit_delta == 0.1
        assert config.max_voxels == 1000000
        assert config.max_steps == 10000
        assert config.seed_radius == 2.0
        assert config.seed_amplitude == 1.0
        assert config.queue_strategy == "max_prob"

    def test_custom_config(self):
        """Test custom configuration."""
        config = FloodFillConfig(
            fov_size=(17, 17, 9),
            center_crop_size=(9, 9, 5),
            move_delta=(4, 4, 2),
            move_threshold=0.3,
            accept_threshold=0.7,
            stop_threshold=0.05,
            revisit_delta=0.05,
            max_voxels=10000,
            max_steps=1000,
            seed_radius=1.5,
            seed_amplitude=0.8,
            queue_strategy="fifo",
        )

        assert config.fov_size == (17, 17, 9)
        assert config.center_crop_size == (9, 9, 5)
        assert config.move_delta == (4, 4, 2)
        assert config.move_threshold == 0.3
        assert config.accept_threshold == 0.7
        assert config.stop_threshold == 0.05
        assert config.revisit_delta == 0.05
        assert config.max_voxels == 10000
        assert config.max_steps == 1000
        assert config.seed_radius == 1.5
        assert config.seed_amplitude == 0.8
        assert config.queue_strategy == "fifo"


class TestFloodFillEngine:
    """Test FloodFillEngine."""

    def create_test_model(self):
        """Create a test model."""
        return FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=2,
            block_channels=[16],
            kernel_size=3,
            center_crop_size=(9, 9, 5),
            use_dilation=False,
            dropout=0.0,
        )

    def create_test_volume(self, size=(32, 32, 16)):
        """Create a test volume with a simple object."""
        volume = torch.randn(size) * 0.1

        # Add a simple object in the center
        center = (size[0] // 2, size[1] // 2, size[2] // 2)
        radius = 4

        for d in range(size[0]):
            for h in range(size[1]):
                for w in range(size[2]):
                    dist = (
                        (d - center[0]) ** 2
                        + (h - center[1]) ** 2
                        + (w - center[2]) ** 2
                    ) ** 0.5
                    if dist <= radius:
                        volume[d, h, w] += 0.8

        return volume

    def test_engine_initialization(self):
        """Test engine initialization."""
        model = self.create_test_model()
        config = FloodFillConfig()

        engine = FloodFillEngine(model, config, device="cpu")

        assert engine.model is model
        assert engine.config is config
        assert engine.device == "cpu"

    def test_engine_initialization_with_config_string(self):
        """Test engine initialization with config file path."""
        import tempfile
        import yaml

        model = self.create_test_model()

        config = {
            "inference": {
                "fov_size": [17, 17, 9],
                "center_crop_size": [9, 9, 5],
                "move_delta": [4, 4, 2],
                "move_threshold": 0.5,
                "accept_threshold": 0.5,
                "stop_threshold": 0.1,
                "revisit_delta": 0.1,
                "max_voxels": 10000,
                "max_steps": 1000,
                "seed_radius": 2.0,
                "seed_amplitude": 1.0,
                "queue_strategy": "max_prob",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            engine = FloodFillEngine(model, config_path, device="cpu")

            assert engine.config.fov_size == (17, 17, 9)
            assert engine.config.center_crop_size == (9, 9, 5)
        finally:
            import os

            os.unlink(config_path)

    def test_create_seed_mask(self):
        """Test seed mask creation."""
        model = self.create_test_model()
        config = FloodFillConfig(fov_size=(17, 17, 9))
        engine = FloodFillEngine(model, config, device="cpu")

        seed_pos = (8, 8, 4)  # Center of FOV
        seed_mask = engine._create_seed_mask(config.fov_size, seed_pos)

        assert seed_mask.shape == config.fov_size
        assert seed_mask.max() > 0.0
        assert seed_mask[seed_pos] > 0.0

    def test_extract_fov(self):
        """Test FOV extraction."""
        model = self.create_test_model()
        config = FloodFillConfig(fov_size=(17, 17, 9))
        engine = FloodFillEngine(model, config, device="cpu")

        volume = torch.randn(32, 32, 16)
        center = (16, 16, 8)

        fov = engine._extract_fov(volume, center, config.fov_size)

        assert fov.shape == config.fov_size

    def test_extract_fov_boundary(self):
        """Test FOV extraction at volume boundaries."""
        model = self.create_test_model()
        config = FloodFillConfig(fov_size=(17, 17, 9))
        engine = FloodFillEngine(model, config, device="cpu")

        volume = torch.randn(32, 32, 16)

        # Test corner case
        center = (0, 0, 0)
        fov = engine._extract_fov(volume, center, config.fov_size)

        assert fov.shape == config.fov_size

        # Test edge case
        center = (31, 31, 15)
        fov = engine._extract_fov(volume, center, config.fov_size)

        assert fov.shape == config.fov_size

    def test_get_neighbor_positions(self):
        """Test neighbor position generation."""
        model = self.create_test_model()
        config = FloodFillConfig(move_delta=(4, 4, 2))
        engine = FloodFillEngine(model, config, device="cpu")

        center = (16, 16, 8)
        neighbors = engine._get_neighbor_positions(center)

        assert len(neighbors) == 26  # 3^3 - 1 (excluding center)
        assert center not in neighbors

        # Check that all neighbors are at correct distances
        for neighbor in neighbors:
            dist_d = abs(neighbor[0] - center[0])
            dist_h = abs(neighbor[1] - center[1])
            dist_w = abs(neighbor[2] - center[2])

            assert dist_d in [0, 4]
            assert dist_h in [0, 4]
            assert dist_w in [0, 2]

    def test_should_move(self):
        """Test movement decision logic."""
        model = self.create_test_model()
        config = FloodFillConfig(
            fov_size=(17, 17, 9), center_crop_size=(9, 9, 5), move_threshold=0.5
        )
        engine = FloodFillEngine(model, config, device="cpu")

        # Create test probabilities
        probs = torch.zeros(config.fov_size)

        # Test case 1: No movement (low probabilities)
        probs[4, 4, 2] = 0.3  # Center
        probs[0, 4, 2] = 0.2  # Edge
        assert not engine._should_move(probs, config.center_crop_size)

        # Test case 2: Movement (high edge probability)
        probs[0, 4, 2] = 0.7  # Edge above threshold
        assert engine._should_move(probs, config.center_crop_size)

    def test_update_object_map(self):
        """Test object map updating."""
        model = self.create_test_model()
        config = FloodFillConfig(fov_size=(17, 17, 9), center_crop_size=(9, 9, 5))
        engine = FloodFillEngine(model, config, device="cpu")

        # Create test data
        object_map = torch.zeros(32, 32, 16)
        probs = torch.rand(config.fov_size)
        center = (16, 16, 8)

        # Update object map
        engine._update_object_map(object_map, probs, center, config.center_crop_size)

        # Check that some values were updated
        assert object_map.sum() > 0.0

    def test_flood_fill_from_seed(self):
        """Test flood-fill from a single seed."""
        model = self.create_test_model()
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
        engine = FloodFillEngine(model, config, device="cpu")

        # Create test volume
        volume = self.create_test_volume()
        seed_position = (16, 16, 8)  # Center of volume

        # Run flood-fill
        object_mask, stats = engine.flood_fill_from_seed(volume, seed_position)

        # Check results
        assert object_mask.shape == volume.shape
        assert stats["steps"] > 0
        assert stats["steps"] <= config.max_steps
        assert stats["object_size"] >= 0
        assert stats["max_prob"] >= 0.0
        assert stats["mean_prob"] >= 0.0

    def test_flood_fill_empty_volume(self):
        """Test flood-fill on empty volume (no objects)."""
        model = self.create_test_model()
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
        engine = FloodFillEngine(model, config, device="cpu")

        # Create empty volume (just noise)
        volume = torch.randn(32, 32, 16) * 0.1
        seed_position = (16, 16, 8)

        # Run flood-fill
        object_mask, stats = engine.flood_fill_from_seed(volume, seed_position)

        # Check results
        assert object_mask.shape == volume.shape
        assert object_mask.sum() == 0.0  # No object should be found
        # Note: stats['object_size'] is the size before rejection, so it can be > 0
        # The important check is that the final mask is empty

    def test_segment_volume(self):
        """Test volume segmentation with multiple seeds."""
        model = self.create_test_model()
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
        engine = FloodFillEngine(model, config, device="cpu")

        # Create test volume
        volume = self.create_test_volume()
        seed_positions = [(16, 16, 8), (8, 8, 4), (24, 24, 12)]

        # Run segmentation
        labeled_volume = engine.segment_volume(volume, seed_positions)

        # Check results
        assert labeled_volume.shape == volume.shape
        assert labeled_volume.dtype == torch.int32

        # Check that we have some labeled objects
        unique_labels = torch.unique(labeled_volume)
        assert len(unique_labels) > 1  # Should have background + at least one object

    def test_engine_deterministic(self):
        """Test that engine is deterministic with same input."""
        model = self.create_test_model()
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
        engine = FloodFillEngine(model, config, device="cpu")

        # Create test volume
        volume = self.create_test_volume()
        seed_position = (16, 16, 8)

        # Run flood-fill twice
        object_mask1, stats1 = engine.flood_fill_from_seed(volume, seed_position)
        object_mask2, stats2 = engine.flood_fill_from_seed(volume, seed_position)

        # Check that results are identical
        assert torch.allclose(object_mask1, object_mask2, atol=1e-6)
        assert stats1["steps"] == stats2["steps"]
        assert abs(stats1["max_prob"] - stats2["max_prob"]) < 1e-6
        assert abs(stats1["mean_prob"] - stats2["mean_prob"]) < 1e-6
        assert stats1["object_size"] == stats2["object_size"]


if __name__ == "__main__":
    pytest.main([__file__])
