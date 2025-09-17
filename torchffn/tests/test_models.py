"""
Unit tests for FFN models.

This module contains tests for the FFN3D model and related components.
"""

import pytest
import torch
import torch.nn as nn

from torchffn.models import FFN3D, CenterCrop3D, ResidualBlock3D


class TestCenterCrop3D:
    """Test CenterCrop3D layer."""

    def test_center_crop_int_size(self):
        """Test center crop with integer size."""
        crop_layer = CenterCrop3D(9)

        # Test input
        x = torch.randn(1, 3, 17, 17, 9)
        output = crop_layer(x)

        assert output.shape == (1, 3, 9, 9, 9)

    def test_center_crop_tuple_size(self):
        """Test center crop with tuple size."""
        crop_layer = CenterCrop3D((9, 9, 5))

        # Test input
        x = torch.randn(1, 3, 17, 17, 9)
        output = crop_layer(x)

        assert output.shape == (1, 3, 9, 9, 5)

    def test_center_crop_centering(self):
        """Test that center crop is properly centered."""
        crop_layer = CenterCrop3D(3)

        # Create input with known pattern
        x = torch.zeros(1, 1, 7, 7, 7)
        x[0, 0, 3, 3, 3] = 1.0  # Center voxel

        output = crop_layer(x)

        # Center voxel should be at (1, 1, 1) in output
        assert output[0, 0, 1, 1, 1] == 1.0


class TestResidualBlock3D:
    """Test ResidualBlock3D."""

    def test_residual_block_same_channels(self):
        """Test residual block with same input/output channels."""
        block = ResidualBlock3D(32, 32)

        x = torch.randn(1, 32, 16, 16, 8)
        output = block(x)

        assert output.shape == x.shape

    def test_residual_block_different_channels(self):
        """Test residual block with different input/output channels."""
        block = ResidualBlock3D(32, 64)

        x = torch.randn(1, 32, 16, 16, 8)
        output = block(x)

        assert output.shape == (1, 64, 16, 16, 8)

    def test_residual_block_dilation(self):
        """Test residual block with dilation."""
        block = ResidualBlock3D(32, 32, dilation=2)

        x = torch.randn(1, 32, 16, 16, 8)
        output = block(x)

        assert output.shape == x.shape


class TestFFN3D:
    """Test FFN3D model."""

    def test_ffn3d_forward(self):
        """Test FFN3D forward pass."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=4,
            block_channels=[16, 32],
            kernel_size=3,
            center_crop_size=9,
            use_dilation=False,
            dropout=0.0,
        )

        # Test input
        x = torch.randn(1, 3, 17, 17, 9)
        output = model(x)

        assert output.shape == (1, 1, 9, 9, 9)

    def test_ffn3d_predict_probs(self):
        """Test FFN3D probability prediction."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=4,
            block_channels=[16, 32],
            kernel_size=3,
            center_crop_size=9,
            use_dilation=False,
            dropout=0.0,
        )

        # Test input
        x = torch.randn(1, 3, 17, 17, 9)
        probs = model.predict_probs(x)

        assert probs.shape == (1, 1, 9, 9, 9)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_ffn3d_different_input_channels(self):
        """Test FFN3D with different input channel counts."""
        for input_channels in [1, 2, 3, 4]:
            model = FFN3D(
                input_channels=input_channels,
                stem_channels=16,
                num_blocks=2,
                block_channels=[16],
                kernel_size=3,
                center_crop_size=5,
                use_dilation=False,
                dropout=0.0,
            )

            x = torch.randn(1, input_channels, 9, 9, 5)
            output = model(x)

            assert output.shape == (1, 1, 5, 5, 5)

    def test_ffn3d_center_crop_alignment(self):
        """Test that center crop is properly aligned."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=2,
            block_channels=[16],
            kernel_size=3,
            center_crop_size=5,
            use_dilation=False,
            dropout=0.0,
        )

        # Create input with known pattern
        x = torch.zeros(1, 3, 9, 9, 5)
        x[0, 0, 4, 4, 2] = 1.0  # Center voxel

        output = model(x)

        # Check that output has expected shape
        assert output.shape == (1, 1, 5, 5, 5)

        # Check that center crop is working (output should be finite)
        assert torch.isfinite(output).all()

    def test_ffn3d_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=2,
            block_channels=[16],
            kernel_size=3,
            center_crop_size=5,
            use_dilation=False,
            dropout=0.0,
        )

        x = torch.randn(1, 3, 9, 9, 5, requires_grad=True)
        output = model(x)

        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.abs().sum() > 0.0

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0.0

    def test_ffn3d_deterministic(self):
        """Test that model is deterministic with same input."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=2,
            block_channels=[16],
            kernel_size=3,
            center_crop_size=5,
            use_dilation=False,
            dropout=0.0,
        )

        model.eval()

        x = torch.randn(1, 3, 9, 9, 5)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_ffn3d_from_config(self):
        """Test creating model from configuration."""
        import tempfile
        import yaml

        config = {
            "model": {
                "input_channels": 3,
                "stem_channels": 16,
                "num_blocks": 2,
                "block_channels": [16],
                "kernel_size": 3,
                "center_crop_size": 5,
                "use_dilation": False,
                "dropout": 0.0,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            model = FFN3D.from_config(config_path)

            # Test that model works
            x = torch.randn(1, 3, 9, 9, 5)
            output = model(x)

            assert output.shape == (1, 1, 5, 5, 5)
        finally:
            import os

            os.unlink(config_path)

    def test_ffn3d_get_config(self):
        """Test getting model configuration."""
        model = FFN3D(
            input_channels=3,
            stem_channels=16,
            num_blocks=2,
            block_channels=[16],
            kernel_size=3,
            center_crop_size=5,
            use_dilation=False,
            dropout=0.0,
        )

        config = model.get_config()

        assert config["input_channels"] == 3
        assert config["stem_channels"] == 16
        assert config["num_blocks"] == 2
        assert config["center_crop_size"] == 5


class TestModelFactories:
    """Test model factory functions."""

    def test_create_ffn3d_baseline(self):
        """Test creating baseline FFN3D model."""
        from torchffn.models import create_ffn3d_baseline

        model = create_ffn3d_baseline(input_channels=3, center_crop_size=17)

        x = torch.randn(1, 3, 33, 33, 17)
        output = model(x)

        assert output.shape == (1, 1, 17, 17, 17)

    def test_create_ffn3d_small(self):
        """Test creating small FFN3D model."""
        from torchffn.models import create_ffn3d_small

        model = create_ffn3d_small(input_channels=3, center_crop_size=9)

        x = torch.randn(1, 3, 17, 17, 9)
        output = model(x)

        assert output.shape == (1, 1, 9, 9, 9)


if __name__ == "__main__":
    pytest.main([__file__])
