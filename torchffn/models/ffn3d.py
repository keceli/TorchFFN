"""
3D CNN model for Flood-Filling Networks.

This module implements the core FFN3D model with residual blocks, center crop supervision,
and support for mixed precision training.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class CenterCrop3D(nn.Module):
    """3D center crop layer for center-voxel supervision."""

    def __init__(self, crop_size: Union[int, Tuple[int, int, int]]):
        """
        Args:
            crop_size: Size of the center crop. If int, applies to all dimensions.
                      If tuple, should be (depth, height, width).
        """
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Center-cropped tensor of shape [B, C, crop_d, crop_h, crop_w]
        """
        B, C, D, H, W = x.shape
        crop_d, crop_h, crop_w = self.crop_size

        # Calculate start indices for center crop
        start_d = (D - crop_d) // 2
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        return x[
            :,
            :,
            start_d : start_d + crop_d,
            start_h : start_h + crop_h,
            start_w : start_w + crop_w,
        ]


class ResidualBlock3D(nn.Module):
    """3D residual block without batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        # Projection shortcut if input/output channels differ
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)

        out += residual
        out = F.relu(out)

        return out


class FFN3D(nn.Module):
    """
    3D CNN model for Flood-Filling Networks.

    This model takes a 3D volume with optional mask and seed channels and predicts
    object probabilities for the center region of the field of view.
    """

    def __init__(
        self,
        input_channels: int = 3,  # raw + mask + seed
        stem_channels: int = 32,
        num_blocks: int = 8,
        block_channels: List[int] = None,
        kernel_size: int = 3,
        center_crop_size: Union[int, Tuple[int, int, int]] = 17,
        use_dilation: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_channels: Number of input channels (raw + mask + optional seed)
            stem_channels: Number of channels in the stem layer
            num_blocks: Number of residual blocks
            block_channels: List of channel counts for each block stage
            kernel_size: Kernel size for convolutions
            center_crop_size: Size of center crop for supervision
            use_dilation: Whether to use dilated convolutions in deeper blocks
            dropout: Dropout rate
        """
        super().__init__()

        if block_channels is None:
            # Default channel progression: 32 -> 64 -> 96
            block_channels = [stem_channels, stem_channels * 2, stem_channels * 3]

        self.input_channels = input_channels
        self.center_crop_size = (
            center_crop_size if isinstance(center_crop_size, int) else center_crop_size
        )

        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                stem_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        current_channels = stem_channels

        for i in range(num_blocks):
            # Determine output channels for this block
            stage_idx = min(
                i // (num_blocks // len(block_channels)), len(block_channels) - 1
            )
            out_channels = block_channels[stage_idx]

            # Use dilation in deeper blocks if enabled
            dilation = 2 if use_dilation and i >= num_blocks // 2 else 1

            self.blocks.append(
                ResidualBlock3D(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        # Output head
        self.head = nn.Conv3d(current_channels, 1, kernel_size=1, bias=True)

        # Center crop layer
        self.center_crop = CenterCrop3D(center_crop_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, D, H, W] where C includes raw + mask + optional seed

        Returns:
            Logits for center region of shape [B, 1, crop_d, crop_h, crop_w]
        """
        # Stem
        x = self.stem(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output head
        x = self.head(x)

        # Center crop for supervision
        x = self.center_crop(x)

        return x

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities (sigmoid of logits)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @classmethod
    def from_config(cls, config_path: str) -> "FFN3D":
        """Create model from YAML configuration file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        return cls(**model_config)

    def get_config(self) -> Dict:
        """Get model configuration as dictionary."""
        return {
            "input_channels": self.input_channels,
            "stem_channels": self.stem[0].out_channels,
            "num_blocks": len(self.blocks),
            "kernel_size": 3,  # Default, could be made configurable
            "center_crop_size": self.center_crop_size,
        }


def create_ffn3d_baseline(
    input_channels: int = 3,
    center_crop_size: Union[int, Tuple[int, int, int]] = 17,
) -> FFN3D:
    """
    Create a baseline FFN3D model with standard architecture.

    Args:
        input_channels: Number of input channels
        center_crop_size: Size of center crop for supervision

    Returns:
        FFN3D model instance
    """
    return FFN3D(
        input_channels=input_channels,
        stem_channels=32,
        num_blocks=12,
        block_channels=[32, 64, 96],
        kernel_size=3,
        center_crop_size=center_crop_size,
        use_dilation=False,
        dropout=0.0,
    )


def create_ffn3d_small(
    input_channels: int = 3,
    center_crop_size: Union[int, Tuple[int, int, int]] = 9,
) -> FFN3D:
    """
    Create a small FFN3D model for testing and quick experiments.

    Args:
        input_channels: Number of input channels
        center_crop_size: Size of center crop for supervision

    Returns:
        FFN3D model instance
    """
    return FFN3D(
        input_channels=input_channels,
        stem_channels=16,
        num_blocks=4,
        block_channels=[16, 32],
        kernel_size=3,
        center_crop_size=center_crop_size,
        use_dilation=False,
        dropout=0.0,
    )
