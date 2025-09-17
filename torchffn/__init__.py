"""
TorchFFN: A PyTorch implementation of Flood-Filling Networks for EM connectomics.

This package provides a faithful port of Google's FFN algorithm for 3D volumetric
segmentation, with modern engineering practices and production-ready features.
"""

__version__ = "0.1.0"
__author__ = "TorchFFN Contributors"

from .models import FFN3D
from .inference import FloodFillEngine

__all__ = ["FFN3D", "FloodFillEngine"]
