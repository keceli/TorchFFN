"""
Volume tiling utilities for handling large volumes in memory-constrained environments.

This module provides utilities for chunked I/O over large volumes with halo regions
and memory-safe stitching.
"""

import math
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch


class VolumeTiler:
    """
    Utility class for tiling large volumes into manageable chunks with halo regions.
    
    This is essential for processing large EM volumes that don't fit in memory.
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        tile_size: Tuple[int, int, int],
        halo_size: Tuple[int, int, int] = (0, 0, 0),
        overlap: Tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Args:
            volume_shape: Shape of the full volume (D, H, W)
            tile_size: Size of each tile (D, H, W)
            halo_size: Halo size around each tile (D, H, W)
            overlap: Overlap between adjacent tiles (D, H, W)
        """
        self.volume_shape = volume_shape
        self.tile_size = tile_size
        self.halo_size = halo_size
        self.overlap = overlap
        
        # Calculate effective tile size (including halo)
        self.effective_tile_size = (
            tile_size[0] + 2 * halo_size[0],
            tile_size[1] + 2 * halo_size[1],
            tile_size[2] + 2 * halo_size[2],
        )
        
        # Calculate number of tiles needed
        self.num_tiles = (
            math.ceil(volume_shape[0] / (tile_size[0] - overlap[0])),
            math.ceil(volume_shape[1] / (tile_size[1] - overlap[1])),
            math.ceil(volume_shape[2] / (tile_size[2] - overlap[2])),
        )
    
    def get_tile_bounds(self, tile_idx: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the bounds of a tile in the full volume.
        
        Args:
            tile_idx: (d, h, w) tile index
            
        Returns:
            Tuple of ((d_start, d_end), (h_start, h_end), (w_start, w_end))
        """
        d_idx, h_idx, w_idx = tile_idx
        
        # Calculate step size (accounting for overlap)
        d_step = self.tile_size[0] - self.overlap[0]
        h_step = self.tile_size[1] - self.overlap[1]
        w_step = self.tile_size[2] - self.overlap[2]
        
        # Calculate tile bounds in volume coordinates
        d_start = d_idx * d_step
        h_start = h_idx * h_step
        w_start = w_idx * w_step
        
        d_end = min(d_start + self.tile_size[0], self.volume_shape[0])
        h_end = min(h_start + self.tile_size[1], self.volume_shape[1])
        w_end = min(w_start + self.tile_size[2], self.volume_shape[2])
        
        return (d_start, d_end), (h_start, h_end), (w_start, w_end)
    
    def get_halo_bounds(self, tile_idx: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the bounds of a tile including halo region.
        
        Args:
            tile_idx: (d, h, w) tile index
            
        Returns:
            Tuple of ((d_start, d_end), (h_start, h_end), (w_start, w_end)) including halo
        """
        (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_tile_bounds(tile_idx)
        
        # Add halo
        d_start = max(0, d_start - self.halo_size[0])
        h_start = max(0, h_start - self.halo_size[1])
        w_start = max(0, w_start - self.halo_size[2])
        
        d_end = min(self.volume_shape[0], d_end + self.halo_size[0])
        h_end = min(self.volume_shape[1], h_end + self.halo_size[1])
        w_end = min(self.volume_shape[2], w_end + self.halo_size[2])
        
        return (d_start, d_end), (h_start, h_end), (w_start, w_end)
    
    def get_tile_coords(self) -> Iterator[Tuple[int, int, int]]:
        """Generate all tile coordinates."""
        for d_idx in range(self.num_tiles[0]):
            for h_idx in range(self.num_tiles[1]):
                for w_idx in range(self.num_tiles[2]):
                    yield (d_idx, h_idx, w_idx)
    
    def extract_tile(
        self, 
        volume: torch.Tensor, 
        tile_idx: Tuple[int, int, int],
        with_halo: bool = True
    ) -> torch.Tensor:
        """
        Extract a tile from the volume.
        
        Args:
            volume: Full volume tensor [D, H, W]
            tile_idx: (d, h, w) tile index
            with_halo: Whether to include halo region
            
        Returns:
            Tile tensor
        """
        if with_halo:
            (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_halo_bounds(tile_idx)
        else:
            (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_tile_bounds(tile_idx)
        
        return volume[d_start:d_end, h_start:h_end, w_start:w_end]
    
    def place_tile(
        self,
        output_volume: torch.Tensor,
        tile: torch.Tensor,
        tile_idx: Tuple[int, int, int],
        with_halo: bool = True,
        blend_method: str = "max"
    ):
        """
        Place a tile back into the output volume.
        
        Args:
            output_volume: Output volume tensor to update
            tile: Tile tensor to place
            tile_idx: (d, h, w) tile index
            with_halo: Whether tile includes halo region
            blend_method: Method for blending overlapping regions ("max", "mean", "overwrite")
        """
        if with_halo:
            (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_halo_bounds(tile_idx)
            # Remove halo from tile
            tile = tile[
                self.halo_size[0]:tile.shape[0] - self.halo_size[0],
                self.halo_size[1]:tile.shape[1] - self.halo_size[1],
                self.halo_size[2]:tile.shape[2] - self.halo_size[2],
            ]
        else:
            (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_tile_bounds(tile_idx)
        
        # Ensure bounds are within output volume
        d_start = max(0, d_start)
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        d_end = min(output_volume.shape[0], d_end)
        h_end = min(output_volume.shape[1], h_end)
        w_end = min(output_volume.shape[2], w_end)
        
        # Adjust tile if needed
        tile_d = d_end - d_start
        tile_h = h_end - h_start
        tile_w = w_end - w_start
        
        if tile.shape != (tile_d, tile_h, tile_w):
            tile = tile[:tile_d, :tile_h, :tile_w]
        
        # Blend with existing values
        if blend_method == "max":
            output_volume[d_start:d_end, h_start:h_end, w_start:w_end] = torch.maximum(
                output_volume[d_start:d_end, h_start:h_end, w_start:w_end],
                tile
            )
        elif blend_method == "mean":
            existing = output_volume[d_start:d_end, h_start:h_end, w_start:w_end]
            output_volume[d_start:d_end, h_start:h_end, w_start:w_end] = (existing + tile) / 2
        else:  # overwrite
            output_volume[d_start:d_end, h_start:h_end, w_start:w_end] = tile
    
    def get_tile_info(self, tile_idx: Tuple[int, int, int]) -> dict:
        """Get information about a tile."""
        (d_start, d_end), (h_start, h_end), (w_start, w_end) = self.get_tile_bounds(tile_idx)
        (d_halo_start, d_halo_end), (h_halo_start, h_halo_end), (w_halo_start, w_halo_end) = self.get_halo_bounds(tile_idx)
        
        return {
            'tile_idx': tile_idx,
            'tile_bounds': ((d_start, d_end), (h_start, h_end), (w_start, w_end)),
            'halo_bounds': ((d_halo_start, d_halo_end), (h_halo_start, h_halo_end), (w_halo_start, w_halo_end)),
            'tile_size': (d_end - d_start, h_end - h_start, w_end - w_start),
            'halo_size': (d_halo_end - d_halo_start, h_halo_end - h_halo_start, w_halo_end - w_halo_start),
        }


def create_tiler_for_ffn(
    volume_shape: Tuple[int, int, int],
    fov_size: Tuple[int, int, int] = (33, 33, 17),
    memory_limit_gb: float = 8.0,
) -> VolumeTiler:
    """
    Create a tiler optimized for FFN inference.
    
    Args:
        volume_shape: Shape of the full volume (D, H, W)
        fov_size: FFN field of view size (D, H, W)
        memory_limit_gb: Memory limit in GB
        
    Returns:
        VolumeTiler instance
    """
    # Estimate memory usage per tile
    # Assume float32 (4 bytes) and some overhead
    bytes_per_voxel = 4
    memory_per_voxel = bytes_per_voxel * 3  # raw + mask + seed channels
    
    # Calculate maximum tile size that fits in memory
    max_voxels = int(memory_limit_gb * 1e9 / memory_per_voxel)
    
    # Start with a reasonable tile size and adjust
    tile_size = list(fov_size)
    while tile_size[0] * tile_size[1] * tile_size[2] > max_voxels:
        # Reduce largest dimension
        max_dim = max(range(3), key=lambda i: tile_size[i])
        tile_size[max_dim] = max(tile_size[max_dim] // 2, fov_size[max_dim])
    
    # Add some overlap to handle boundary effects
    overlap = (fov_size[0] // 2, fov_size[1] // 2, fov_size[2] // 2)
    
    # Add halo for better boundary handling
    halo_size = (fov_size[0] // 2, fov_size[1] // 2, fov_size[2] // 2)
    
    return VolumeTiler(
        volume_shape=volume_shape,
        tile_size=tuple(tile_size),
        halo_size=halo_size,
        overlap=overlap,
    )
