"""
Flood-fill inference engine for FFN.

This module implements the core flood-fill algorithm that iteratively grows
object masks from seed points using a moving 3D field of view.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class FloodFillConfig:
    """Configuration for flood-fill inference."""

    # Field of view and movement parameters
    fov_size: Tuple[int, int, int] = (33, 33, 17)  # (depth, height, width)
    center_crop_size: Tuple[int, int, int] = (17, 17, 9)
    move_delta: Tuple[int, int, int] = (8, 8, 4)  # Step size in voxels

    # Thresholds
    move_threshold: float = 0.5  # Probability threshold for movement
    accept_threshold: float = 0.5  # Minimum mean probability to accept object
    stop_threshold: float = 0.1  # Minimum probability to continue growing
    revisit_delta: float = 0.1  # Minimum improvement to revisit a position

    # Object limits
    max_voxels: int = 1000000  # Maximum object size
    max_steps: int = 10000  # Maximum flood-fill steps

    # Seed parameters
    seed_radius: float = 2.0  # Gaussian seed radius
    seed_amplitude: float = 1.0  # Seed amplitude

    # Queue strategy
    queue_strategy: str = "max_prob"  # "max_prob", "fifo", "entropy"

    @classmethod
    def from_yaml(cls, config_path: str) -> "FloodFillConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        inference_config = config.get("inference", {})

        # Convert lists to tuples for size parameters
        for key in ["fov_size", "center_crop_size", "move_delta"]:
            if key in inference_config and isinstance(inference_config[key], list):
                inference_config[key] = tuple(inference_config[key])

        return cls(**inference_config)


class FloodFillEngine:
    """
    Flood-fill inference engine for FFN.

    This engine implements the core iterative algorithm that grows object masks
    from seed points using a moving 3D field of view.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[FloodFillConfig, str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: Trained FFN3D model
            config: Flood-fill configuration or path to YAML config
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device

        if isinstance(config, str):
            self.config = FloodFillConfig.from_yaml(config)
        else:
            self.config = config

        self.model.eval()

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        fov_d, fov_h, fov_w = self.config.fov_size
        crop_d, crop_h, crop_w = self.config.center_crop_size

        assert crop_d <= fov_d, f"Center crop depth {crop_d} > FOV depth {fov_d}"
        assert crop_h <= fov_h, f"Center crop height {crop_h} > FOV height {fov_h}"
        assert crop_w <= fov_w, f"Center crop width {crop_w} > FOV width {fov_w}"

        # Ensure center crop is centered
        assert (
            fov_d - crop_d
        ) % 2 == 0, "FOV and center crop depth difference must be even"
        assert (
            fov_h - crop_h
        ) % 2 == 0, "FOV and center crop height difference must be even"
        assert (
            fov_w - crop_w
        ) % 2 == 0, "FOV and center crop width difference must be even"

    def _create_seed_mask(
        self, fov_size: Tuple[int, int, int], seed_pos: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Create a Gaussian seed mask within the FOV."""
        fov_d, fov_h, fov_w = fov_size
        seed_d, seed_h, seed_w = seed_pos

        # Create coordinate grids
        d_coords = torch.arange(fov_d, dtype=torch.float32, device=self.device)
        h_coords = torch.arange(fov_h, dtype=torch.float32, device=self.device)
        w_coords = torch.arange(fov_w, dtype=torch.float32, device=self.device)

        D, H, W = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")

        # Calculate distances from seed position
        dist_sq = (D - seed_d) ** 2 + (H - seed_h) ** 2 + (W - seed_w) ** 2

        # Create Gaussian seed
        seed_mask = self.config.seed_amplitude * torch.exp(
            -dist_sq / (2 * self.config.seed_radius**2)
        )

        return seed_mask

    def _extract_fov(
        self,
        volume: torch.Tensor,
        center: Tuple[int, int, int],
        fov_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Extract field of view around center position."""
        fov_d, fov_h, fov_w = fov_size
        center_d, center_h, center_w = center

        # Calculate bounds
        start_d = center_d - fov_d // 2
        start_h = center_h - fov_h // 2
        start_w = center_w - fov_w // 2

        end_d = start_d + fov_d
        end_h = start_h + fov_h
        end_w = start_w + fov_w

        # Handle boundary conditions with padding
        pad_d = max(0, -start_d, end_d - volume.shape[0])
        pad_h = max(0, -start_h, end_h - volume.shape[1])
        pad_w = max(0, -start_w, end_w - volume.shape[2])

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = F.pad(
                volume,
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                mode="constant",
                value=0,
            )
            start_d += pad_d
            start_h += pad_h
            start_w += pad_w
            end_d += pad_d
            end_h += pad_h
            end_w += pad_w

        # Extract FOV
        fov = volume[start_d:end_d, start_h:end_h, start_w:end_w]

        return fov

    def _get_neighbor_positions(
        self, center: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Get neighboring positions for frontier expansion."""
        center_d, center_h, center_w = center
        delta_d, delta_h, delta_w = self.config.move_delta

        neighbors = []
        for dd in [-delta_d, 0, delta_d]:
            for dh in [-delta_h, 0, delta_h]:
                for dw in [-delta_w, 0, delta_w]:
                    if dd == 0 and dh == 0 and dw == 0:
                        continue
                    neighbors.append((center_d + dd, center_h + dh, center_w + dw))

        return neighbors

    def _should_move(
        self, probs: torch.Tensor, center_crop_size: Tuple[int, int, int]
    ) -> bool:
        """Check if we should move based on probability threshold outside center crop."""
        crop_d, crop_h, crop_w = center_crop_size
        fov_d, fov_h, fov_w = probs.shape

        # If center crop is the same size as FOV, check if any probability is above threshold
        if crop_d == fov_d and crop_h == fov_h and crop_w == fov_w:
            return probs.max().item() > self.config.move_threshold

        # Get center crop region
        start_d = (fov_d - crop_d) // 2
        start_h = (fov_h - crop_h) // 2
        start_w = (fov_w - crop_w) // 2

        end_d = start_d + crop_d
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        # Check for high probabilities outside the center crop
        # Create a mask for the center crop region
        center_mask = torch.zeros_like(probs, dtype=torch.bool)
        center_mask[start_d:end_d, start_h:end_h, start_w:end_w] = True

        # Get probabilities outside center crop
        outside_probs = probs[~center_mask]

        if outside_probs.numel() == 0:
            return False

        max_outside_prob = outside_probs.max().item()
        return max_outside_prob > self.config.move_threshold

    def _update_object_map(
        self,
        object_map: torch.Tensor,
        probs: torch.Tensor,
        center: Tuple[int, int, int],
        center_crop_size: Tuple[int, int, int],
        update_method: str = "max",
    ):
        """Update the global object probability map with new predictions."""
        fov_d, fov_h, fov_w = probs.shape
        crop_d, crop_h, crop_w = center_crop_size
        center_d, center_h, center_w = center

        # Extract center crop from predictions
        crop_start_d = (fov_d - crop_d) // 2
        crop_start_h = (fov_h - crop_h) // 2
        crop_start_w = (fov_w - crop_w) // 2

        crop_probs = probs[
            crop_start_d : crop_start_d + crop_d,
            crop_start_h : crop_start_h + crop_h,
            crop_start_w : crop_start_w + crop_w,
        ]

        # Calculate the region in the global object map that corresponds to this center crop
        # The center crop should be centered around the current position
        global_start_d = center_d - crop_d // 2
        global_start_h = center_h - crop_h // 2
        global_start_w = center_w - crop_w // 2

        global_end_d = global_start_d + crop_d
        global_end_h = global_start_h + crop_h
        global_end_w = global_start_w + crop_w

        # Calculate the overlap between the global region and the object map
        overlap_start_d = max(0, global_start_d)
        overlap_start_h = max(0, global_start_h)
        overlap_start_w = max(0, global_start_w)

        overlap_end_d = min(object_map.shape[0], global_end_d)
        overlap_end_h = min(object_map.shape[1], global_end_h)
        overlap_end_w = min(object_map.shape[2], global_end_w)

        # Ensure overlap bounds are valid
        if (
            overlap_end_d <= overlap_start_d
            or overlap_end_h <= overlap_start_h
            or overlap_end_w <= overlap_start_w
        ):
            # No overlap, return early
            return

        # Calculate the corresponding region in the crop
        crop_offset_d = overlap_start_d - global_start_d
        crop_offset_h = overlap_start_h - global_start_h
        crop_offset_w = overlap_start_w - global_start_w

        crop_end_d = crop_offset_d + (overlap_end_d - overlap_start_d)
        crop_end_h = crop_offset_h + (overlap_end_h - overlap_start_h)
        crop_end_w = crop_offset_w + (overlap_end_w - overlap_start_w)

        # Extract the overlapping region from the crop
        overlap_crop = crop_probs[
            crop_offset_d:crop_end_d, crop_offset_h:crop_end_h, crop_offset_w:crop_end_w
        ]

        # Update object map
        if update_method == "max":
            object_map[
                overlap_start_d:overlap_end_d,
                overlap_start_h:overlap_end_h,
                overlap_start_w:overlap_end_w,
            ] = torch.maximum(
                object_map[
                    overlap_start_d:overlap_end_d,
                    overlap_start_h:overlap_end_h,
                    overlap_start_w:overlap_end_w,
                ],
                overlap_crop,
            )
        elif update_method == "ema":
            alpha = 0.1  # EMA coefficient
            object_map[
                overlap_start_d:overlap_end_d,
                overlap_start_h:overlap_end_h,
                overlap_start_w:overlap_end_w,
            ] = (
                alpha * overlap_crop
                + (1 - alpha)
                * object_map[
                    overlap_start_d:overlap_end_d,
                    overlap_start_h:overlap_end_h,
                    overlap_start_w:overlap_end_w,
                ]
            )

    def flood_fill_from_seed(
        self,
        raw_volume: torch.Tensor,
        seed_position: Tuple[int, int, int],
        initial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform flood-fill from a single seed position.

        Args:
            raw_volume: Raw 3D volume tensor [D, H, W]
            seed_position: (d, h, w) coordinates of seed
            initial_mask: Optional initial object mask [D, H, W]

        Returns:
            Tuple of (object_mask, metadata_dict)
        """
        # Initialize object map
        if initial_mask is not None:
            object_map = initial_mask.clone().to(self.device)
        else:
            object_map = torch.zeros_like(raw_volume, device=self.device)

        # Initialize frontier with seed position
        frontier = []
        visited = set()
        step_count = 0

        # Add seed to frontier
        seed_d, seed_h, seed_w = seed_position
        heapq.heappush(frontier, (-1.0, step_count, (seed_d, seed_h, seed_w)))

        # Track statistics
        stats = {
            "steps": 0,
            "max_prob": 0.0,
            "mean_prob": 0.0,
            "object_size": 0,
            "frontier_size": 0,
        }

        while frontier and step_count < self.config.max_steps:
            # Get next position from frontier
            if self.config.queue_strategy == "fifo":
                # Simple FIFO for testing
                _, _, current_pos = frontier.pop(0)
            else:
                # Max probability first (default)
                _, _, current_pos = heapq.heappop(frontier)

            if current_pos in visited:
                continue

            visited.add(current_pos)
            step_count += 1

            # Extract FOV around current position
            raw_fov = self._extract_fov(raw_volume, current_pos, self.config.fov_size)
            mask_fov = self._extract_fov(object_map, current_pos, self.config.fov_size)

            # Create seed mask if at seed position
            if current_pos == seed_position:
                seed_fov = self._create_seed_mask(
                    self.config.fov_size,
                    (
                        self.config.fov_size[0] // 2,
                        self.config.fov_size[1] // 2,
                        self.config.fov_size[2] // 2,
                    ),
                )
            else:
                seed_fov = torch.zeros_like(mask_fov)

            # Prepare input tensor
            input_tensor = torch.stack([raw_fov, mask_fov, seed_fov], dim=0).unsqueeze(
                0
            )  # [1, 3, D, H, W]

            # Run model
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # [D, H, W]

            # Update object map
            self._update_object_map(
                object_map, probs, current_pos, self.config.center_crop_size
            )

            # Check if we should continue
            mean_prob = probs.mean().item()
            max_prob = probs.max().item()

            if mean_prob < self.config.stop_threshold:
                break

            # Add neighbors to frontier if movement threshold is met
            if self._should_move(probs, self.config.center_crop_size):
                neighbors = self._get_neighbor_positions(current_pos)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Calculate priority (negative for max-heap behavior)
                        if self.config.queue_strategy == "entropy":
                            # Use entropy as priority (higher entropy = higher priority)
                            neighbor_fov = self._extract_fov(
                                object_map, neighbor, self.config.fov_size
                            )
                            entropy = -(
                                neighbor_fov * torch.log(neighbor_fov + 1e-8)
                                + (1 - neighbor_fov)
                                * torch.log(1 - neighbor_fov + 1e-8)
                            ).mean()
                            priority = -entropy.item()
                        else:
                            # Use max probability
                            priority = -max_prob

                        heapq.heappush(frontier, (priority, step_count, neighbor))

            # Update statistics
            stats["steps"] = step_count
            stats["max_prob"] = max(stats["max_prob"], max_prob)
            stats["mean_prob"] = mean_prob
            stats["frontier_size"] = len(frontier)

        # Calculate final object size and mean probability
        object_binary = (object_map > self.config.accept_threshold).float()
        stats["object_size"] = object_binary.sum().item()

        # Calculate mean probability of the entire object
        if stats["object_size"] > 0:
            object_probs = object_map[object_binary > 0]
            stats["object_mean_prob"] = object_probs.mean().item()
        else:
            stats["object_mean_prob"] = 0.0

        # Accept or reject object based on criteria
        if (
            stats["object_size"] > 0
            and stats["object_mean_prob"] > self.config.accept_threshold
            and stats["object_size"] < self.config.max_voxels
        ):
            final_mask = object_binary
        else:
            final_mask = torch.zeros_like(object_map)

        return final_mask, stats

    def segment_volume(
        self,
        raw_volume: torch.Tensor,
        seed_positions: List[Tuple[int, int, int]],
        overlap_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Segment entire volume from multiple seed positions.

        Args:
            raw_volume: Raw 3D volume tensor [D, H, W]
            seed_positions: List of (d, h, w) seed coordinates
            overlap_threshold: Threshold for handling overlapping objects

        Returns:
            Labeled volume tensor [D, H, W] with unique object IDs
        """
        raw_volume = raw_volume.to(self.device)
        labeled_volume = torch.zeros_like(raw_volume, dtype=torch.int32)
        object_id = 1

        for seed_pos in seed_positions:
            # Perform flood-fill from this seed
            object_mask, stats = self.flood_fill_from_seed(raw_volume, seed_pos)

            if object_mask.sum() > 0:
                # Handle overlaps with existing objects
                overlap_mask = (labeled_volume > 0) & (object_mask > 0)

                if overlap_mask.sum() > 0:
                    # Simple overlap resolution: keep the new object if it's larger
                    existing_ids = labeled_volume[overlap_mask].unique()
                    for existing_id in existing_ids:
                        existing_mask = labeled_volume == existing_id
                        existing_size = existing_mask.sum().item()
                        new_size = object_mask.sum().item()

                        if new_size > existing_size * overlap_threshold:
                            # Replace existing object
                            labeled_volume[existing_mask] = 0
                        else:
                            # Skip this object
                            object_mask = torch.zeros_like(object_mask)
                            break

                # Assign new object ID
                if object_mask.sum() > 0:
                    labeled_volume[object_mask > 0] = object_id
                    object_id += 1

        return labeled_volume
