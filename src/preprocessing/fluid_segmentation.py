"""
Fluid segmentation module.

Isolates fluid/liquid regions from video using motion and appearance cues.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2


class FluidSegmenter:
    """
    Segments fluid regions from video frames using motion analysis.

    Uses optical flow magnitude + temporal consistency to identify
    moving fluid regions and separate them from static background.
    """

    def __init__(
        self,
        motion_threshold: float = 2.0,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.8,
        temporal_window: int = 5,
        morphology_kernel_size: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize the fluid segmenter.

        Args:
            motion_threshold: Minimum optical flow magnitude to consider as motion
            min_area_ratio: Minimum area ratio for valid fluid regions
            max_area_ratio: Maximum area ratio (to filter out full-frame motion like camera shake)
            temporal_window: Number of frames for temporal consistency
            morphology_kernel_size: Kernel size for morphological operations
            device: Device for computations
        """
        self.motion_threshold = motion_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.temporal_window = temporal_window
        self.morphology_kernel_size = morphology_kernel_size
        self.device = device

    def segment_from_flow(
        self,
        optical_flows: List[torch.Tensor],
        frames: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Segment fluid regions using optical flow.

        Args:
            optical_flows: List of optical flow tensors (H, W, 2)
            frames: Optional list of frame tensors for appearance-based refinement

        Returns:
            List of binary mask tensors (H, W) where 1 = fluid region
        """
        masks = []

        # Compute flow magnitudes
        flow_magnitudes = []
        for flow in optical_flows:
            if isinstance(flow, torch.Tensor):
                flow_np = flow.cpu().numpy()
            else:
                flow_np = flow
            magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
            flow_magnitudes.append(magnitude)

        # Compute temporal statistics for adaptive thresholding
        all_magnitudes = np.stack(flow_magnitudes, axis=0)
        global_mean = np.mean(all_magnitudes)
        global_std = np.std(all_magnitudes)

        # Adaptive threshold based on flow statistics
        adaptive_threshold = max(
            self.motion_threshold,
            global_mean + 0.5 * global_std
        )

        print(f"  Flow stats: mean={global_mean:.2f}, std={global_std:.2f}")
        print(f"  Using motion threshold: {adaptive_threshold:.2f}")

        # Process each frame
        for i, magnitude in enumerate(flow_magnitudes):
            # Initial motion mask
            motion_mask = (magnitude > adaptive_threshold).astype(np.uint8)

            # Apply temporal consistency
            start_idx = max(0, i - self.temporal_window // 2)
            end_idx = min(len(flow_magnitudes), i + self.temporal_window // 2 + 1)

            temporal_masks = []
            for j in range(start_idx, end_idx):
                temp_mask = (flow_magnitudes[j] > adaptive_threshold * 0.7).astype(np.uint8)
                temporal_masks.append(temp_mask)

            # Require motion in majority of temporal window
            temporal_stack = np.stack(temporal_masks, axis=0)
            temporal_consistent = (np.mean(temporal_stack, axis=0) > 0.3).astype(np.uint8)

            # Combine with current frame mask
            combined_mask = motion_mask & temporal_consistent

            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morphology_kernel_size, self.morphology_kernel_size)
            )

            # Close small holes
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Open to remove noise
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Dilate slightly to include edges
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            # Filter by area
            h, w = combined_mask.shape
            total_pixels = h * w
            mask_pixels = np.sum(combined_mask)
            area_ratio = mask_pixels / total_pixels

            if area_ratio < self.min_area_ratio:
                # Too little motion - might be static frame, use larger threshold
                combined_mask = (magnitude > adaptive_threshold * 0.5).astype(np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            elif area_ratio > self.max_area_ratio:
                # Too much motion (camera shake?) - use stricter threshold
                combined_mask = (magnitude > adaptive_threshold * 2.0).astype(np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Convert to tensor
            mask_tensor = torch.from_numpy(combined_mask.astype(np.float32))
            masks.append(mask_tensor)

        # Add mask for last frame (same as second-to-last)
        if len(masks) > 0:
            masks.append(masks[-1].clone())

        return masks

    def refine_with_appearance(
        self,
        masks: List[torch.Tensor],
        frames: List[torch.Tensor],
        saturation_threshold: float = 0.15,
        brightness_range: Tuple[float, float] = (0.1, 0.9)
    ) -> List[torch.Tensor]:
        """
        Refine masks using color/appearance cues.

        Fluids often have specific appearance characteristics:
        - Water tends to have low saturation
        - Reflective surfaces have high brightness variation

        Args:
            masks: Initial motion-based masks
            frames: Video frames (H, W, 3) in [0, 1] range
            saturation_threshold: Maximum saturation for water-like appearance
            brightness_range: Valid brightness range for fluid

        Returns:
            Refined masks
        """
        refined_masks = []

        for mask, frame in zip(masks, frames):
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = frame

            # Convert to HSV
            if frame_np.max() <= 1.0:
                frame_uint8 = (frame_np * 255).astype(np.uint8)
            else:
                frame_uint8 = frame_np.astype(np.uint8)

            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)

            # Saturation and value channels (normalized)
            saturation = hsv[..., 1] / 255.0
            value = hsv[..., 2] / 255.0

            # Appearance mask for water-like regions
            # Low saturation OR high value variation suggests water/liquid
            appearance_mask = (
                (saturation < saturation_threshold) |
                (value > brightness_range[0]) & (value < brightness_range[1])
            ).astype(np.float32)

            # Combine with motion mask (intersection)
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            refined = mask_np * 0.7 + appearance_mask * mask_np * 0.3
            refined = (refined > 0.5).astype(np.float32)

            refined_masks.append(torch.from_numpy(refined))

        return refined_masks

    def get_fluid_bounding_box(
        self,
        masks: List[torch.Tensor],
        padding: int = 20
    ) -> Tuple[int, int, int, int]:
        """
        Get overall bounding box containing fluid across all frames.

        Args:
            masks: List of fluid masks
            padding: Padding around the bounding box

        Returns:
            (x_min, y_min, x_max, y_max) bounding box
        """
        if not masks:
            return (0, 0, 100, 100)

        h, w = masks[0].shape

        # Stack all masks and find union
        all_masks = torch.stack(masks, dim=0)
        union_mask = (all_masks.sum(dim=0) > 0)

        # Find bounding box
        rows = torch.any(union_mask, dim=1)
        cols = torch.any(union_mask, dim=0)

        if not rows.any() or not cols.any():
            return (0, 0, w, h)

        y_min = torch.where(rows)[0].min().item()
        y_max = torch.where(rows)[0].max().item()
        x_min = torch.where(cols)[0].min().item()
        x_max = torch.where(cols)[0].max().item()

        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        return (x_min, y_min, x_max, y_max)


def create_fluid_masks_from_pipeline_data(
    data: Dict,
    motion_threshold: float = 2.0,
    device: str = "cpu"
) -> Dict:
    """
    Create fluid masks from pipeline data containing optical flows.

    Args:
        data: Pipeline data dict with 'optical_flows' and 'frames'
        motion_threshold: Motion threshold for segmentation
        device: Device for computations

    Returns:
        Updated data dict with 'fluid_masks' added
    """
    segmenter = FluidSegmenter(
        motion_threshold=motion_threshold,
        device=device
    )

    all_masks = {}

    # Process each view
    for view_idx in range(len(data.get('frames', []))):
        view_key = f'view_{view_idx}'

        if 'optical_flows' in data and view_idx < len(data['optical_flows']):
            flows = data['optical_flows'][view_idx]
            frames = data['frames'][view_idx] if 'frames' in data else None

            print(f"  Segmenting fluid in view {view_idx}...")
            masks = segmenter.segment_from_flow(flows, frames)

            # Optionally refine with appearance
            if frames is not None:
                masks = segmenter.refine_with_appearance(masks, frames)

            all_masks[view_key] = masks

            # Compute coverage statistics
            if masks:
                coverage = torch.stack(masks).mean().item() * 100
                print(f"    Average fluid coverage: {coverage:.1f}%")

    data['fluid_masks'] = all_masks
    return data


def apply_mask_to_features(
    features: Dict,
    masks: List[torch.Tensor],
    frame_idx: int
) -> Dict:
    """
    Filter features to only include those within fluid mask.

    Args:
        features: Dict with 'keypoints' (N, 2), 'descriptors' (N, D)
        masks: List of fluid masks
        frame_idx: Frame index

    Returns:
        Filtered features dict
    """
    if frame_idx >= len(masks):
        return features

    mask = masks[frame_idx]
    keypoints = features['keypoints']

    # Check which keypoints fall within the fluid mask
    if isinstance(keypoints, torch.Tensor):
        kp_np = keypoints.cpu().numpy()
    else:
        kp_np = keypoints

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    valid_indices = []
    for i, (x, y) in enumerate(kp_np):
        xi, yi = int(x), int(y)
        if 0 <= yi < mask_np.shape[0] and 0 <= xi < mask_np.shape[1]:
            if mask_np[yi, xi] > 0.5:
                valid_indices.append(i)

    if not valid_indices:
        return features

    valid_indices = np.array(valid_indices)

    filtered = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor) and len(value.shape) > 0:
            if value.shape[0] == len(kp_np):
                filtered[key] = value[valid_indices]
            else:
                filtered[key] = value
        elif isinstance(value, np.ndarray) and len(value.shape) > 0:
            if value.shape[0] == len(kp_np):
                filtered[key] = value[valid_indices]
            else:
                filtered[key] = value
        else:
            filtered[key] = value

    return filtered
