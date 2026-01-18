"""
Optical flow estimation using RAFT.

Computes dense optical flow between consecutive frames for motion analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ..config import VideoData


class OpticalFlowEstimator:
    """
    Estimate optical flow using RAFT (Recurrent All-Pairs Field Transforms).

    Uses torchvision's RAFT implementation with pretrained weights.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "raft_large",
        num_iterations: int = 20
    ):
        """
        Initialize the optical flow estimator.

        Args:
            device: Device to run inference on
            model_name: RAFT model variant ('raft_small' or 'raft_large')
            num_iterations: Number of refinement iterations
        """
        self.device = device
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.model = None

    def _load_model(self):
        """Load the RAFT model lazily."""
        if self.model is not None:
            return

        try:
            from torchvision.models.optical_flow import raft_large, raft_small
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights

            if self.model_name == "raft_large":
                self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            else:
                self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)

            self.model = self.model.to(self.device)
            self.model.eval()

        except ImportError:
            raise ImportError(
                "torchvision with optical flow support is required. "
                "Install with: pip install torchvision>=0.14"
            )

    def estimate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate optical flow between two frames.

        Args:
            frame1: First frame, shape (H, W, 3) or (1, 3, H, W), values in [0, 1]
            frame2: Second frame, same shape as frame1

        Returns:
            Optical flow tensor, shape (H, W, 2) with (dx, dy) displacements
        """
        self._load_model()

        # Prepare inputs
        img1, img2 = self._prepare_inputs(frame1, frame2)

        with torch.no_grad():
            # RAFT returns list of flow predictions at different iterations
            flow_predictions = self.model(img1, img2, num_flow_updates=self.num_iterations)
            # Take the final prediction
            flow = flow_predictions[-1]

        # Convert from (1, 2, H, W) to (H, W, 2)
        flow = flow.squeeze(0).permute(1, 2, 0)

        return flow

    def estimate_bidirectional(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate bidirectional optical flow.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Tuple of (forward_flow, backward_flow)
        """
        forward_flow = self.estimate(frame1, frame2)
        backward_flow = self.estimate(frame2, frame1)
        return forward_flow, backward_flow

    def estimate_sequence(
        self,
        frames: torch.Tensor,
        bidirectional: bool = False
    ) -> torch.Tensor:
        """
        Estimate optical flow for a sequence of frames.

        Args:
            frames: Sequence of frames, shape (N, H, W, 3)
            bidirectional: If True, compute both forward and backward flow

        Returns:
            Optical flow tensor, shape (N-1, H, W, 2) for forward only,
            or (N-1, H, W, 4) for bidirectional (forward_x, forward_y, backward_x, backward_y)
        """
        self._load_model()

        n_frames = frames.shape[0]
        flows = []

        for i in tqdm(range(n_frames - 1), desc="Computing optical flow"):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            if bidirectional:
                forward_flow, backward_flow = self.estimate_bidirectional(frame1, frame2)
                flow = torch.cat([forward_flow, backward_flow], dim=-1)
            else:
                flow = self.estimate(frame1, frame2)

            flows.append(flow)

        return torch.stack(flows, dim=0)

    def compute_for_video(
        self,
        video: VideoData,
        bidirectional: bool = False,
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Compute optical flow for all consecutive frame pairs in a video.

        Args:
            video: VideoData object with frames
            bidirectional: Whether to compute bidirectional flow
            batch_size: Batch size for processing (reduces memory usage)

        Returns:
            Optical flow tensor, shape (n_frames-1, H, W, 2) or (n_frames-1, H, W, 4)
        """
        frames = video.frames

        # Process in batches to manage memory
        if batch_size is None or batch_size >= frames.shape[0]:
            return self.estimate_sequence(frames, bidirectional)

        # Batched processing
        n_frames = frames.shape[0]
        all_flows = []

        for start_idx in tqdm(range(0, n_frames - 1, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size + 1, n_frames)
            batch_frames = frames[start_idx:end_idx]
            batch_flows = self.estimate_sequence(batch_frames, bidirectional)
            all_flows.append(batch_flows)

        return torch.cat(all_flows, dim=0)

    def _prepare_inputs(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare frames for RAFT input.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Tuple of prepared tensors in RAFT input format
        """
        # Handle different input formats
        if frame1.dim() == 3:  # (H, W, 3)
            frame1 = frame1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            frame2 = frame2.permute(2, 0, 1).unsqueeze(0)

        # Ensure on correct device
        frame1 = frame1.to(self.device)
        frame2 = frame2.to(self.device)

        # RAFT expects values in [0, 255]
        if frame1.max() <= 1.0:
            frame1 = frame1 * 255
            frame2 = frame2 * 255

        return frame1, frame2


class OpticalFlowVisualization:
    """Utilities for visualizing optical flow."""

    @staticmethod
    def flow_to_color(
        flow: torch.Tensor,
        max_flow: Optional[float] = None
    ) -> torch.Tensor:
        """
        Convert optical flow to RGB visualization using HSV color wheel.

        Args:
            flow: Optical flow tensor, shape (H, W, 2)
            max_flow: Maximum flow magnitude for normalization

        Returns:
            RGB image tensor, shape (H, W, 3), values in [0, 1]
        """
        flow_np = flow.cpu().numpy() if isinstance(flow, torch.Tensor) else flow

        # Compute magnitude and angle
        u = flow_np[..., 0]
        v = flow_np[..., 1]

        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)

        # Normalize magnitude
        if max_flow is None:
            max_flow = np.max(magnitude) + 1e-8

        magnitude = np.clip(magnitude / max_flow, 0, 1)

        # Convert to HSV
        hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
        saturation = np.ones_like(hue)
        value = magnitude

        # HSV to RGB
        import colorsys
        rgb = np.zeros((*flow_np.shape[:2], 3), dtype=np.float32)
        for i in range(flow_np.shape[0]):
            for j in range(flow_np.shape[1]):
                rgb[i, j] = colorsys.hsv_to_rgb(hue[i, j], saturation[i, j], value[i, j])

        return torch.from_numpy(rgb)

    @staticmethod
    def flow_to_arrows(
        flow: torch.Tensor,
        step: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert optical flow to arrow coordinates for quiver plot.

        Args:
            flow: Optical flow tensor, shape (H, W, 2)
            step: Step size for arrow sampling

        Returns:
            Tuple of (x, y, u, v) arrays for matplotlib quiver
        """
        flow_np = flow.cpu().numpy() if isinstance(flow, torch.Tensor) else flow

        h, w = flow_np.shape[:2]
        y, x = np.mgrid[0:h:step, 0:w:step]
        u = flow_np[::step, ::step, 0]
        v = flow_np[::step, ::step, 1]

        return x, y, u, v


class FarnebackOpticalFlow:
    """
    Fallback optical flow using OpenCV's Farneback method.

    Faster but less accurate than RAFT. Useful for quick prototyping.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2
    ):
        """Initialize Farneback optical flow estimator."""
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def estimate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate optical flow using Farneback method.

        Args:
            frame1: First frame, shape (H, W, 3), values in [0, 1]
            frame2: Second frame

        Returns:
            Optical flow tensor, shape (H, W, 2)
        """
        import cv2

        # Convert to numpy and grayscale
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
            frame2 = frame2.cpu().numpy()

        if frame1.max() <= 1.0:
            frame1 = (frame1 * 255).astype(np.uint8)
            frame2 = (frame2 * 255).astype(np.uint8)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0
        )

        return torch.from_numpy(flow)

    def estimate_sequence(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """Estimate optical flow for a sequence of frames."""
        n_frames = frames.shape[0]
        flows = []

        for i in tqdm(range(n_frames - 1), desc="Computing optical flow (Farneback)"):
            flow = self.estimate(frames[i], frames[i + 1])
            flows.append(flow)

        return torch.stack(flows, dim=0)
