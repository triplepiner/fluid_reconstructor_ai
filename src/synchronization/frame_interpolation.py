"""
Frame interpolation for temporal alignment.

Interpolates frames to align videos to a common timeline using
optical flow-based warping for high-quality intermediate frames.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..config import VideoData, AlignedFrames, SyncParameters


class FrameInterpolator:
    """
    Interpolate video frames to align with a common timeline.
    """

    def __init__(
        self,
        method: str = "flow",
        device: str = "cuda"
    ):
        """
        Initialize the frame interpolator.

        Args:
            method: Interpolation method ('linear', 'flow', 'slerp')
            device: Device for computations
        """
        self.method = method
        self.device = device

    def interpolate_videos(
        self,
        videos: List[VideoData],
        sync_params: SyncParameters
    ) -> AlignedFrames:
        """
        Interpolate all videos to a common timeline.

        Args:
            videos: List of VideoData objects
            sync_params: Synchronization parameters with offsets

        Returns:
            AlignedFrames object with synchronized frames
        """
        n_videos = len(videos)
        n_frames = sync_params.n_common_frames

        # Determine output dimensions (use max resolution)
        max_h = max(v.frames.shape[1] for v in videos)
        max_w = max(v.frames.shape[2] for v in videos)

        aligned_frames = torch.zeros(
            n_videos, n_frames, max_h, max_w, 3,
            dtype=torch.float32,
            device=self.device
        )

        # Use float32 for MPS compatibility (MPS doesn't support float64)
        common_timestamps = torch.linspace(
            sync_params.common_start,
            sync_params.common_end,
            n_frames,
            dtype=torch.float32
        )

        for v_idx, video in enumerate(tqdm(videos, desc="Aligning videos")):
            offset = sync_params.offsets[v_idx]
            fps = video.metadata.fps
            frames = video.frames.to(self.device)
            flow = video.optical_flow

            if flow is not None:
                flow = flow.to(self.device)

            # Resize frames if needed
            if frames.shape[1] != max_h or frames.shape[2] != max_w:
                frames = self._resize_batch(frames, max_h, max_w)
                if flow is not None:
                    flow = self._resize_flow(flow, max_h, max_w)

            for t_idx, common_time in enumerate(tqdm(
                common_timestamps, desc=f"Video {v_idx+1}", leave=False
            )):
                # Time in original video
                original_time = common_time.item() + offset

                # Frame indices
                frame_float = original_time * fps
                frame_low = int(frame_float)
                frame_high = min(frame_low + 1, len(frames) - 1)
                alpha = frame_float - frame_low

                # Clamp indices
                frame_low = max(0, min(frame_low, len(frames) - 1))
                frame_high = max(0, min(frame_high, len(frames) - 1))

                if frame_low == frame_high or alpha < 1e-6:
                    # No interpolation needed
                    aligned_frames[v_idx, t_idx] = frames[frame_low]
                elif self.method == "linear":
                    # Simple linear interpolation
                    aligned_frames[v_idx, t_idx] = (
                        (1 - alpha) * frames[frame_low] +
                        alpha * frames[frame_high]
                    )
                elif self.method == "flow" and flow is not None:
                    # Optical flow-based interpolation
                    aligned_frames[v_idx, t_idx] = self._flow_interpolate(
                        frames[frame_low],
                        frames[frame_high],
                        flow[min(frame_low, len(flow) - 1)],
                        alpha
                    )
                else:
                    # Fallback to linear
                    aligned_frames[v_idx, t_idx] = (
                        (1 - alpha) * frames[frame_low] +
                        alpha * frames[frame_high]
                    )

        return AlignedFrames(
            frames=aligned_frames,
            timestamps=common_timestamps
        )

    def interpolate_single_frame(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        alpha: float,
        flow_forward: Optional[torch.Tensor] = None,
        flow_backward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Interpolate a single frame between two frames.

        Args:
            frame1: First frame, shape (H, W, 3)
            frame2: Second frame, shape (H, W, 3)
            alpha: Interpolation factor (0=frame1, 1=frame2)
            flow_forward: Forward optical flow from frame1 to frame2
            flow_backward: Backward optical flow from frame2 to frame1

        Returns:
            Interpolated frame, shape (H, W, 3)
        """
        if self.method == "linear" or (flow_forward is None and flow_backward is None):
            return (1 - alpha) * frame1 + alpha * frame2

        if flow_forward is not None and flow_backward is None:
            return self._flow_interpolate(frame1, frame2, flow_forward, alpha)

        if flow_forward is not None and flow_backward is not None:
            return self._bidirectional_flow_interpolate(
                frame1, frame2, flow_forward, flow_backward, alpha
            )

        return (1 - alpha) * frame1 + alpha * frame2

    def _flow_interpolate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate using forward optical flow only.

        Args:
            frame1: First frame (H, W, 3)
            frame2: Second frame (H, W, 3)
            flow: Forward flow from frame1 to frame2 (H, W, 2)
            alpha: Interpolation factor

        Returns:
            Interpolated frame
        """
        H, W = frame1.shape[:2]

        # Warp frame1 forward by alpha * flow
        warped1 = self._warp_frame(frame1, flow * alpha)

        # Warp frame2 backward by (1-alpha) * (-flow)
        warped2 = self._warp_frame(frame2, -flow * (1 - alpha))

        # Blend
        interpolated = (1 - alpha) * warped1 + alpha * warped2

        return interpolated

    def _bidirectional_flow_interpolate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate using bidirectional optical flow.

        This produces higher quality results by using both forward and
        backward flow to warp both frames to the intermediate time.

        Args:
            frame1: First frame (H, W, 3)
            frame2: Second frame (H, W, 3)
            flow_forward: Flow from frame1 to frame2 (H, W, 2)
            flow_backward: Flow from frame2 to frame1 (H, W, 2)
            alpha: Interpolation factor

        Returns:
            Interpolated frame
        """
        # Intermediate flow at time alpha
        flow_t0 = -alpha * (1 - alpha) * flow_backward + alpha * alpha * flow_forward
        flow_t1 = (1 - alpha) * (1 - alpha) * flow_forward - alpha * (1 - alpha) * flow_backward

        # Warp frames to intermediate time
        warped1 = self._warp_frame(frame1, flow_t0)
        warped2 = self._warp_frame(frame2, flow_t1)

        # Compute occlusion masks (simplified)
        # Areas where backward warp differs significantly from forward
        diff1 = torch.abs(warped1 - frame1).mean(dim=-1, keepdim=True)
        diff2 = torch.abs(warped2 - frame2).mean(dim=-1, keepdim=True)

        # Soft blending weights
        weight1 = torch.exp(-diff1 * 10)
        weight2 = torch.exp(-diff2 * 10)

        # Normalize weights
        total_weight = weight1 * (1 - alpha) + weight2 * alpha + 1e-8
        weight1 = weight1 * (1 - alpha) / total_weight
        weight2 = weight2 * alpha / total_weight

        interpolated = weight1 * warped1 + weight2 * warped2

        return interpolated

    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp a frame using optical flow.

        Args:
            frame: Input frame (H, W, 3)
            flow: Optical flow (H, W, 2) with (dx, dy)

        Returns:
            Warped frame (H, W, 3)
        """
        H, W = frame.shape[:2]

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=torch.float32),
            torch.arange(W, device=frame.device, dtype=torch.float32),
            indexing='ij'
        )

        # Add flow to get new coordinates
        new_x = x + flow[..., 0]
        new_y = y + flow[..., 1]

        # Normalize to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0

        # Stack and reshape for grid_sample
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)

        # Warp
        frame_batched = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        warped = F.grid_sample(
            frame_batched, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped.squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    def _resize_batch(
        self,
        frames: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """Resize a batch of frames."""
        # frames: (N, H, W, 3) -> (N, 3, H, W) for interpolation
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(target_h, target_w), mode='bilinear', align_corners=True)
        return frames.permute(0, 2, 3, 1)  # Back to (N, H, W, 3)

    def _resize_flow(
        self,
        flow: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """Resize optical flow and adjust magnitudes."""
        orig_h, orig_w = flow.shape[1], flow.shape[2]

        # flow: (N, H, W, 2) -> (N, 2, H, W) for interpolation
        flow = flow.permute(0, 3, 1, 2)
        flow = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
        flow = flow.permute(0, 2, 3, 1)  # Back to (N, H, W, 2)

        # Scale flow magnitudes
        flow[..., 0] *= target_w / orig_w
        flow[..., 1] *= target_h / orig_h

        return flow


class SLERPInterpolator:
    """
    Spherical linear interpolation for smooth frame blending.

    Useful for scenes where linear interpolation produces ghosting.
    """

    @staticmethod
    def interpolate(
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        alpha: float,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Interpolate frames using SLERP.

        Args:
            frame1: First frame
            frame2: Second frame
            alpha: Interpolation factor
            epsilon: Small value to avoid division by zero

        Returns:
            Interpolated frame
        """
        # Flatten frames for dot product
        v1 = frame1.flatten()
        v2 = frame2.flatten()

        # Normalize
        v1_norm = v1 / (v1.norm() + epsilon)
        v2_norm = v2 / (v2.norm() + epsilon)

        # Compute angle
        dot = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta.abs() < epsilon:
            # Vectors nearly parallel, use linear interpolation
            result = (1 - alpha) * v1 + alpha * v2
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - alpha) * theta) / sin_theta
            w2 = torch.sin(alpha * theta) / sin_theta
            result = w1 * v1 + w2 * v2

        return result.reshape(frame1.shape)
