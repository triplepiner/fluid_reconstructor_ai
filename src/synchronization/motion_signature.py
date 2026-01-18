"""
Motion signature extraction from optical flow.

Creates 1D motion signatures that capture the temporal pattern of motion
in each video, used for cross-correlation based synchronization.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MotionSignature:
    """Container for motion signature data."""
    mean_magnitude: torch.Tensor  # (T,) - mean flow magnitude per frame
    std_magnitude: torch.Tensor  # (T,) - std of flow magnitude
    dominant_direction: torch.Tensor  # (T,) - dominant flow direction in radians
    spatial_variance: torch.Tensor  # (T,) - spatial variance of flow
    timestamps: torch.Tensor  # (T,) - timestamp of each measurement
    fps: float  # Original FPS of the video

    def __len__(self):
        return len(self.mean_magnitude)

    def normalized(self) -> "MotionSignature":
        """Return a normalized version of the signature."""
        def normalize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean()
            std = x.std()
            if std < 1e-8:
                return x - mean
            return (x - mean) / std

        return MotionSignature(
            mean_magnitude=normalize(self.mean_magnitude),
            std_magnitude=normalize(self.std_magnitude),
            dominant_direction=self.dominant_direction,  # Don't normalize direction
            spatial_variance=normalize(self.spatial_variance),
            timestamps=self.timestamps,
            fps=self.fps
        )

    def combined_signature(self, weights: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
        """
        Combine multiple signature components into a single 1D signal.

        Args:
            weights: Weights for (magnitude, std, variance) components.
                    Default is (1.0, 0.3, 0.2)

        Returns:
            Combined 1D signature tensor
        """
        if weights is None:
            weights = (1.0, 0.3, 0.2)

        normalized = self.normalized()

        combined = (
            weights[0] * normalized.mean_magnitude +
            weights[1] * normalized.std_magnitude +
            weights[2] * normalized.spatial_variance
        )

        return combined


class MotionSignatureExtractor:
    """
    Extract motion signatures from optical flow sequences.
    """

    def __init__(
        self,
        use_magnitude: bool = True,
        use_direction: bool = True,
        use_variance: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize the motion signature extractor.

        Args:
            use_magnitude: Include mean magnitude in signature
            use_direction: Include dominant direction in signature
            use_variance: Include spatial variance in signature
            roi: Region of interest (x1, y1, x2, y2) to analyze, None for full frame
        """
        self.use_magnitude = use_magnitude
        self.use_direction = use_direction
        self.use_variance = use_variance
        self.roi = roi

    def extract(
        self,
        optical_flow: torch.Tensor,
        fps: float,
        timestamps: Optional[torch.Tensor] = None
    ) -> MotionSignature:
        """
        Extract motion signature from optical flow sequence.

        Args:
            optical_flow: Optical flow tensor, shape (T, H, W, 2) with (dx, dy)
            fps: Frame rate of the video
            timestamps: Optional timestamps for each flow frame

        Returns:
            MotionSignature object
        """
        T, H, W, _ = optical_flow.shape

        # Apply ROI if specified
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            flow = optical_flow[:, y1:y2, x1:x2, :]
        else:
            flow = optical_flow

        # Extract flow components
        u = flow[..., 0]  # (T, H', W')
        v = flow[..., 1]

        # Compute magnitude
        magnitude = torch.sqrt(u**2 + v**2)

        # Compute per-frame statistics
        mean_mag = magnitude.mean(dim=(1, 2))  # (T,)
        std_mag = magnitude.std(dim=(1, 2))  # (T,)

        # Compute dominant direction
        mean_u = u.mean(dim=(1, 2))
        mean_v = v.mean(dim=(1, 2))
        direction = torch.atan2(mean_v, mean_u)  # (T,)

        # Compute spatial variance
        spatial_var = magnitude.var(dim=(1, 2))  # (T,)

        # Generate timestamps if not provided (use float32 for MPS compatibility)
        if timestamps is None:
            timestamps = torch.arange(T, dtype=torch.float32) / fps

        return MotionSignature(
            mean_magnitude=mean_mag,
            std_magnitude=std_mag,
            dominant_direction=direction,
            spatial_variance=spatial_var,
            timestamps=timestamps,
            fps=fps
        )

    def extract_from_video(
        self,
        video_data,  # VideoData type
        compute_flow: bool = True
    ) -> MotionSignature:
        """
        Extract motion signature from a VideoData object.

        Args:
            video_data: VideoData object with frames and optionally optical flow
            compute_flow: If True and optical flow not present, compute it

        Returns:
            MotionSignature object
        """
        if video_data.optical_flow is None:
            if compute_flow:
                from ..preprocessing.optical_flow import OpticalFlowEstimator
                estimator = OpticalFlowEstimator()
                video_data.optical_flow = estimator.estimate_sequence(video_data.frames)
            else:
                raise ValueError("Video has no optical flow and compute_flow=False")

        return self.extract(
            video_data.optical_flow,
            video_data.metadata.fps,
            video_data.timestamps[:-1]  # One less timestamp than frames
        )


class MotionSignatureAnalyzer:
    """
    Analyze and compare motion signatures for synchronization.
    """

    @staticmethod
    def find_motion_events(
        signature: MotionSignature,
        threshold_std: float = 2.0
    ) -> List[Tuple[int, float]]:
        """
        Find significant motion events in a signature.

        Args:
            signature: MotionSignature to analyze
            threshold_std: Number of standard deviations above mean for event detection

        Returns:
            List of (frame_index, magnitude) tuples for detected events
        """
        mag = signature.mean_magnitude
        mean = mag.mean()
        std = mag.std()

        threshold = mean + threshold_std * std
        event_mask = mag > threshold

        events = []
        for i, is_event in enumerate(event_mask):
            if is_event:
                events.append((i, mag[i].item()))

        return events

    @staticmethod
    def compute_derivative(
        signature: MotionSignature
    ) -> torch.Tensor:
        """
        Compute the derivative of the motion signature.

        Useful for detecting sudden changes in motion.

        Args:
            signature: MotionSignature to differentiate

        Returns:
            Derivative tensor, shape (T-1,)
        """
        mag = signature.mean_magnitude
        dt = 1.0 / signature.fps
        derivative = (mag[1:] - mag[:-1]) / dt
        return derivative

    @staticmethod
    def smooth(
        signal: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Apply smoothing to a signal using a moving average.

        Args:
            signal: 1D signal to smooth
            window_size: Size of the smoothing window

        Returns:
            Smoothed signal
        """
        if window_size < 2:
            return signal

        kernel = torch.ones(window_size) / window_size
        kernel = kernel.to(signal.device)

        # Pad signal
        pad_size = window_size // 2
        padded = torch.nn.functional.pad(
            signal.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size),
            mode='reflect'
        )

        # Convolve
        smoothed = torch.nn.functional.conv1d(
            padded,
            kernel.unsqueeze(0).unsqueeze(0)
        )

        return smoothed.squeeze()

    @staticmethod
    def resample(
        signature: MotionSignature,
        target_fps: float
    ) -> MotionSignature:
        """
        Resample a motion signature to a different frame rate.

        Args:
            signature: MotionSignature to resample
            target_fps: Target frame rate

        Returns:
            Resampled MotionSignature
        """
        if abs(signature.fps - target_fps) < 1e-6:
            return signature

        # Compute new timestamps
        duration = len(signature) / signature.fps
        n_new_frames = int(duration * target_fps)
        new_timestamps = torch.linspace(0, duration, n_new_frames)

        # Interpolate each component
        old_times = signature.timestamps.float()
        new_times = new_timestamps.float()

        def interp1d(values: torch.Tensor) -> torch.Tensor:
            # Simple linear interpolation
            indices = torch.searchsorted(old_times, new_times)
            indices = torch.clamp(indices, 1, len(old_times) - 1)

            t0 = old_times[indices - 1]
            t1 = old_times[indices]
            v0 = values[indices - 1]
            v1 = values[indices]

            alpha = (new_times - t0) / (t1 - t0 + 1e-8)
            return v0 + alpha * (v1 - v0)

        return MotionSignature(
            mean_magnitude=interp1d(signature.mean_magnitude),
            std_magnitude=interp1d(signature.std_magnitude),
            dominant_direction=interp1d(signature.dominant_direction),
            spatial_variance=interp1d(signature.spatial_variance),
            timestamps=new_timestamps,
            fps=target_fps
        )
