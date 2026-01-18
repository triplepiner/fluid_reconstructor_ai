"""
Temporal alignment for multi-view video synchronization.

Uses cross-correlation of motion signatures to find time offsets
between videos recorded from different viewpoints.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .motion_signature import MotionSignature, MotionSignatureAnalyzer
from ..config import SyncParameters


@dataclass
class AlignmentResult:
    """Result of pairwise alignment."""
    offset_seconds: float  # Offset in seconds (positive = video2 starts later)
    correlation: float  # Peak correlation value
    confidence: float  # Confidence score (0-1)
    offset_frames: int  # Offset in frames


class TemporalAligner:
    """
    Align multiple videos temporally using motion signature cross-correlation.
    """

    def __init__(
        self,
        min_offset: float = 0.0,
        max_offset: float = 3.0,
        sub_frame_refinement: bool = True,
        min_correlation: float = 0.3
    ):
        """
        Initialize the temporal aligner.

        Args:
            min_offset: Minimum expected offset in seconds
            max_offset: Maximum expected offset in seconds
            sub_frame_refinement: Whether to refine offset to sub-frame precision
            min_correlation: Minimum correlation for valid alignment
        """
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.sub_frame_refinement = sub_frame_refinement
        self.min_correlation = min_correlation

    def align_pair(
        self,
        sig1: MotionSignature,
        sig2: MotionSignature
    ) -> AlignmentResult:
        """
        Find the time offset between two motion signatures.

        Args:
            sig1: Motion signature from first video (reference)
            sig2: Motion signature from second video

        Returns:
            AlignmentResult with offset and confidence
        """
        # Resample to common FPS if needed
        common_fps = max(sig1.fps, sig2.fps)
        if abs(sig1.fps - common_fps) > 0.1:
            sig1 = MotionSignatureAnalyzer.resample(sig1, common_fps)
        if abs(sig2.fps - common_fps) > 0.1:
            sig2 = MotionSignatureAnalyzer.resample(sig2, common_fps)

        # Get normalized combined signatures
        signal1 = sig1.combined_signature()
        signal2 = sig2.combined_signature()

        # Normalize
        signal1 = self._normalize(signal1)
        signal2 = self._normalize(signal2)

        # Compute cross-correlation
        correlation = self._cross_correlation(signal1, signal2)

        # Find peak within valid offset range
        max_lag_frames = int(self.max_offset * common_fps)
        min_lag_frames = int(self.min_offset * common_fps)

        center = len(correlation) // 2
        valid_range = slice(
            center - max_lag_frames,
            center + max_lag_frames + 1
        )

        valid_corr = correlation[valid_range]
        peak_idx_local = torch.argmax(valid_corr)
        peak_idx = peak_idx_local + (center - max_lag_frames)

        # Compute offset
        offset_frames = peak_idx.item() - center
        offset_seconds = offset_frames / common_fps

        # Get correlation value
        peak_corr = correlation[peak_idx].item()

        # Sub-frame refinement using parabolic interpolation
        if self.sub_frame_refinement and 0 < peak_idx < len(correlation) - 1:
            offset_seconds = self._parabolic_refinement(
                correlation, peak_idx, common_fps
            )
            offset_frames = int(offset_seconds * common_fps)

        # Compute confidence based on peak sharpness
        confidence = self._compute_confidence(correlation, peak_idx, peak_corr)

        return AlignmentResult(
            offset_seconds=offset_seconds,
            correlation=peak_corr,
            confidence=confidence,
            offset_frames=offset_frames
        )

    def align_multiple(
        self,
        signatures: List[MotionSignature]
    ) -> Tuple[SyncParameters, List[AlignmentResult]]:
        """
        Align multiple videos and compute a common timeline.

        Args:
            signatures: List of motion signatures from each video

        Returns:
            Tuple of (SyncParameters, list of pairwise alignment results)
        """
        n_videos = len(signatures)
        if n_videos < 2:
            raise ValueError("Need at least 2 videos to align")

        # Compute all pairwise offsets relative to first video
        pairwise_results = []
        offsets_from_ref = [0.0]  # First video is reference

        for i in range(1, n_videos):
            result = self.align_pair(signatures[0], signatures[i])
            pairwise_results.append(result)
            offsets_from_ref.append(result.offset_seconds)

        # Verify consistency if more than 2 videos
        if n_videos > 2:
            offsets_from_ref = self._verify_and_adjust_offsets(
                signatures, offsets_from_ref
            )

        # Normalize offsets so minimum is 0
        min_offset = min(offsets_from_ref)
        normalized_offsets = [o - min_offset for o in offsets_from_ref]

        # Compute common timeline parameters
        common_fps = max(sig.fps for sig in signatures)
        durations = [len(sig) / sig.fps for sig in signatures]
        effective_durations = [
            d - offset for d, offset in zip(durations, normalized_offsets)
        ]
        common_end = min(effective_durations)
        n_common_frames = int(common_end * common_fps)

        sync_params = SyncParameters(
            offsets=normalized_offsets,
            common_fps=common_fps,
            common_start=0.0,
            common_end=common_end,
            n_common_frames=n_common_frames
        )

        return sync_params, pairwise_results

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize a signal to zero mean and unit variance."""
        mean = signal.mean()
        std = signal.std()
        if std < 1e-8:
            return signal - mean
        return (signal - mean) / std

    def _cross_correlation(
        self,
        signal1: torch.Tensor,
        signal2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized cross-correlation between two signals.

        Args:
            signal1: First signal (reference)
            signal2: Second signal

        Returns:
            Cross-correlation tensor
        """
        n1 = len(signal1)
        n2 = len(signal2)

        # Pad for full cross-correlation
        padded1 = torch.nn.functional.pad(signal1, (n2 - 1, n2 - 1))

        # Use convolution for cross-correlation
        # Flip signal2 for correlation (not convolution)
        signal2_flipped = signal2.flip(0)

        correlation = torch.nn.functional.conv1d(
            padded1.unsqueeze(0).unsqueeze(0),
            signal2_flipped.unsqueeze(0).unsqueeze(0)
        ).squeeze()

        # Normalize by length
        correlation = correlation / max(n1, n2)

        return correlation

    def _parabolic_refinement(
        self,
        correlation: torch.Tensor,
        peak_idx: int,
        fps: float
    ) -> float:
        """
        Refine peak location using parabolic interpolation.

        Args:
            correlation: Cross-correlation tensor
            peak_idx: Index of the peak
            fps: Frame rate for converting to seconds

        Returns:
            Refined offset in seconds
        """
        if peak_idx <= 0 or peak_idx >= len(correlation) - 1:
            return (peak_idx - len(correlation) // 2) / fps

        y_prev = correlation[peak_idx - 1].item()
        y_peak = correlation[peak_idx].item()
        y_next = correlation[peak_idx + 1].item()

        # Parabolic interpolation
        denominator = y_prev - 2 * y_peak + y_next
        if abs(denominator) < 1e-8:
            delta = 0
        else:
            delta = 0.5 * (y_prev - y_next) / denominator

        refined_idx = peak_idx + delta
        center = len(correlation) // 2

        return (refined_idx - center) / fps

    def _compute_confidence(
        self,
        correlation: torch.Tensor,
        peak_idx: int,
        peak_value: float
    ) -> float:
        """
        Compute confidence score based on peak prominence.

        Args:
            correlation: Cross-correlation tensor
            peak_idx: Index of the peak
            peak_value: Value at the peak

        Returns:
            Confidence score between 0 and 1
        """
        # Confidence based on:
        # 1. Absolute correlation value
        # 2. Peak prominence (how much higher than surroundings)

        if peak_value < self.min_correlation:
            return 0.0

        # Compute mean correlation excluding peak region
        n = len(correlation)
        peak_region = 10  # frames around peak to exclude
        mask = torch.ones(n, dtype=torch.bool)
        start = max(0, peak_idx - peak_region)
        end = min(n, peak_idx + peak_region + 1)
        mask[start:end] = False

        if mask.sum() > 0:
            mean_other = correlation[mask].mean().item()
        else:
            mean_other = 0

        # Prominence
        prominence = peak_value - mean_other

        # Combine into confidence score
        confidence = min(1.0, prominence * peak_value)

        return max(0.0, confidence)

    def _verify_and_adjust_offsets(
        self,
        signatures: List[MotionSignature],
        offsets: List[float]
    ) -> List[float]:
        """
        Verify offset consistency and adjust if needed.

        For n videos, we have n-1 independent offsets but n(n-1)/2 pairwise
        relationships. Use least squares to find optimal offsets.

        Args:
            signatures: List of motion signatures
            offsets: Initial offsets from pairwise alignment

        Returns:
            Adjusted offsets
        """
        n = len(signatures)
        if n <= 2:
            return offsets

        # Compute all pairwise offsets
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                result = self.align_pair(signatures[i], signatures[j])
                all_pairs.append((i, j, result.offset_seconds, result.confidence))

        # Weighted least squares to find optimal offsets
        # Constraint: offset[0] = 0 (reference)

        # Build system: For each pair (i, j, delta), we have offset[j] - offset[i] = delta
        # With weights based on confidence

        A = []
        b = []
        weights = []

        for i, j, delta, conf in all_pairs:
            row = [0.0] * n
            row[i] = -1.0
            row[j] = 1.0
            A.append(row)
            b.append(delta)
            weights.append(conf)

        # Add constraint: offset[0] = 0
        constraint = [0.0] * n
        constraint[0] = 1.0
        A.append(constraint)
        b.append(0.0)
        weights.append(10.0)  # High weight for constraint

        A = torch.tensor(A)
        b = torch.tensor(b)
        W = torch.diag(torch.tensor(weights))

        # Weighted least squares: (A^T W A)^{-1} A^T W b
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b

        try:
            offsets_opt = torch.linalg.solve(AtWA, AtWb)
            return offsets_opt.tolist()
        except:
            # Fall back to original offsets if solve fails
            return offsets


def compute_frame_mapping(
    sync_params: SyncParameters,
    video_fps: List[float],
    video_n_frames: List[int]
) -> List[List[Tuple[int, int, float]]]:
    """
    Compute mapping from common timeline frames to original video frames.

    Args:
        sync_params: Synchronization parameters
        video_fps: FPS of each original video
        video_n_frames: Number of frames in each original video

    Returns:
        List of frame mappings per video. Each mapping is a list of
        (common_frame_idx, original_frame_idx, interpolation_weight) tuples
    """
    n_videos = len(video_fps)
    mappings = []

    for v in range(n_videos):
        video_mapping = []
        offset = sync_params.offsets[v]

        for common_idx in range(sync_params.n_common_frames):
            # Time in common timeline
            common_time = common_idx / sync_params.common_fps

            # Time in original video
            original_time = common_time + offset

            # Frame index (potentially fractional)
            frame_float = original_time * video_fps[v]

            # Clamp to valid range
            frame_float = max(0, min(frame_float, video_n_frames[v] - 1))

            # Integer frame and interpolation weight
            frame_low = int(frame_float)
            frame_high = min(frame_low + 1, video_n_frames[v] - 1)
            alpha = frame_float - frame_low

            video_mapping.append((common_idx, frame_low, frame_high, alpha))

        mappings.append(video_mapping)

    return mappings
