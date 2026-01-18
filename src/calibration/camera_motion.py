"""
Camera motion estimation and video stabilization.

Handles camera shake compensation for handheld captures.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import cv2

from ..config import Camera, CameraExtrinsics


@dataclass
class CameraTrajectory:
    """Time-varying camera poses."""
    timestamps: torch.Tensor  # (T,) timestamps
    rotations: torch.Tensor   # (T, 3, 3) rotation matrices per frame
    translations: torch.Tensor  # (T, 3) translation vectors per frame

    def __len__(self):
        return len(self.timestamps)

    def get_pose(self, frame_idx: int) -> CameraExtrinsics:
        """Get camera extrinsics at specific frame."""
        return CameraExtrinsics(
            R=self.rotations[frame_idx],
            t=self.translations[frame_idx]
        )

    def interpolate(self, t: float) -> CameraExtrinsics:
        """Interpolate camera pose at arbitrary time."""
        # Find surrounding frames
        idx = torch.searchsorted(self.timestamps, t)

        if idx == 0:
            return self.get_pose(0)
        if idx >= len(self):
            return self.get_pose(-1)

        # Linear interpolation weight
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        alpha = (t - t0) / (t1 - t0 + 1e-8)

        # Interpolate translation
        t_interp = (1 - alpha) * self.translations[idx - 1] + alpha * self.translations[idx]

        # Interpolate rotation (simple SLERP approximation)
        R0, R1 = self.rotations[idx - 1], self.rotations[idx]
        R_interp = self._interpolate_rotation(R0, R1, alpha)

        return CameraExtrinsics(R=R_interp, t=t_interp)

    def _interpolate_rotation(
        self,
        R0: torch.Tensor,
        R1: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Interpolate between two rotation matrices."""
        # Convert to axis-angle, interpolate, convert back
        # For small motions (shake), linear interpolation is reasonable
        R_diff = R0.T @ R1

        # Log map (approximate for small rotations)
        angle = torch.acos(torch.clamp((torch.trace(R_diff) - 1) / 2, -1, 1))

        if angle < 1e-6:
            return R0

        # Axis of rotation
        skew = (R_diff - R_diff.T) / (2 * torch.sin(angle) + 1e-8)
        axis = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])

        # Interpolated angle
        angle_interp = alpha * angle

        # Rodrigues formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=R0.dtype, device=R0.device)

        R_interp_from_identity = (
            torch.eye(3, dtype=R0.dtype, device=R0.device) +
            torch.sin(angle_interp) * K +
            (1 - torch.cos(angle_interp)) * (K @ K)
        )

        return R0 @ R_interp_from_identity


class CameraMotionEstimator:
    """
    Estimate per-frame camera motion from video.

    Uses feature tracking to estimate frame-to-frame homographies
    or essential matrices, then chains them to get absolute poses.
    """

    def __init__(
        self,
        method: str = "homography",  # "homography" or "essential"
        max_features: int = 1000,
        ransac_threshold: float = 3.0,
        smoothing_window: int = 5
    ):
        """
        Initialize camera motion estimator.

        Args:
            method: Motion estimation method
            max_features: Maximum features to track
            ransac_threshold: RANSAC inlier threshold
            smoothing_window: Window size for trajectory smoothing
        """
        self.method = method
        self.max_features = max_features
        self.ransac_threshold = ransac_threshold
        self.smoothing_window = smoothing_window

        # Feature detector
        self.detector = cv2.SIFT_create(nfeatures=max_features)

        # Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def estimate_trajectory(
        self,
        frames: torch.Tensor,
        base_camera: Camera,
        timestamps: Optional[torch.Tensor] = None
    ) -> CameraTrajectory:
        """
        Estimate camera trajectory from video frames.

        Args:
            frames: Video frames (T, H, W, 3)
            base_camera: Reference camera (pose at frame 0)
            timestamps: Optional timestamps for each frame

        Returns:
            CameraTrajectory with per-frame poses
        """
        n_frames = frames.shape[0]
        device = base_camera.extrinsics.R.device

        if timestamps is None:
            timestamps = torch.arange(n_frames, dtype=torch.float32, device=device)
        else:
            # MPS doesn't support float64, so convert to float32
            timestamps = timestamps.to(dtype=torch.float32, device=device)

        # Initialize with base camera pose
        rotations = [base_camera.extrinsics.R.clone()]
        translations = [base_camera.extrinsics.t.clone()]

        # Track frame-to-frame motion
        prev_frame = self._to_gray(frames[0])
        prev_kpts, prev_desc = self.detector.detectAndCompute(prev_frame, None)

        for i in range(1, n_frames):
            curr_frame = self._to_gray(frames[i])
            curr_kpts, curr_desc = self.detector.detectAndCompute(curr_frame, None)

            if prev_desc is None or curr_desc is None or len(prev_kpts) < 10 or len(curr_kpts) < 10:
                # Not enough features, assume no motion
                rotations.append(rotations[-1].clone())
                translations.append(translations[-1].clone())
            else:
                # Match features
                matches = self.matcher.knnMatch(prev_desc, curr_desc, k=2)

                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) < 8:
                    rotations.append(rotations[-1].clone())
                    translations.append(translations[-1].clone())
                else:
                    # Get matched points
                    pts1 = np.float32([prev_kpts[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([curr_kpts[m.trainIdx].pt for m in good_matches])

                    # Estimate relative motion
                    dR, dt = self._estimate_motion(
                        pts1, pts2, base_camera, prev_frame.shape
                    )

                    # Chain with previous pose
                    R_prev = rotations[-1]
                    t_prev = translations[-1]

                    R_curr = dR @ R_prev
                    t_curr = dR @ t_prev + dt

                    rotations.append(R_curr)
                    translations.append(t_curr)

            prev_frame = curr_frame
            prev_kpts, prev_desc = curr_kpts, curr_desc

        rotations = torch.stack(rotations)
        translations = torch.stack(translations)

        # Apply temporal smoothing to reduce jitter
        rotations, translations = self._smooth_trajectory(rotations, translations)

        return CameraTrajectory(
            timestamps=timestamps,
            rotations=rotations,
            translations=translations
        )

    def _to_gray(self, frame: torch.Tensor) -> np.ndarray:
        """Convert frame to grayscale uint8."""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        return gray

    def _estimate_motion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        camera: Camera,
        image_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate relative camera motion between frames."""
        # Get device from camera extrinsics
        device = camera.extrinsics.R.device

        if self.method == "homography":
            # Estimate homography (assumes planar scene or pure rotation)
            H, mask = cv2.findHomography(
                pts1, pts2,
                cv2.RANSAC,
                self.ransac_threshold
            )

            if H is None:
                return torch.eye(3, device=device), torch.zeros(3, device=device)

            # Decompose homography
            K = camera.intrinsics.to_matrix().cpu().numpy()
            num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)

            if num == 0:
                return torch.eye(3, device=device), torch.zeros(3, device=device)

            # Choose best decomposition (smallest rotation angle)
            best_idx = 0
            min_angle = float('inf')
            for i in range(num):
                angle = np.arccos(np.clip((np.trace(Rs[i]) - 1) / 2, -1, 1))
                if angle < min_angle:
                    min_angle = angle
                    best_idx = i

            R = torch.from_numpy(Rs[best_idx]).float().to(device)
            t = torch.from_numpy(ts[best_idx].flatten()).float().to(device)

            # Scale translation (homography gives unit translation)
            # For shake, translations are typically small
            t = t * 0.01  # Assume small motion

        else:  # essential matrix
            K = camera.intrinsics.to_matrix().cpu().numpy()

            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                cv2.RANSAC,
                threshold=self.ransac_threshold
            )

            if E is None:
                return torch.eye(3, device=device), torch.zeros(3, device=device)

            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

            R = torch.from_numpy(R).float().to(device)
            t = torch.from_numpy(t.flatten()).float().to(device)

            # Scale for typical camera shake
            t = t * 0.01

        return R, t

    def _smooth_trajectory(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal smoothing to trajectory."""
        if self.smoothing_window <= 1:
            return rotations, translations

        n_frames = len(rotations)
        window = self.smoothing_window
        half_w = window // 2

        # Smooth translations with moving average
        translations_smooth = translations.clone()
        for i in range(n_frames):
            start = max(0, i - half_w)
            end = min(n_frames, i + half_w + 1)
            translations_smooth[i] = translations[start:end].mean(dim=0)

        # Smooth rotations (approximate - average in tangent space)
        rotations_smooth = rotations.clone()
        for i in range(n_frames):
            start = max(0, i - half_w)
            end = min(n_frames, i + half_w + 1)

            # Average rotation matrices (orthogonalize after)
            R_avg = rotations[start:end].mean(dim=0)

            # Re-orthogonalize using SVD
            U, _, Vt = torch.linalg.svd(R_avg)
            rotations_smooth[i] = U @ Vt

        return rotations_smooth, translations_smooth


class VideoStabilizer:
    """
    Stabilize video by removing camera shake.

    Estimates camera motion and warps frames to a stable reference.
    """

    def __init__(
        self,
        smoothing_radius: int = 30,
        border_mode: str = "replicate"
    ):
        """
        Initialize video stabilizer.

        Args:
            smoothing_radius: Temporal smoothing radius
            border_mode: How to handle frame borders after warping
        """
        self.smoothing_radius = smoothing_radius
        self.border_mode = border_mode

        self.border_modes = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "constant": cv2.BORDER_CONSTANT
        }

    def stabilize(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Stabilize video frames.

        Args:
            frames: Video frames (T, H, W, 3)

        Returns:
            Tuple of (stabilized frames, original transforms)
        """
        n_frames, H, W = frames.shape[:3]

        # Track features
        transforms = []
        prev_gray = self._to_gray(frames[0])

        for i in range(1, n_frames):
            curr_gray = self._to_gray(frames[i])

            # Optical flow for dense motion
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30
            )

            if prev_pts is None:
                transforms.append(np.eye(2, 3, dtype=np.float32))
            else:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None
                )

                # Filter valid points
                valid = status.flatten() == 1
                prev_valid = prev_pts[valid]
                curr_valid = curr_pts[valid]

                if len(prev_valid) < 4:
                    transforms.append(np.eye(2, 3, dtype=np.float32))
                else:
                    # Estimate affine transform
                    T, _ = cv2.estimateAffine2D(
                        prev_valid, curr_valid,
                        method=cv2.RANSAC
                    )
                    if T is None:
                        T = np.eye(2, 3, dtype=np.float32)
                    transforms.append(T)

            prev_gray = curr_gray

        # Build cumulative trajectory
        trajectory = np.zeros((n_frames, 3))  # x, y, angle
        trajectory[0] = [0, 0, 0]

        for i, T in enumerate(transforms):
            # Extract translation and rotation from affine
            dx = T[0, 2]
            dy = T[1, 2]
            da = np.arctan2(T[1, 0], T[0, 0])

            trajectory[i + 1] = trajectory[i] + [dx, dy, da]

        # Smooth trajectory
        smoothed = self._smooth_trajectory_1d(trajectory)

        # Compute corrective transforms
        difference = smoothed - trajectory

        # Apply stabilization
        stabilized = torch.zeros_like(frames)
        stabilized[0] = frames[0]

        for i in range(1, n_frames):
            dx, dy, da = difference[i]

            # Create corrective transform
            T_correct = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy]
            ], dtype=np.float32)

            # Warp frame
            frame_np = frames[i].cpu().numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)

            warped = cv2.warpAffine(
                frame_np,
                T_correct,
                (W, H),
                borderMode=self.border_modes.get(self.border_mode, cv2.BORDER_REPLICATE)
            )

            stabilized[i] = torch.from_numpy(warped.astype(np.float32) / 255.0)

        return stabilized, transforms

    def _to_gray(self, frame: torch.Tensor) -> np.ndarray:
        """Convert to grayscale uint8."""
        frame_np = frame.cpu().numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)

        if frame_np.ndim == 3:
            return cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        return frame_np

    def _smooth_trajectory_1d(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply 1D smoothing to trajectory."""
        smoothed = np.copy(trajectory)

        for i in range(3):  # x, y, angle
            kernel = np.ones(2 * self.smoothing_radius + 1) / (2 * self.smoothing_radius + 1)
            smoothed[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')

        return smoothed


def compensate_shake_in_triangulation(
    pts2d: torch.Tensor,
    frame_idx: int,
    base_camera: Camera,
    trajectory: CameraTrajectory
) -> torch.Tensor:
    """
    Compensate for camera shake when triangulating.

    Projects 2D points back to where they would be
    if the camera was static at the base pose.

    Args:
        pts2d: 2D points observed at frame_idx (N, 2)
        frame_idx: Frame index
        base_camera: Reference camera pose
        trajectory: Estimated camera trajectory

    Returns:
        Corrected 2D points (N, 2)
    """
    # Get actual camera pose at this frame
    actual_extrinsics = trajectory.get_pose(frame_idx)

    # Create actual camera
    actual_camera = Camera(
        intrinsics=base_camera.intrinsics,
        extrinsics=actual_extrinsics,
        width=base_camera.width,
        height=base_camera.height
    )

    K = base_camera.intrinsics.to_matrix().to(pts2d.device)
    K_inv = torch.linalg.inv(K)

    # Unproject to normalized camera coordinates at actual pose
    pts_hom = torch.cat([pts2d, torch.ones(len(pts2d), 1, device=pts2d.device)], dim=1)
    pts_norm = (K_inv @ pts_hom.T).T[:, :2]

    # Transform from actual camera to base camera coordinates
    R_actual = actual_extrinsics.R.to(pts2d.device)
    R_base = base_camera.extrinsics.R.to(pts2d.device)

    # Relative rotation
    R_rel = R_base @ R_actual.T

    # Apply rotation in normalized coordinates (approximate for small angles)
    # For more accuracy, would need depth estimation
    pts_norm_corrected = (R_rel[:2, :2] @ pts_norm.T).T

    # Project back to pixels
    pts_corrected = (K[:2, :2] @ pts_norm_corrected.T).T + K[:2, 2]

    return pts_corrected
