"""
Camera parameter estimation from feature matches.

Estimates camera intrinsics and extrinsics from matched features
across multiple views.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import cv2

from ..config import CameraIntrinsics, CameraExtrinsics, Camera
from .feature_matching import Matches


@dataclass
class PoseEstimate:
    """Estimated camera pose with confidence."""
    R: torch.Tensor  # (3, 3) rotation matrix
    t: torch.Tensor  # (3,) translation vector
    inliers: int  # Number of inliers
    error: float  # Mean reprojection error


class CameraEstimator:
    """
    Estimate camera intrinsics and extrinsics from feature correspondences.
    """

    def __init__(
        self,
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.999,
        refine_estimate: bool = True
    ):
        """
        Initialize the camera estimator.

        Args:
            ransac_threshold: RANSAC inlier threshold in pixels
            ransac_confidence: RANSAC confidence level
            refine_estimate: Whether to refine estimates with non-linear optimization
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        self.refine_estimate = refine_estimate

    def estimate_intrinsics_from_video(
        self,
        width: int,
        height: int,
        fov_degrees: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        focal_length_mm: Optional[float] = None
    ) -> CameraIntrinsics:
        """
        Estimate camera intrinsics from video metadata.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Optional horizontal field of view in degrees
            sensor_width_mm: Optional sensor width in mm
            focal_length_mm: Optional focal length in mm

        Returns:
            Estimated CameraIntrinsics
        """
        # Principal point at image center
        cx = width / 2
        cy = height / 2

        # Estimate focal length
        if fov_degrees is not None:
            # f = (w/2) / tan(fov/2)
            fov_rad = np.radians(fov_degrees)
            fx = (width / 2) / np.tan(fov_rad / 2)
            fy = fx  # Assume square pixels
        elif sensor_width_mm is not None and focal_length_mm is not None:
            # f_pixels = f_mm * (width_pixels / sensor_width_mm)
            fx = focal_length_mm * (width / sensor_width_mm)
            fy = fx
        else:
            # Default: assume ~60 degree FOV (common for phones)
            default_fov = 60.0
            fx = (width / 2) / np.tan(np.radians(default_fov / 2))
            fy = fx

        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def estimate_intrinsics_from_motion(
        self,
        matches_list: List[Matches],
        image_size: Tuple[int, int]
    ) -> CameraIntrinsics:
        """
        Estimate camera intrinsics from motion (self-calibration).

        This is a simplified implementation that works best when there's
        significant camera rotation in the video.

        Args:
            matches_list: List of matches between consecutive frames
            image_size: (width, height) of images

        Returns:
            Estimated CameraIntrinsics
        """
        width, height = image_size

        # Collect fundamental matrices
        focal_estimates = []

        for matches in matches_list:
            if len(matches) < 15:
                continue

            pts1 = matches.matched_keypoints1.numpy()
            pts2 = matches.matched_keypoints2.numpy()

            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                cv2.FM_RANSAC,
                self.ransac_threshold,
                self.ransac_confidence
            )

            if F is None or F.shape != (3, 3):
                continue

            # Estimate focal length from fundamental matrix
            # Using the constraint that E = K^T F K should be an essential matrix
            # (two equal singular values, one zero)
            f_est = self._estimate_focal_from_fundamental(F, width, height)
            if f_est is not None and 0.3 * width < f_est < 3 * width:
                focal_estimates.append(f_est)

        if len(focal_estimates) > 0:
            fx = np.median(focal_estimates)
        else:
            # Fallback to default
            fx = width * 0.8  # Roughly 50 degree FOV

        return CameraIntrinsics(
            fx=fx, fy=fx,
            cx=width / 2, cy=height / 2
        )

    def _estimate_focal_from_fundamental(
        self,
        F: np.ndarray,
        width: int,
        height: int
    ) -> Optional[float]:
        """
        Estimate focal length from fundamental matrix.

        Uses the constraint that for a pure essential matrix,
        the two non-zero singular values should be equal.
        """
        cx, cy = width / 2, height / 2

        # Try different focal lengths and find one that gives equal singular values
        best_f = None
        best_ratio = float('inf')

        for f in np.linspace(0.3 * width, 3 * width, 100):
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])

            E = K.T @ F @ K
            U, S, Vt = np.linalg.svd(E)

            if S[2] < 1e-10:  # Third singular value should be ~0
                ratio = S[0] / (S[1] + 1e-10)
                if abs(ratio - 1) < best_ratio:
                    best_ratio = abs(ratio - 1)
                    best_f = f

        if best_ratio < 0.3:  # Accept if singular values are reasonably equal
            return best_f
        return None

    def estimate_relative_pose(
        self,
        matches: Matches,
        K1: torch.Tensor,
        K2: Optional[torch.Tensor] = None
    ) -> Tuple[PoseEstimate, np.ndarray]:
        """
        Estimate relative pose between two cameras from matches.

        Args:
            matches: Feature matches between views
            K1: Intrinsic matrix of first camera (3, 3)
            K2: Intrinsic matrix of second camera (same as K1 if None)

        Returns:
            Tuple of (PoseEstimate, inlier_mask)
        """
        if K2 is None:
            K2 = K1

        K1_np = K1.cpu().numpy() if isinstance(K1, torch.Tensor) else K1
        K2_np = K2.cpu().numpy() if isinstance(K2, torch.Tensor) else K2
        # Preserve device for output tensors
        device = K1.device if isinstance(K1, torch.Tensor) else 'cpu'

        pts1 = matches.matched_keypoints1.numpy()
        pts2 = matches.matched_keypoints2.numpy()

        if len(pts1) < 8:
            raise ValueError(f"Need at least 8 matches, got {len(pts1)}")

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            K1_np,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold
        )

        if E is None:
            raise RuntimeError("Failed to estimate essential matrix")

        # Decompose essential matrix to get R, t
        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E, pts1, pts2, K1_np, mask=mask
        )

        # Compute reprojection error for inliers
        inlier_mask = mask_pose.ravel() > 0
        if inlier_mask.sum() > 0:
            pts1_inlier = pts1[inlier_mask]
            pts2_inlier = pts2[inlier_mask]
            error = self._compute_reprojection_error(
                pts1_inlier, pts2_inlier, K1_np, K2_np, R, t
            )
        else:
            error = float('inf')

        pose = PoseEstimate(
            R=torch.from_numpy(R).float().to(device),
            t=torch.from_numpy(t.ravel()).float().to(device),
            inliers=int(inlier_mask.sum()),
            error=error
        )

        return pose, inlier_mask

    def _compute_reprojection_error(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        K1: np.ndarray,
        K2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> float:
        """Compute mean reprojection error."""
        # Triangulate points
        P1 = K1 @ np.eye(3, 4)
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3:]).T

        # Reproject to both views
        pts1_reproj, _ = cv2.projectPoints(
            pts3d, np.zeros(3), np.zeros(3), K1, None
        )
        pts2_reproj, _ = cv2.projectPoints(
            pts3d, cv2.Rodrigues(R)[0], t, K2, None
        )

        pts1_reproj = pts1_reproj.reshape(-1, 2)
        pts2_reproj = pts2_reproj.reshape(-1, 2)

        error1 = np.linalg.norm(pts1 - pts1_reproj, axis=1).mean()
        error2 = np.linalg.norm(pts2 - pts2_reproj, axis=1).mean()

        return (error1 + error2) / 2

    def estimate_multi_view_poses(
        self,
        all_matches: Dict[Tuple[int, int], Matches],
        intrinsics: List[CameraIntrinsics],
        n_views: int
    ) -> List[CameraExtrinsics]:
        """
        Estimate poses for all cameras in a multi-view setup.

        Uses the first camera as the world reference.

        Args:
            all_matches: Dictionary of pairwise matches
            intrinsics: Intrinsics for each camera
            n_views: Number of views

        Returns:
            List of CameraExtrinsics for each view
        """
        # First camera is at origin
        # Use device from intrinsics matrix if available
        device = intrinsics[0].to_matrix().device if hasattr(intrinsics[0].to_matrix(), 'device') else 'cpu'
        extrinsics = [
            CameraExtrinsics(
                R=torch.eye(3, device=device),
                t=torch.zeros(3, device=device)
            )
        ]

        # Estimate pose for each subsequent camera relative to first
        for v in range(1, n_views):
            if (0, v) in all_matches:
                matches = all_matches[(0, v)]
            elif (v, 0) in all_matches:
                # Swap order
                matches = all_matches[(v, 0)]
                matches = Matches(
                    matches_idx=matches.matches_idx.flip(1),
                    confidence=matches.confidence,
                    keypoints1=matches.keypoints2,
                    keypoints2=matches.keypoints1
                )
            else:
                raise ValueError(f"No matches between view 0 and view {v}")

            K1 = intrinsics[0].to_matrix()
            K2 = intrinsics[v].to_matrix()

            pose, _ = self.estimate_relative_pose(matches, K1, K2)

            extrinsics.append(CameraExtrinsics(
                R=pose.R,
                t=pose.t
            ))

        return extrinsics

    def create_cameras(
        self,
        intrinsics: List[CameraIntrinsics],
        extrinsics: List[CameraExtrinsics],
        image_sizes: List[Tuple[int, int]]
    ) -> List[Camera]:
        """
        Create Camera objects from intrinsics and extrinsics.

        Args:
            intrinsics: List of intrinsics
            extrinsics: List of extrinsics
            image_sizes: List of (width, height) tuples

        Returns:
            List of Camera objects
        """
        cameras = []
        for i in range(len(intrinsics)):
            cameras.append(Camera(
                intrinsics=intrinsics[i],
                extrinsics=extrinsics[i],
                width=image_sizes[i][0],
                height=image_sizes[i][1]
            ))
        return cameras


def decompose_essential_matrix(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose essential matrix into possible R, t pairs.

    Returns 4 possible solutions; the correct one needs to be
    determined by checking that triangulated points have positive depth.

    Args:
        E: Essential matrix (3, 3)

    Returns:
        List of 4 (R, t) tuples
    """
    U, S, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrix (det = 1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return solutions


def select_correct_pose(
    solutions: List[Tuple[np.ndarray, np.ndarray]],
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the correct R, t from 4 possible solutions.

    The correct solution is the one where triangulated points
    have positive depth in both cameras.

    Args:
        solutions: List of 4 (R, t) tuples
        pts1: Points in first image
        pts2: Points in second image
        K: Camera intrinsic matrix

    Returns:
        Correct (R, t) tuple
    """
    P1 = K @ np.eye(3, 4)

    best_solution = solutions[0]
    best_count = 0

    for R, t in solutions:
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3:]

        # Check depths
        # Depth in camera 1
        depth1 = pts3d[2]
        # Depth in camera 2
        pts3d_cam2 = R @ pts3d + t.reshape(3, 1)
        depth2 = pts3d_cam2[2]

        # Count points with positive depth in both cameras
        valid = (depth1 > 0) & (depth2 > 0)
        count = valid.sum()

        if count > best_count:
            best_count = count
            best_solution = (R, t)

    return best_solution
