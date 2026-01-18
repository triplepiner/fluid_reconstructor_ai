"""
Multi-view triangulation for 3D point reconstruction.

Triangulates 3D points from 2D observations across multiple camera views.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import cv2

from ..config import Camera


@dataclass
class PointCloud:
    """Container for triangulated 3D point cloud."""
    points: torch.Tensor  # (N, 3) 3D points
    colors: Optional[torch.Tensor] = None  # (N, 3) RGB colors
    confidence: Optional[torch.Tensor] = None  # (N,) confidence scores
    views: Optional[List[List[int]]] = None  # Which views see each point

    def __len__(self):
        return self.points.shape[0]

    def filter_by_confidence(self, threshold: float) -> "PointCloud":
        """Return point cloud filtered by confidence threshold."""
        if self.confidence is None:
            return self

        mask = self.confidence >= threshold
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            confidence=self.confidence[mask],
            views=[v for i, v in enumerate(self.views) if mask[i]] if self.views else None
        )

    def to_numpy(self) -> np.ndarray:
        """Convert points to numpy array."""
        return self.points.cpu().numpy()


class Triangulator:
    """
    Triangulate 3D points from multi-view observations.
    """

    def __init__(
        self,
        min_views: int = 2,
        max_reprojection_error: float = 2.0,
        min_angle_degrees: float = 2.0
    ):
        """
        Initialize triangulator.

        Args:
            min_views: Minimum number of views required
            max_reprojection_error: Maximum allowed reprojection error
            min_angle_degrees: Minimum triangulation angle
        """
        self.min_views = min_views
        self.max_reprojection_error = max_reprojection_error
        self.min_angle_degrees = min_angle_degrees

    def triangulate_two_views(
        self,
        pts1: torch.Tensor,
        pts2: torch.Tensor,
        cam1: Camera,
        cam2: Camera
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Triangulate points from two views.

        Args:
            pts1: 2D points in first view (N, 2)
            pts2: Corresponding 2D points in second view (N, 2)
            cam1: First camera
            cam2: Second camera

        Returns:
            Tuple of (3D points (N, 3), reprojection errors (N,))
        """
        # Get device from camera
        device = cam1.extrinsics.R.device

        K1 = cam1.intrinsics.to_matrix().cpu().numpy()
        K2 = cam2.intrinsics.to_matrix().cpu().numpy()

        P1 = K1 @ cam1.extrinsics.to_matrix().cpu().numpy()
        P2 = K2 @ cam2.extrinsics.to_matrix().cpu().numpy()

        pts1_np = pts1.cpu().numpy() if isinstance(pts1, torch.Tensor) else pts1
        pts2_np = pts2.cpu().numpy() if isinstance(pts2, torch.Tensor) else pts2

        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1_np.T, pts2_np.T)
        pts3d = (pts4d[:3] / pts4d[3:]).T  # (N, 3)

        # Compute reprojection errors
        pts3d_torch = torch.from_numpy(pts3d).float().to(device)
        reproj1 = cam1.project(pts3d_torch)
        reproj2 = cam2.project(pts3d_torch)

        pts1_device = pts1.to(device) if isinstance(pts1, torch.Tensor) else torch.tensor(pts1, device=device)
        pts2_device = pts2.to(device) if isinstance(pts2, torch.Tensor) else torch.tensor(pts2, device=device)
        error1 = torch.norm(reproj1 - pts1_device, dim=1)
        error2 = torch.norm(reproj2 - pts2_device, dim=1)
        errors = (error1 + error2) / 2

        return pts3d_torch, errors

    def triangulate_multi_view(
        self,
        observations: Dict[int, torch.Tensor],
        cameras: List[Camera]
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Triangulate a single 3D point from multiple views using DLT.

        Args:
            observations: Dict mapping view_idx to 2D observation (2,)
            cameras: List of all cameras

        Returns:
            Tuple of (3D point (3,), mean error, per-view errors)
        """
        if len(observations) < self.min_views:
            raise ValueError(f"Need at least {self.min_views} views")

        # Get device from cameras
        device = cameras[0].extrinsics.R.device if cameras else 'cpu'

        # Build DLT matrix
        A = []
        for view_idx, obs in observations.items():
            cam = cameras[view_idx]
            K = cam.intrinsics.to_matrix().cpu().numpy()
            Rt = cam.extrinsics.to_matrix().cpu().numpy()
            P = K @ Rt

            x, y = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        A = np.array(A)

        # SVD solve
        _, S, Vt = np.linalg.svd(A)
        X = Vt[-1, :3] / Vt[-1, 3]

        pt_3d = torch.from_numpy(X).float().to(device)

        # Compute reprojection errors
        errors = []
        for view_idx, obs in observations.items():
            cam = cameras[view_idx]
            reproj = cam.project(pt_3d.unsqueeze(0)).squeeze()
            obs_tensor = obs.to(device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, device=device)
            error = torch.norm(reproj - obs_tensor)
            errors.append(error)

        errors_tensor = torch.stack(errors)
        mean_error = errors_tensor.mean()

        return pt_3d, mean_error.item(), errors_tensor

    def triangulate_matches(
        self,
        matches_dict: Dict[Tuple[int, int], "Matches"],
        cameras: List[Camera],
        frames: Optional[List[torch.Tensor]] = None
    ) -> PointCloud:
        """
        Triangulate all matched points across views.

        Args:
            matches_dict: Dict mapping (view_i, view_j) to Matches
            cameras: List of cameras
            frames: Optional frames for extracting point colors

        Returns:
            PointCloud with triangulated 3D points
        """
        from .feature_matching import Matches

        n_views = len(cameras)

        # Build tracks: sequences of observations of the same point
        # For simplicity, start with two-view triangulation
        all_points = []
        all_colors = []
        all_confidence = []
        all_views = []

        for (i, j), matches in matches_dict.items():
            pts1 = matches.matched_keypoints1
            pts2 = matches.matched_keypoints2

            if len(pts1) == 0:
                continue

            # Triangulate
            pts3d, errors = self.triangulate_two_views(
                pts1, pts2, cameras[i], cameras[j]
            )

            # Filter by reprojection error
            valid = errors < self.max_reprojection_error

            # Filter by triangulation angle
            angle_valid = self._check_triangulation_angle(
                pts3d, cameras[i], cameras[j]
            )
            valid = valid & angle_valid

            # Filter by depth (positive in both cameras)
            depth_valid = self._check_positive_depth(pts3d, cameras[i], cameras[j])
            valid = valid & depth_valid

            valid_pts = pts3d[valid]
            valid_conf = 1.0 / (errors[valid] + 0.1)  # Higher conf for lower error

            all_points.append(valid_pts)
            all_confidence.append(valid_conf)

            # Extract colors if frames provided
            if frames is not None:
                colors = self._extract_colors(
                    pts1[valid], frames[i]
                )
                all_colors.append(colors)

            # Record which views see each point
            n_valid = valid.sum().item()
            all_views.extend([[i, j]] * n_valid)

        if len(all_points) == 0:
            # Get device from cameras if available
            device = cameras[0].extrinsics.R.device if cameras else 'cpu'
            return PointCloud(
                points=torch.zeros(0, 3, device=device),
                colors=None,
                confidence=torch.zeros(0, device=device),
                views=[]
            )

        points = torch.cat(all_points, dim=0)
        confidence = torch.cat(all_confidence, dim=0)
        colors = torch.cat(all_colors, dim=0) if all_colors else None

        return PointCloud(
            points=points,
            colors=colors,
            confidence=confidence,
            views=all_views
        )

    def _check_triangulation_angle(
        self,
        pts3d: torch.Tensor,
        cam1: Camera,
        cam2: Camera
    ) -> torch.Tensor:
        """Check if triangulation angle is sufficient."""
        # Camera centers in world coordinates
        R1 = cam1.extrinsics.R
        t1 = cam1.extrinsics.t
        C1 = -R1.T @ t1  # Camera 1 center

        R2 = cam2.extrinsics.R
        t2 = cam2.extrinsics.t
        C2 = -R2.T @ t2  # Camera 2 center

        # Rays from cameras to points
        ray1 = pts3d - C1.unsqueeze(0)
        ray2 = pts3d - C2.unsqueeze(0)

        # Normalize rays
        ray1 = ray1 / torch.norm(ray1, dim=1, keepdim=True)
        ray2 = ray2 / torch.norm(ray2, dim=1, keepdim=True)

        # Compute angle
        cos_angle = torch.sum(ray1 * ray2, dim=1)
        angle_rad = torch.acos(torch.clamp(cos_angle, -1, 1))
        angle_deg = torch.rad2deg(angle_rad)

        return angle_deg >= self.min_angle_degrees

    def _check_positive_depth(
        self,
        pts3d: torch.Tensor,
        cam1: Camera,
        cam2: Camera
    ) -> torch.Tensor:
        """Check if points have positive depth in both cameras."""
        # Transform to camera coordinates
        pts_cam1 = (cam1.extrinsics.R @ pts3d.T).T + cam1.extrinsics.t
        pts_cam2 = (cam2.extrinsics.R @ pts3d.T).T + cam2.extrinsics.t

        return (pts_cam1[:, 2] > 0) & (pts_cam2[:, 2] > 0)

    def _extract_colors(
        self,
        pts2d: torch.Tensor,
        frame: torch.Tensor
    ) -> torch.Tensor:
        """Extract colors at 2D point locations."""
        H, W = frame.shape[:2]
        n_points = pts2d.shape[0]
        device = pts2d.device

        colors = torch.zeros(n_points, 3, device=device)

        for i, (x, y) in enumerate(pts2d.cpu()):
            x_int = int(torch.clamp(x, 0, W - 1))
            y_int = int(torch.clamp(y, 0, H - 1))
            colors[i] = frame[y_int, x_int].to(device)

        return colors


def filter_outliers_statistical(
    point_cloud: PointCloud,
    n_neighbors: int = 20,
    std_ratio: float = 2.0
) -> PointCloud:
    """
    Filter outlier points using statistical analysis.

    Removes points that are farther than std_ratio * std from their neighbors.

    Args:
        point_cloud: Input point cloud
        n_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation multiplier for threshold

    Returns:
        Filtered point cloud
    """
    points = point_cloud.points.cpu().numpy()
    n_points = len(points)

    if n_points < n_neighbors + 1:
        return point_cloud

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=n_neighbors + 1)

    # Exclude self-distance (first column)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Compute global statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold

    inlier_mask_torch = torch.from_numpy(inlier_mask)

    return PointCloud(
        points=point_cloud.points[inlier_mask_torch],
        colors=point_cloud.colors[inlier_mask_torch] if point_cloud.colors is not None else None,
        confidence=point_cloud.confidence[inlier_mask_torch] if point_cloud.confidence is not None else None,
        views=[v for i, v in enumerate(point_cloud.views) if inlier_mask[i]] if point_cloud.views else None
    )


def estimate_scene_bounds(point_cloud: PointCloud) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate scene bounding box from point cloud.

    Args:
        point_cloud: Input point cloud

    Returns:
        Tuple of (min_bounds (3,), max_bounds (3,))
    """
    points = point_cloud.points
    device = points.device if len(points) > 0 else 'cpu'

    if len(points) == 0:
        return torch.zeros(3, device=device), torch.ones(3, device=device)

    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]

    # Add small margin
    margin = (max_bounds - min_bounds) * 0.1
    min_bounds = min_bounds - margin
    max_bounds = max_bounds + margin

    return min_bounds, max_bounds
