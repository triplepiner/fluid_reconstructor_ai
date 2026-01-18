"""
Bundle adjustment for camera calibration refinement.

Jointly optimizes camera parameters and 3D point positions
to minimize reprojection error.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm
import cv2

from ..config import Camera, CameraIntrinsics, CameraExtrinsics


@dataclass
class BundleAdjustmentResult:
    """Result of bundle adjustment optimization."""
    cameras: List[Camera]
    points_3d: torch.Tensor  # (N, 3) optimized 3D points
    final_error: float  # Final mean reprojection error
    n_iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged


class BundleAdjuster:
    """
    Bundle adjustment using PyTorch for automatic differentiation.
    """

    def __init__(
        self,
        optimize_intrinsics: bool = False,
        optimize_distortion: bool = False,
        n_iterations: int = 100,
        lr: float = 1e-3,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize bundle adjuster.

        Args:
            optimize_intrinsics: Whether to optimize camera intrinsics
            optimize_distortion: Whether to optimize distortion coefficients
            n_iterations: Maximum number of optimization iterations
            lr: Learning rate for optimizer
            convergence_threshold: Convergence threshold for error change
        """
        self.optimize_intrinsics = optimize_intrinsics
        self.optimize_distortion = optimize_distortion
        self.n_iterations = n_iterations
        self.lr = lr
        self.convergence_threshold = convergence_threshold

    def optimize(
        self,
        cameras: List[Camera],
        points_3d: torch.Tensor,
        observations: Dict[Tuple[int, int], torch.Tensor],
        point_indices: Dict[Tuple[int, int], int]
    ) -> BundleAdjustmentResult:
        """
        Run bundle adjustment optimization.

        Args:
            cameras: List of initial camera estimates
            points_3d: Initial 3D point estimates (N, 3)
            observations: Dict mapping (view_idx, point_idx) to 2D observation (2,)
            point_indices: Same keys as observations, maps to index in points_3d

        Returns:
            BundleAdjustmentResult with optimized parameters
        """
        n_cameras = len(cameras)
        n_points = points_3d.shape[0]

        # Create optimizable parameters
        # Camera rotations as axis-angle (3,) for each camera except first
        rotations = []
        translations = []
        intrinsics_params = []

        # Determine device from input
        device = points_3d.device

        for i, cam in enumerate(cameras):
            if i == 0:
                # First camera is fixed at origin
                rot = torch.zeros(3, requires_grad=False, device=device)
                trans = torch.zeros(3, requires_grad=False, device=device)
            else:
                # Convert rotation matrix to axis-angle
                R = cam.extrinsics.R.cpu().numpy()
                rvec, _ = cv2.Rodrigues(R)
                rot = torch.from_numpy(rvec.ravel()).float().to(device).requires_grad_(True)
                trans = cam.extrinsics.t.clone().to(device).requires_grad_(True)

            rotations.append(rot)
            translations.append(trans)

            if self.optimize_intrinsics:
                intr = cam.intrinsics
                params = torch.tensor([intr.fx, intr.fy, intr.cx, intr.cy],
                                     requires_grad=True, device=device)
                intrinsics_params.append(params)

        # 3D points
        points_opt = points_3d.clone().requires_grad_(True)

        # Collect parameters
        params = [points_opt]
        for i in range(1, n_cameras):  # Skip first camera
            params.extend([rotations[i], translations[i]])

        if self.optimize_intrinsics:
            params.extend(intrinsics_params)

        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Prepare observation data
        obs_list = list(observations.items())

        prev_error = float('inf')
        converged = False

        for iteration in tqdm(range(self.n_iterations), desc="Bundle adjustment"):
            optimizer.zero_grad()

            total_error = torch.tensor(0.0, device=device)
            n_obs = 0

            for (view_idx, point_idx), obs_2d in obs_list:
                # Get camera parameters
                if view_idx == 0:
                    R = torch.eye(3, device=device)
                    t = torch.zeros(3, device=device)
                else:
                    R = self._rodrigues(rotations[view_idx])
                    t = translations[view_idx]

                if self.optimize_intrinsics:
                    K = self._intrinsics_to_matrix(intrinsics_params[view_idx])
                else:
                    K = cameras[view_idx].intrinsics.to_matrix().to(device)

                # Get 3D point
                pt_3d = points_opt[point_idx]

                # Project
                pt_cam = R @ pt_3d + t
                pt_2d = K[:2, :3] @ pt_cam / pt_cam[2]
                pt_2d = pt_2d[:2]

                # Reprojection error (ensure obs_2d is on device)
                obs_2d_device = obs_2d.to(device) if isinstance(obs_2d, torch.Tensor) else torch.tensor(obs_2d, device=device)
                error = torch.sum((pt_2d - obs_2d_device) ** 2)
                total_error = total_error + error
                n_obs += 1

            mean_error = torch.sqrt(total_error / n_obs)

            # Backprop
            mean_error.backward()
            optimizer.step()

            # Check convergence
            error_val = mean_error.item()
            if abs(prev_error - error_val) < self.convergence_threshold:
                converged = True
                break

            prev_error = error_val

        # Extract optimized cameras
        optimized_cameras = []
        for i, cam in enumerate(cameras):
            if i == 0:
                R = torch.eye(3, device=device)
                t = torch.zeros(3, device=device)
            else:
                R = self._rodrigues(rotations[i]).detach()
                t = translations[i].detach()

            if self.optimize_intrinsics:
                intr_p = intrinsics_params[i].detach()
                intrinsics = CameraIntrinsics(
                    fx=intr_p[0].item(),
                    fy=intr_p[1].item(),
                    cx=intr_p[2].item(),
                    cy=intr_p[3].item()
                )
            else:
                intrinsics = cam.intrinsics

            extrinsics = CameraExtrinsics(R=R, t=t)
            optimized_cameras.append(Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=cam.width,
                height=cam.height
            ))

        return BundleAdjustmentResult(
            cameras=optimized_cameras,
            points_3d=points_opt.detach(),
            final_error=prev_error,
            n_iterations=iteration + 1,
            converged=converged
        )

    def _rodrigues(self, rvec: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix (differentiable)."""
        theta = torch.norm(rvec)
        if theta < 1e-8:
            return torch.eye(3, device=rvec.device, dtype=rvec.dtype)

        k = rvec / theta
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=rvec.device, dtype=rvec.dtype)

        R = torch.eye(3, device=rvec.device, dtype=rvec.dtype) + \
            torch.sin(theta) * K + \
            (1 - torch.cos(theta)) * (K @ K)

        return R

    def _intrinsics_to_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """Convert intrinsics parameters to matrix."""
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = torch.zeros(3, 3, device=params.device, dtype=params.dtype)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        return K


def run_opencv_bundle_adjustment(
    cameras: List[Camera],
    points_3d: np.ndarray,
    observations: List[Tuple[int, int, np.ndarray]],
    n_iterations: int = 100
) -> Tuple[List[Camera], np.ndarray]:
    """
    Run bundle adjustment using OpenCV's solvePnPRansac iteratively.

    This is a simpler alternative when full BA is not needed.

    Args:
        cameras: List of initial cameras
        points_3d: Initial 3D points (N, 3)
        observations: List of (camera_idx, point_idx, observation_2d)
        n_iterations: Number of refinement iterations

    Returns:
        Tuple of (refined_cameras, refined_points)
    """
    n_cameras = len(cameras)
    n_points = len(points_3d)

    # Group observations by camera
    cam_observations = {i: [] for i in range(n_cameras)}
    for cam_idx, pt_idx, obs in observations:
        cam_observations[cam_idx].append((pt_idx, obs))

    refined_points = points_3d.copy()

    for _ in range(n_iterations):
        # Refine each camera pose
        for cam_idx in range(1, n_cameras):  # Skip first camera
            cam = cameras[cam_idx]
            obs = cam_observations[cam_idx]

            if len(obs) < 6:
                continue

            obj_points = np.array([refined_points[pt_idx] for pt_idx, _ in obs])
            img_points = np.array([o for _, o in obs])

            K = cam.intrinsics.to_matrix().cpu().numpy()

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                R_np, _ = cv2.Rodrigues(rvec)
                # Preserve original device
                orig_device = cam.extrinsics.R.device
                cameras[cam_idx] = Camera(
                    intrinsics=cam.intrinsics,
                    extrinsics=CameraExtrinsics(
                        R=torch.from_numpy(R_np).float().to(orig_device),
                        t=torch.from_numpy(tvec.ravel()).float().to(orig_device)
                    ),
                    width=cam.width,
                    height=cam.height
                )

        # Refine 3D points (triangulation)
        # Group observations by point
        point_observations = {i: [] for i in range(n_points)}
        for cam_idx, pt_idx, obs in observations:
            point_observations[pt_idx].append((cam_idx, obs))

        for pt_idx, obs_list in point_observations.items():
            if len(obs_list) < 2:
                continue

            # Multi-view triangulation
            A = []
            for cam_idx, obs in obs_list:
                cam = cameras[cam_idx]
                K = cam.intrinsics.to_matrix().cpu().numpy()
                R = cam.extrinsics.R.numpy()
                t = cam.extrinsics.t.numpy()
                P = K @ np.hstack([R, t.reshape(3, 1)])

                x, y = obs
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])

            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :3] / Vt[-1, 3]
            refined_points[pt_idx] = X

    return cameras, refined_points


def compute_reprojection_errors(
    cameras: List[Camera],
    points_3d: np.ndarray,
    observations: List[Tuple[int, int, np.ndarray]]
) -> np.ndarray:
    """
    Compute reprojection errors for all observations.

    Args:
        cameras: List of cameras
        points_3d: 3D points (N, 3)
        observations: List of (camera_idx, point_idx, observation_2d)

    Returns:
        Array of reprojection errors
    """
    errors = []

    for cam_idx, pt_idx, obs in observations:
        cam = cameras[cam_idx]
        device = cam.extrinsics.R.device
        pt_3d = torch.from_numpy(points_3d[pt_idx]).float().to(device)

        # Project
        pt_2d = cam.project(pt_3d.unsqueeze(0)).squeeze()

        error = torch.norm(pt_2d - torch.from_numpy(obs).float().to(device))
        errors.append(error.item())

    return np.array(errors)
