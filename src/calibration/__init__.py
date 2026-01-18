"""Camera calibration modules for multi-view geometry estimation."""

from .feature_matching import FeatureMatcher
from .camera_estimation import CameraEstimator, CameraIntrinsics, CameraExtrinsics, Camera
from .bundle_adjustment import BundleAdjuster
from .triangulation import Triangulator, filter_outliers_statistical, estimate_scene_bounds
from .camera_motion import (
    CameraTrajectory,
    CameraMotionEstimator,
    VideoStabilizer,
    compensate_shake_in_triangulation
)

__all__ = [
    "FeatureMatcher",
    "CameraEstimator",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "Camera",
    "BundleAdjuster",
    "Triangulator",
    "filter_outliers_statistical",
    "estimate_scene_bounds",
    "CameraTrajectory",
    "CameraMotionEstimator",
    "VideoStabilizer",
    "compensate_shake_in_triangulation",
]
