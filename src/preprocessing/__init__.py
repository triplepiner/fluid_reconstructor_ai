"""Video preprocessing modules: loading, optical flow, and feature extraction."""

from .video_loader import VideoLoader, VideoMetadata, VideoData
from .optical_flow import OpticalFlowEstimator
from .feature_extraction import FeatureExtractor
from .depth_estimation import DepthEstimator, depth_to_pointcloud, estimate_camera_intrinsics
from .fluid_segmentation import FluidSegmenter, create_fluid_masks_from_pipeline_data, apply_mask_to_features

__all__ = [
    "VideoLoader",
    "VideoMetadata",
    "VideoData",
    "OpticalFlowEstimator",
    "FeatureExtractor",
    "DepthEstimator",
    "depth_to_pointcloud",
    "estimate_camera_intrinsics",
    "FluidSegmenter",
    "create_fluid_masks_from_pipeline_data",
    "apply_mask_to_features",
]
