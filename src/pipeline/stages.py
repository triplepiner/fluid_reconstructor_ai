"""
Pipeline stage definitions.

Each stage handles one step of the fluid reconstruction process.
"""

import torch
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import time
from tqdm import tqdm


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    status: StageStatus
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class PipelineStage:
    """
    Base class for pipeline stages.
    """

    name: str = "base"
    description: str = "Base pipeline stage"

    def __init__(self, config):
        """
        Initialize stage.

        Args:
            config: PipelineConfig
        """
        self.config = config
        self.status = StageStatus.PENDING
        self.result = None

    def run(self, inputs: Dict[str, Any]) -> StageResult:
        """
        Execute the stage.

        Args:
            inputs: Input data from previous stages

        Returns:
            StageResult
        """
        self.status = StageStatus.RUNNING
        start_time = time.time()

        try:
            output = self._execute(inputs)
            duration = time.time() - start_time

            self.result = StageResult(
                status=StageStatus.COMPLETED,
                output=output,
                duration_seconds=duration
            )
            self.status = StageStatus.COMPLETED

        except Exception as e:
            duration = time.time() - start_time
            self.result = StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_seconds=duration
            )
            self.status = StageStatus.FAILED
            raise

        return self.result

    def _execute(self, inputs: Dict[str, Any]) -> Any:
        """Override in subclass to implement stage logic."""
        raise NotImplementedError


class VideoLoadingStage(PipelineStage):
    """Stage 0-1: Load videos and extract frames."""

    name = "video_loading"
    description = "Load videos and extract frames"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..preprocessing import VideoLoader
        from pathlib import Path

        video_paths = inputs['video_paths']

        # Convert to Path objects and validate
        paths = []
        for p in video_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Not a file: {path}")
            paths.append(path)

        loader = VideoLoader(
            max_resolution=self.config.max_resolution,
            device="cpu"  # Load to CPU first, move later
        )

        videos = []
        pbar = tqdm(paths, desc="    Loading videos", unit="video", leave=False)
        for path in pbar:
            pbar.set_postfix(file=path.name[:20])
            try:
                video = loader.load(path)
                print(f"    Loaded {path.name}: {video.metadata.n_frames} frames, "
                      f"{video.metadata.fps:.1f} fps, {video.metadata.width}x{video.metadata.height}")
                videos.append(video)
            except Exception as e:
                raise RuntimeError(f"Failed to load {path.name}: {e}")
        pbar.close()

        # Validate videos
        warnings = VideoLoader.validate_videos_for_reconstruction(videos)
        if warnings:
            print("    Warnings:")
            for w in warnings:
                print(f"      - {w}")

        return {'videos': videos}


class VideoStabilizationStage(PipelineStage):
    """Stage 1a: Stabilize videos to handle camera shake."""

    name = "video_stabilization"
    description = "Stabilize videos and track camera motion"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..calibration import VideoStabilizer, CameraMotionEstimator

        videos = inputs['videos']

        if not self.config.enable_stabilization and not self.config.track_camera_motion:
            print("    Skipping stabilization (disabled in config)")
            return {'videos': videos, 'camera_trajectories': None}

        stabilizer = VideoStabilizer(
            smoothing_radius=self.config.stabilization_smoothing
        )
        motion_estimator = CameraMotionEstimator()

        camera_trajectories = []

        pbar = tqdm(videos, desc="    Processing videos", unit="video", leave=False)
        for video in pbar:
            pbar.set_postfix(file=video.metadata.path.name[:15])

            if self.config.enable_stabilization:
                # Stabilize frames to remove shake
                stabilized_frames, _ = stabilizer.stabilize(video.frames)
                video.frames = stabilized_frames
                print(f"    Stabilized {video.metadata.path.name}")

            if self.config.track_camera_motion:
                # Track residual camera motion for later compensation
                from ..config import Camera, CameraIntrinsics, CameraExtrinsics

                # Create approximate camera from video metadata
                H, W = video.frames.shape[1:3]
                fx = fy = W  # Rough approximation
                intrinsics = CameraIntrinsics(
                    fx=fx, fy=fy,
                    cx=W / 2, cy=H / 2
                )
                extrinsics = CameraExtrinsics(
                    R=torch.eye(3, device=self.config.device),
                    t=torch.zeros(3, device=self.config.device)
                )
                base_camera = Camera(
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    width=W, height=H
                )

                trajectory = motion_estimator.estimate_trajectory(
                    video.frames,
                    base_camera,
                    video.timestamps
                )
                camera_trajectories.append(trajectory)
            else:
                camera_trajectories.append(None)
        pbar.close()

        return {
            'videos': videos,
            'camera_trajectories': camera_trajectories
        }


class OpticalFlowStage(PipelineStage):
    """Stage 1b: Compute optical flow."""

    name = "optical_flow"
    description = "Compute optical flow for all videos"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..preprocessing import OpticalFlowEstimator

        videos = inputs['videos']
        estimator = OpticalFlowEstimator(device=self.config.device)

        pbar = tqdm(videos, desc="    Computing flow", unit="video", leave=False)
        for video in pbar:
            pbar.set_postfix(frames=video.metadata.n_frames)
            video.optical_flow = estimator.compute_for_video(video)
        pbar.close()

        return {'videos': videos}


class FluidSegmentationStage(PipelineStage):
    """Stage 1c: Segment fluid regions from video."""

    name = "fluid_segmentation"
    description = "Isolate fluid/liquid regions using motion analysis"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..preprocessing import FluidSegmenter

        videos = inputs['videos']

        segmenter = FluidSegmenter(
            motion_threshold=self.config.fluid_motion_threshold,
            min_area_ratio=0.001,
            max_area_ratio=0.8,
            temporal_window=5,
            device=self.config.device
        )

        all_fluid_masks = []

        pbar = tqdm(enumerate(videos), total=len(videos), desc="    Segmenting fluid", unit="video", leave=False)
        for v_idx, video in pbar:
            pbar.set_postfix(frames=video.metadata.n_frames)

            if video.optical_flow is None:
                print(f"    Warning: No optical flow for video {v_idx}, skipping segmentation")
                # Create full masks (no segmentation)
                h, w = video.frames.shape[1:3]
                masks = [torch.ones(h, w) for _ in range(video.metadata.n_frames)]
            else:
                # Segment using optical flow
                masks = segmenter.segment_from_flow(video.optical_flow, video.frames)

                # Optionally refine with appearance
                if self.config.use_appearance_segmentation:
                    masks = segmenter.refine_with_appearance(masks, video.frames)

            all_fluid_masks.append(masks)

            # Compute and report coverage
            coverage = sum(m.mean().item() for m in masks) / len(masks) * 100
            print(f"    View {v_idx}: {coverage:.1f}% average fluid coverage")

        pbar.close()

        return {**inputs, 'fluid_masks': all_fluid_masks}


class TemporalSyncStage(PipelineStage):
    """Stage 2: Temporal synchronization."""

    name = "temporal_sync"
    description = "Synchronize video timelines"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..synchronization import (
            MotionSignatureExtractor,
            TemporalAligner,
            FrameInterpolator
        )
        from ..config import AlignedFrames, SyncParameters

        videos = inputs['videos']
        n_views = len(videos)

        # Single video: no synchronization needed
        if n_views == 1:
            print("    Single video mode - skipping synchronization")
            video = videos[0]
            # Create aligned frames from single video
            aligned_frames = AlignedFrames(
                frames=video.frames.unsqueeze(0),  # (1, T, H, W, 3)
                timestamps=video.timestamps
            )
            sync_params = SyncParameters(
                offsets=[0.0],
                common_fps=video.metadata.fps,
                common_start=0.0,
                common_end=video.metadata.duration,
                n_common_frames=video.metadata.n_frames
            )
            return {
                'videos': videos,
                'sync_params': sync_params,
                'aligned_frames': aligned_frames,
                'single_video_mode': True
            }

        # Multi-view: extract motion signatures and align
        print("    Extracting motion signatures...")
        extractor = MotionSignatureExtractor()

        signatures = []
        pbar = tqdm(videos, desc="    Motion signatures", unit="video", leave=False)
        for v in pbar:
            sig = extractor.extract_from_video(v)
            signatures.append(sig)
        pbar.close()

        # Align
        print("    Finding temporal alignment...")
        aligner = TemporalAligner(
            max_offset=self.config.max_offset_seconds,
            min_offset=self.config.min_offset_seconds
        )
        sync_params, _ = aligner.align_multiple(signatures)
        print(f"    Found offsets: {sync_params.offsets}")

        # Interpolate to common timeline
        print("    Interpolating to common timeline...")
        interpolator = FrameInterpolator(device=self.config.device)
        aligned_frames = interpolator.interpolate_videos(videos, sync_params)

        return {
            'videos': videos,
            'sync_params': sync_params,
            'aligned_frames': aligned_frames,
            'single_video_mode': False
        }


class FeatureExtractionStage(PipelineStage):
    """Stage 3a: Extract features for calibration."""

    name = "feature_extraction"
    description = "Extract features from aligned frames"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..preprocessing import FeatureExtractor

        aligned_frames = inputs['aligned_frames']
        extractor = FeatureExtractor(
            method="superpoint" if self.config.device == "cuda" else "sift",
            device=self.config.device
        )

        n_views = aligned_frames.frames.shape[0]
        print(f"    Using {extractor.method} feature extractor")

        # Extract features from first frame of each view
        features_list = []
        pbar = tqdm(range(n_views), desc="    Extracting features", unit="view", leave=False)
        for v in pbar:
            features = extractor.extract(aligned_frames.frames[v, 0])
            features_list.append(features)
            pbar.set_postfix(keypoints=len(features.keypoints) if hasattr(features, 'keypoints') else 'N/A')
        pbar.close()

        return {**inputs, 'features': features_list}


class CameraCalibrationStage(PipelineStage):
    """Stage 3: Camera calibration."""

    name = "camera_calibration"
    description = "Estimate camera intrinsics and extrinsics"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..calibration import (
            FeatureMatcher,
            CameraEstimator
        )
        from ..config import Camera, CameraIntrinsics, CameraExtrinsics

        aligned_frames = inputs['aligned_frames']
        features_list = inputs['features']
        videos = inputs['videos']
        single_video_mode = inputs.get('single_video_mode', False)

        n_views = len(videos)

        # Single video mode: use estimated intrinsics, identity extrinsics
        if single_video_mode:
            print("    Single video mode - using estimated camera parameters")
            video = videos[0]
            W, H = video.metadata.width, video.metadata.height

            # Estimate intrinsics assuming ~60 degree FOV
            import numpy as np
            fov_rad = np.radians(60)
            fx = W / (2 * np.tan(fov_rad / 2))

            intrinsics = CameraIntrinsics(
                fx=fx, fy=fx,
                cx=W / 2, cy=H / 2
            )
            extrinsics = CameraExtrinsics(
                R=torch.eye(3, device=self.config.device),
                t=torch.zeros(3, device=self.config.device)
            )
            camera = Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=W, height=H
            )
            print(f"    Estimated focal length: {fx:.1f}px (60 deg FOV)")
            return {**inputs, 'cameras': [camera], 'matches': {}}

        # Multi-view: match features and calibrate
        print("    Matching features between views...")
        matcher = FeatureMatcher(device=self.config.device)
        all_matches = matcher.match_multiple(features_list)

        # Estimate intrinsics
        print("    Estimating camera intrinsics...")
        estimator = CameraEstimator()
        intrinsics = []
        for v, video in enumerate(videos):
            intr = estimator.estimate_intrinsics_from_video(
                video.metadata.width,
                video.metadata.height
            )
            intrinsics.append(intr)

        # Estimate extrinsics
        print("    Estimating camera poses...")
        extrinsics = estimator.estimate_multi_view_poses(
            all_matches, intrinsics, n_views
        )

        # Create cameras
        image_sizes = [(v.metadata.width, v.metadata.height) for v in videos]
        cameras = estimator.create_cameras(intrinsics, extrinsics, image_sizes)

        return {**inputs, 'cameras': cameras, 'matches': all_matches}


class TriangulationStage(PipelineStage):
    """Stage 4: Initial point cloud triangulation or depth estimation."""

    name = "triangulation"
    description = "Build initial 3D point cloud"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..calibration import Triangulator, filter_outliers_statistical
        from ..calibration.triangulation import PointCloud

        cameras = inputs['cameras']
        matches = inputs['matches']
        aligned_frames = inputs['aligned_frames']
        single_video_mode = inputs.get('single_video_mode', False)
        fluid_masks = inputs.get('fluid_masks', None)

        # Single video mode: use monocular depth estimation
        if single_video_mode:
            print("    Single video mode - using monocular depth estimation")
            from ..preprocessing import DepthEstimator, depth_to_pointcloud

            # Get first frame
            frame = aligned_frames.frames[0, 0]  # (H, W, 3)
            camera = cameras[0]

            # Get fluid mask if available
            fluid_mask = None
            if fluid_masks is not None and len(fluid_masks) > 0 and len(fluid_masks[0]) > 0:
                fluid_mask = fluid_masks[0][0]  # First view, first frame
                print(f"    Using fluid mask ({fluid_mask.mean().item()*100:.1f}% coverage)")

            # Estimate depth with progress
            print("    Loading depth model...")
            depth_estimator = DepthEstimator(
                model_type="midas_small",  # Use smaller model for speed
                device=self.config.device
            )

            print("    Estimating depth...")
            with tqdm(total=4, desc="    Depth estimation", leave=False) as pbar:
                pbar.set_postfix(step="inference")
                depth = depth_estimator.estimate(frame)
                pbar.update(1)

                # Convert to point cloud
                pbar.set_postfix(step="point cloud")
                fx, fy = camera.intrinsics.fx, camera.intrinsics.fy
                cx, cy = camera.intrinsics.cx, camera.intrinsics.cy

                points, colors = depth_to_pointcloud(
                    depth, frame,
                    fx, fy, cx, cy,
                    depth_scale=10.0  # Scale factor for reasonable depth range
                )
                pbar.update(1)

                # Apply fluid mask to filter points
                pbar.set_postfix(step="fluid filtering")
                if fluid_mask is not None:
                    H, W = depth.shape
                    # Create pixel coordinates for each point
                    y_coords = torch.arange(H, device=points.device).float()
                    x_coords = torch.arange(W, device=points.device).float()
                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

                    # Flatten and match point structure
                    mask_flat = fluid_mask.flatten().to(points.device)

                    # Filter by mask
                    fluid_indices = mask_flat > 0.5
                    points = points[fluid_indices]
                    colors = colors[fluid_indices]
                    print(f"    Filtered to {len(points):,} fluid points")
                pbar.update(1)

                # Subsample if too many points
                pbar.set_postfix(step="subsampling")
                max_points = self.config.initial_n_gaussians
                if len(points) > max_points:
                    indices = torch.randperm(len(points))[:max_points]
                    points = points[indices]
                    colors = colors[indices]
                pbar.update(1)

            point_cloud = PointCloud(
                points=points,
                colors=colors,
                confidence=torch.ones(len(points), device=points.device)
            )

            # Store depth maps for later use
            inputs['depth_maps'] = depth.unsqueeze(0)  # (1, H, W)

            print(f"    Generated {len(points):,} points from depth")
            return {**inputs, 'point_cloud': point_cloud}

        # Multi-view: triangulate from matches
        print("    Triangulating from feature matches...")
        triangulator = Triangulator()
        point_cloud = triangulator.triangulate_matches(
            matches, cameras,
            frames=[aligned_frames.frames[v, 0] for v in range(len(cameras))]
        )

        # Filter outliers
        print("    Filtering outliers...")
        point_cloud = filter_outliers_statistical(point_cloud)

        print(f"    Triangulated {len(point_cloud):,} points")
        return {**inputs, 'point_cloud': point_cloud}


class GaussianInitStage(PipelineStage):
    """Stage 5: Initialize Gaussians from point cloud."""

    name = "gaussian_init"
    description = "Initialize 3D Gaussians"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..gaussian_splatting import GaussianCloud, DynamicGaussianCloud

        point_cloud = inputs['point_cloud']
        sync_params = inputs['sync_params']

        n_points = len(point_cloud.points)
        n_frames = sync_params.n_common_frames

        print(f"    Initializing {n_points:,} Gaussians for {n_frames} frames...")

        with tqdm(total=2, desc="    Gaussian init", leave=False) as pbar:
            # Initialize static Gaussian cloud
            pbar.set_postfix(step="base cloud")
            base_gaussians = GaussianCloud(
                sh_degree=self.config.sh_degree,
                device=self.config.device
            )
            base_gaussians.initialize_from_point_cloud(
                point_cloud.points,
                point_cloud.colors
            )
            pbar.update(1)

            # Create dynamic version
            pbar.set_postfix(step="dynamic cloud")
            dynamic_gaussians = DynamicGaussianCloud(
                base_gaussians,
                n_timesteps=n_frames,
                temporal_mode="velocity",
                device=self.config.device
            )
            pbar.update(1)

        print(f"    Initialized with SH degree {self.config.sh_degree}")
        return {**inputs, 'gaussians': dynamic_gaussians}


class ReconstructionStage(PipelineStage):
    """Stage 6: Main Gaussian reconstruction optimization."""

    name = "reconstruction"
    description = "Optimize Gaussian representation"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..gaussian_splatting import GaussianRasterizer, GaussianLosses

        gaussians = inputs['gaussians']
        cameras = inputs['cameras']
        aligned_frames = inputs['aligned_frames']
        fluid_masks = inputs.get('fluid_masks', None)

        # Setup rasterizer
        H, W = aligned_frames.frames.shape[2:4]
        rasterizer = GaussianRasterizer(H, W, device=self.config.device)

        # Setup losses
        losses = GaussianLosses(
            photometric_weight=self.config.loss_weights.photometric,
            ssim_weight=self.config.loss_weights.ssim,
            temporal_weight=self.config.loss_weights.temporal_consistency
        )

        # Setup optimizer with per-parameter learning rates
        param_groups = gaussians.get_all_parameters()
        optimizer = torch.optim.Adam(param_groups, lr=self.config.optimization.lr_position)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.995
        )

        # Training parameters
        n_epochs = self.config.optimization.n_epochs
        n_views = len(cameras)
        n_frames = aligned_frames.frames.shape[1]

        # Densification settings
        densify_interval = max(100, n_epochs // 10)
        densify_until = int(n_epochs * 0.8)  # Stop densifying in last 20%
        prune_interval = max(50, n_epochs // 20)

        loss_history = []
        best_loss = float('inf')
        best_state = None

        # Compute total iterations (iterate all views and sample frames)
        # Use fewer frames for faster iteration when gsplat is not available
        if rasterizer.gsplat_available:
            frames_per_epoch = min(n_frames, 5)  # Sample multiple frames with fast renderer
        else:
            frames_per_epoch = 1  # Use single frame for slow Python fallback
            print("    Note: Using Python fallback renderer (slower)")
        iterations_per_epoch = n_views * frames_per_epoch
        total_iterations = n_epochs * iterations_per_epoch

        print(f"    Training for {n_epochs:,} epochs ({total_iterations:,} iterations)")
        print(f"    Views: {n_views}, Frames: {n_frames}")
        print(f"    Densification every {densify_interval} epochs until epoch {densify_until}")
        if fluid_masks is not None:
            print(f"    Using fluid masks for loss computation")

        epoch_pbar = tqdm(range(n_epochs), desc="    Training", unit="epoch", leave=False)
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            epoch_samples = 0

            # Sample frames for this epoch
            frame_indices = torch.randperm(n_frames)[:frames_per_epoch].tolist()

            # Iterate over all views and sampled frames
            for v_idx in range(n_views):
                camera = cameras[v_idx]

                for t_idx in frame_indices:
                    # Get target frame
                    target = aligned_frames.frames[v_idx, t_idx].to(self.config.device)

                    # Get fluid mask if available
                    mask = None
                    if fluid_masks is not None and v_idx < len(fluid_masks):
                        view_masks = fluid_masks[v_idx]
                        if t_idx < len(view_masks):
                            mask = view_masks[t_idx].to(self.config.device)

                    # Get Gaussians at this timestep
                    g = gaussians.get_gaussians_at_time(t_idx)

                    # Render
                    render_result = rasterizer(g, camera)
                    rendered = render_result['image']

                    # Compute loss (with optional mask)
                    if mask is not None:
                        # Expand mask to 3 channels
                        mask_3ch = mask.unsqueeze(-1).expand_as(rendered)
                        # Apply mask to both rendered and target
                        rendered_masked = rendered * mask_3ch
                        target_masked = target * mask_3ch
                        # Also add penalty for rendering outside mask region
                        outside_mask = 1.0 - mask_3ch
                        outside_penalty = (rendered * outside_mask).mean() * 0.1
                        loss_dict = losses(rendered_masked, target_masked, g)
                        loss = loss_dict['total'] + outside_penalty
                    else:
                        loss_dict = losses(rendered, target, g)
                        loss = loss_dict['total']

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_samples += 1

            # Average epoch loss
            avg_loss = epoch_loss / max(epoch_samples, 1)
            loss_history.append(avg_loss)

            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = gaussians.state_dict()

            # Update progress bar
            n_gaussians = gaussians.base_gaussians.n_gaussians
            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}",
                n_g=f"{n_gaussians:,}"
            )

            # Learning rate scheduling
            scheduler.step()

            # Adaptive densification
            if epoch > 0 and epoch < densify_until and epoch % densify_interval == 0:
                self._densify_gaussians(gaussians)

            # Pruning (can continue until end)
            if epoch > 0 and epoch % prune_interval == 0:
                self._prune_gaussians(gaussians)

            # Checkpoint
            if epoch > 0 and epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gaussians, epoch)

        epoch_pbar.close()

        # Restore best state
        if best_state is not None:
            gaussians.load_state_dict(best_state)
            print(f"    Restored best model (loss: {best_loss:.6f})")

        print(f"    Final: {n_gaussians:,} Gaussians, loss: {best_loss:.6f}")

        return {**inputs, 'gaussians': gaussians, 'loss_history': loss_history}

    def _densify_gaussians(self, gaussians):
        """Densify Gaussians with high gradients."""
        base = gaussians.base_gaussians
        n_before = base.n_gaussians

        base.densify_and_prune(
            grad_threshold=self.config.optimization.grad_threshold,
            scale_threshold=self.config.optimization.scale_threshold,
            opacity_threshold=self.config.optimization.opacity_threshold
        )

        n_after = base.n_gaussians
        if n_after != n_before:
            print(f"    Densified: {n_before:,} -> {n_after:,} Gaussians")

    def _prune_gaussians(self, gaussians):
        """Prune low-opacity Gaussians."""
        base = gaussians.base_gaussians
        n_before = base.n_gaussians

        # Only prune, don't densify
        opacity_mask = base.opacities.squeeze() < 0.01
        if opacity_mask.any():
            base._prune(opacity_mask)

        n_after = base.n_gaussians
        if n_after != n_before:
            print(f"    Pruned: {n_before:,} -> {n_after:,} Gaussians")

    def _save_checkpoint(self, gaussians, epoch: int):
        """Save checkpoint."""
        path = self.config.checkpoint_dir / f"gaussians_epoch_{epoch}.pt"
        torch.save(gaussians.state_dict(), path)


class PhysicsEstimationStage(PipelineStage):
    """Stage 7-9: Physics property estimation."""

    name = "physics_estimation"
    description = "Estimate fluid properties (velocity, pressure, density, viscosity)"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ..physics import (
            FluidFields,
            FieldBounds,
            NavierStokesLoss,
            ViscosityEstimator
        )
        from ..calibration import estimate_scene_bounds

        gaussians = inputs['gaussians']
        point_cloud = inputs['point_cloud']

        # Estimate scene bounds
        print("    Estimating scene bounds...")
        min_bounds, max_bounds = estimate_scene_bounds(point_cloud)
        bounds = FieldBounds(min_bounds, max_bounds)

        # Initialize fluid fields
        resolution = (64, 64, 64)
        n_timesteps = gaussians.n_timesteps

        print(f"    Initializing {resolution} fluid fields for {n_timesteps} frames...")
        fluid_fields = FluidFields(
            resolution, bounds, n_timesteps,
            device=self.config.device
        )

        # Initialize from Gaussians with progress
        frames_to_init = min(n_timesteps, 10)
        pbar = tqdm(range(frames_to_init), desc="    Init fields", unit="frame", leave=False)
        for t in pbar:
            fluid_fields.initialize_from_gaussians(gaussians, t)
        pbar.close()

        # Estimate viscosity
        print("    Estimating viscosity...")
        viscosity_estimator = ViscosityEstimator()
        if n_timesteps > 1:
            viscosity_estimate = viscosity_estimator.estimate_combined(
                fluid_fields.velocity.data[0],
                fluid_fields.velocity.data[1],
                fluid_fields.density.data[0],
                dt=1.0 / 30.0,  # Assume 30 fps
                spacing=fluid_fields.velocity.spacing
            )
            print(f"    Estimated viscosity: {viscosity_estimate.kinematic_viscosity:.6f} m^2/s")
            print(f"    Confidence: {viscosity_estimate.confidence:.2f}")

        return {**inputs, 'fluid_fields': fluid_fields}


class OutputStage(PipelineStage):
    """Stage 10: Save outputs."""

    name = "output"
    description = "Save results and generate visualizations"

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        gaussians = inputs['gaussians']
        fluid_fields = inputs.get('fluid_fields')

        output_dir = self.config.output_dir

        print("    Saving outputs...")
        with tqdm(total=3, desc="    Saving", leave=False) as pbar:
            # Save Gaussians
            pbar.set_postfix(file="gaussians")
            torch.save(
                gaussians.state_dict(),
                output_dir / "final_gaussians.pt"
            )
            pbar.update(1)

            # Save fluid fields
            if fluid_fields is not None:
                pbar.set_postfix(file="fluid_fields")
                torch.save({
                    'velocity': fluid_fields.velocity.data,
                    'pressure': fluid_fields.pressure.data,
                    'density': fluid_fields.density.data,
                    'viscosity': fluid_fields.viscosity.item()
                }, output_dir / "fluid_fields.pt")
            pbar.update(1)

            # Summary
            pbar.set_postfix(file="summary")
            pbar.update(1)

        print(f"    Results saved to {output_dir}")

        return inputs
