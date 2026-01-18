"""
Video loading and metadata extraction.

Handles videos with different FPS, resolutions, codecs, and zoom levels.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

from ..config import VideoMetadata, VideoData


class VideoLoader:
    """Load and preprocess video files."""

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    SUPPORTED_CODECS = {'h264', 'h265', 'hevc', 'vp9', 'av1', 'prores', 'mjpeg'}

    def __init__(
        self,
        max_resolution: Optional[int] = None,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize VideoLoader.

        Args:
            max_resolution: Maximum resolution (height) to resize to
            normalize: Whether to normalize pixel values to [0, 1]
            device: Device to load tensors to
        """
        self.max_resolution = max_resolution
        self.normalize = normalize
        self.device = device

    def load(self, video_path: Union[str, Path]) -> VideoData:
        """
        Load a video file and extract frames.

        Args:
            video_path: Path to the video file

        Returns:
            VideoData object containing frames and metadata
        """
        video_path = Path(video_path)
        self._validate_path(video_path)

        # Extract metadata first
        metadata = self.extract_metadata(video_path)

        # Load frames
        frames, timestamps = self._load_frames(video_path, metadata)

        return VideoData(
            metadata=metadata,
            frames=frames,
            timestamps=timestamps,
            optical_flow=None
        )

    def extract_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """
        Extract metadata from a video file without loading all frames.

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata object
        """
        video_path = Path(video_path)
        self._validate_path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = n_frames / fps if fps > 0 else 0

            # Get codec (fourcc)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            # Estimate bitrate if available
            bitrate = None
            try:
                bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
            except:
                pass

            return VideoMetadata(
                path=video_path,
                fps=fps,
                width=width,
                height=height,
                duration=duration,
                n_frames=n_frames,
                codec=codec.strip(),
                bitrate=bitrate
            )
        finally:
            cap.release()

    def _validate_path(self, video_path: Path):
        """Validate that the video path exists and has a supported extension."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

    def _load_frames(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load all frames from a video.

        Args:
            video_path: Path to the video file
            metadata: Video metadata

        Returns:
            Tuple of (frames tensor, timestamps tensor)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            frames = []
            timestamps = []

            # Calculate resize dimensions if needed
            target_height, target_width = self._get_target_dimensions(
                metadata.height, metadata.width
            )

            for i in tqdm(range(metadata.n_frames), desc=f"Loading {video_path.name}"):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                if (target_height != metadata.height or
                    target_width != metadata.width):
                    frame = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                frames.append(frame)
                timestamps.append(i / metadata.fps)

            # Convert to tensors
            frames_np = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float()

            if self.normalize:
                frames_tensor = frames_tensor / 255.0

            # Use float32 for MPS compatibility (MPS doesn't support float64)
            timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

            return frames_tensor.to(self.device), timestamps_tensor.to(self.device)

        finally:
            cap.release()

    def _get_target_dimensions(
        self,
        height: int,
        width: int
    ) -> Tuple[int, int]:
        """
        Calculate target dimensions based on max_resolution.

        Args:
            height: Original height
            width: Original width

        Returns:
            Tuple of (target_height, target_width)
        """
        if self.max_resolution is None:
            # Still ensure divisible by 8 for RAFT optical flow
            target_height = (height // 8) * 8
            target_width = (width // 8) * 8
            return target_height if target_height > 0 else 8, target_width if target_width > 0 else 8

        if height <= self.max_resolution:
            # Ensure divisible by 8
            target_height = (height // 8) * 8
            target_width = (width // 8) * 8
            return target_height if target_height > 0 else 8, target_width if target_width > 0 else 8

        scale = self.max_resolution / height
        target_height = self.max_resolution
        target_width = int(width * scale)

        # Ensure dimensions are divisible by 8 (required by RAFT optical flow)
        target_height = (target_height // 8) * 8
        target_width = (target_width // 8) * 8

        # Ensure minimum dimensions
        target_height = max(target_height, 8)
        target_width = max(target_width, 8)

        return target_height, target_width

    def load_multiple(
        self,
        video_paths: List[Union[str, Path]],
        parallel: bool = False
    ) -> List[VideoData]:
        """
        Load multiple video files.

        Args:
            video_paths: List of paths to video files
            parallel: Whether to load in parallel (not implemented yet)

        Returns:
            List of VideoData objects
        """
        videos = []
        for path in video_paths:
            video = self.load(path)
            videos.append(video)
        return videos

    @staticmethod
    def validate_videos_for_reconstruction(
        videos: List[VideoData],
        min_duration: float = 1.0,
        min_fps: float = 10.0
    ) -> List[str]:
        """
        Validate that videos are suitable for fluid reconstruction.

        Args:
            videos: List of VideoData objects
            min_duration: Minimum required duration in seconds
            min_fps: Minimum required FPS

        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []

        for i, video in enumerate(videos):
            metadata = video.metadata
            video_name = metadata.path.name

            if metadata.duration < min_duration:
                warnings.append(
                    f"Video {video_name}: Duration ({metadata.duration:.2f}s) "
                    f"is less than minimum ({min_duration}s)"
                )

            if metadata.fps < min_fps:
                warnings.append(
                    f"Video {video_name}: FPS ({metadata.fps:.1f}) "
                    f"is less than minimum ({min_fps})"
                )

        # Check resolution consistency (warn if very different)
        if len(videos) > 1:
            resolutions = [(v.metadata.width, v.metadata.height) for v in videos]
            areas = [w * h for w, h in resolutions]
            max_area = max(areas)
            min_area = min(areas)

            if max_area / min_area > 4:  # More than 2x difference in each dimension
                warnings.append(
                    "Videos have significantly different resolutions. "
                    "This may affect reconstruction quality."
                )

        return warnings


def try_load_with_decord(video_path: Path) -> Optional[VideoData]:
    """
    Try to load video using decord (faster, GPU-accelerated).

    Falls back to OpenCV if decord is not available.
    """
    try:
        import decord
        from decord import VideoReader, cpu, gpu

        # Try GPU first, fall back to CPU
        try:
            ctx = gpu(0)
        except:
            ctx = cpu(0)

        vr = VideoReader(str(video_path), ctx=ctx)
        fps = vr.get_avg_fps()
        n_frames = len(vr)

        # Get frame dimensions from first frame
        first_frame = vr[0].asnumpy()
        height, width = first_frame.shape[:2]

        # Load all frames
        frames = vr.get_batch(list(range(n_frames))).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0

        # Use float32 for MPS compatibility (MPS doesn't support float64)
        timestamps = torch.tensor([i / fps for i in range(n_frames)], dtype=torch.float32)

        metadata = VideoMetadata(
            path=video_path,
            fps=fps,
            width=width,
            height=height,
            duration=n_frames / fps,
            n_frames=n_frames,
            codec="unknown"
        )

        return VideoData(
            metadata=metadata,
            frames=frames,
            timestamps=timestamps,
            optical_flow=None
        )

    except ImportError:
        return None
    except Exception:
        return None
