"""
Pipeline orchestrator for fluid reconstruction.

Manages the execution of all pipeline stages.
"""

import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import time
import json
from tqdm import tqdm

from ..config import PipelineConfig
from .stages import (
    PipelineStage,
    StageStatus,
    StageResult,
    VideoLoadingStage,
    VideoStabilizationStage,
    OpticalFlowStage,
    FluidSegmentationStage,
    TemporalSyncStage,
    FeatureExtractionStage,
    CameraCalibrationStage,
    TriangulationStage,
    GaussianInitStage,
    ReconstructionStage,
    PhysicsEstimationStage,
    OutputStage
)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool
    stage_results: Dict[str, StageResult]
    total_duration: float
    outputs: Dict[str, Any]


class FluidReconstructionPipeline:
    """
    Main pipeline for physics-integrated fluid reconstruction.

    Orchestrates all stages from video input to physics estimation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, enable_physics: bool = True):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses default if None)
            enable_physics: Whether to run physics estimation stage
        """
        self.config = config or PipelineConfig()
        self.enable_physics = enable_physics

        # Initialize stages in order
        self.stages: List[PipelineStage] = [
            VideoLoadingStage(self.config),
            VideoStabilizationStage(self.config),  # Handle camera shake
            OpticalFlowStage(self.config),
            FluidSegmentationStage(self.config),  # Isolate fluid regions
            TemporalSyncStage(self.config),
            FeatureExtractionStage(self.config),
            CameraCalibrationStage(self.config),
            TriangulationStage(self.config),
            GaussianInitStage(self.config),
            ReconstructionStage(self.config),
        ]

        # Add physics stage if enabled
        if enable_physics:
            self.stages.append(PhysicsEstimationStage(self.config))

        # Always add output stage
        self.stages.append(OutputStage(self.config))

        self._current_stage_idx = 0
        self._data: Dict[str, Any] = {}

    def run(
        self,
        video_paths: List[Path],
        start_stage: int = 0,
        end_stage: Optional[int] = None
    ) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            video_paths: Paths to input video files
            start_stage: Stage index to start from
            end_stage: Stage index to stop at (inclusive)

        Returns:
            PipelineResult with all outputs
        """
        if end_stage is None:
            end_stage = len(self.stages) - 1

        print(f"\n{'='*60}")
        print("FLUID RECONSTRUCTION PIPELINE")
        print(f"{'='*60}")
        print(f"Input: {len(video_paths)} video(s)")
        print(f"Stages: {start_stage} to {end_stage}")
        print(f"Device: {self.config.device}")
        print(f"{'='*60}\n")

        # Initialize data with video paths
        self._data = {'video_paths': video_paths}

        stage_results = {}
        total_start = time.time()
        success = True

        # Create overall progress bar
        stage_names = [self.stages[i].name for i in range(start_stage, end_stage + 1)]
        pbar = tqdm(
            range(start_stage, end_stage + 1),
            desc="Pipeline Progress",
            unit="stage",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]'
        )

        for i in pbar:
            stage = self.stages[i]
            self._current_stage_idx = i

            # Update progress bar description
            pbar.set_description(f"Stage: {stage.name}")

            print(f"\n{'─'*60}")
            print(f"[{i}/{len(self.stages)-1}] {stage.name.upper()}")
            print(f"    {stage.description}")
            print(f"{'─'*60}")

            try:
                result = stage.run(self._data)
                self._data = result.output if isinstance(result.output, dict) else self._data

                stage_results[stage.name] = result
                print(f"    Completed in {result.duration_seconds:.1f}s")

                # Save checkpoint
                self._save_stage_checkpoint(i, stage.name)

            except Exception as e:
                import traceback
                print(f"\n    FAILED: {e}")
                print(f"\n    Traceback:")
                traceback.print_exc()
                stage_results[stage.name] = StageResult(
                    status=StageStatus.FAILED,
                    error=str(e)
                )
                success = False
                pbar.set_description(f"FAILED at: {stage.name}")
                break

        pbar.close()
        total_duration = time.time() - total_start

        print(f"\n{'='*60}")
        if success:
            print("PIPELINE COMPLETED SUCCESSFULLY")
        else:
            print("PIPELINE FAILED")
        print(f"Total time: {total_duration:.1f}s")
        print(f"{'='*60}\n")

        return PipelineResult(
            success=success,
            stage_results=stage_results,
            total_duration=total_duration,
            outputs=self._data
        )

    def resume(self, checkpoint_dir: Path) -> PipelineResult:
        """
        Resume pipeline from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            PipelineResult
        """
        # Find latest checkpoint
        checkpoint_file = checkpoint_dir / "pipeline_state.json"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")

        with open(checkpoint_file) as f:
            state = json.load(f)

        # Load data
        data_file = checkpoint_dir / "pipeline_data.pt"
        if data_file.exists():
            self._data = torch.load(data_file)

        # Resume from next stage
        last_stage = state['last_completed_stage']
        return self.run(
            self._data['video_paths'],
            start_stage=last_stage + 1
        )

    def _save_stage_checkpoint(self, stage_idx: int, stage_name: str):
        """Save checkpoint after stage completion."""
        checkpoint_dir = self.config.checkpoint_dir

        # Save state
        state = {
            'last_completed_stage': stage_idx,
            'stage_name': stage_name,
            'timestamp': time.time()
        }
        with open(checkpoint_dir / "pipeline_state.json", 'w') as f:
            json.dump(state, f)

        # Save data (careful with large tensors)
        # Only save essential data
        essential_data = {
            'video_paths': self._data.get('video_paths'),
            'sync_params': self._data.get('sync_params')
        }
        torch.save(essential_data, checkpoint_dir / "pipeline_data.pt")


def run_pipeline(
    video_paths: List[str],
    output_dir: str = "outputs",
    device: str = "cuda",
    **config_overrides
) -> PipelineResult:
    """
    Convenience function to run the pipeline.

    Args:
        video_paths: Paths to input videos
        output_dir: Output directory
        device: Device to use
        **config_overrides: Additional config overrides

    Returns:
        PipelineResult
    """
    # Create config
    config = PipelineConfig(
        output_dir=Path(output_dir),
        checkpoint_dir=Path(output_dir) / "checkpoints",
        device=device
    )

    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create and run pipeline
    pipeline = FluidReconstructionPipeline(config)
    return pipeline.run([Path(p) for p in video_paths])


def quick_test(video_path: str, n_epochs: int = 100) -> PipelineResult:
    """
    Quick test run with reduced settings.

    Args:
        video_path: Single video path (will be duplicated for 3 views)
        n_epochs: Number of training epochs

    Returns:
        PipelineResult
    """
    # Use same video 3 times for testing
    video_paths = [video_path] * 3

    config = PipelineConfig(
        max_resolution=480,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    config.optimization.n_epochs = n_epochs
    config.log_interval = 10

    pipeline = FluidReconstructionPipeline(config)
    return pipeline.run([Path(p) for p in video_paths])
