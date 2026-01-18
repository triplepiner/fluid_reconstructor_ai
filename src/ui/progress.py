"""
Progress display for pipeline execution.

Provides visual feedback during long-running operations.
"""

import sys
import time
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class StageProgress:
    """Progress information for a single stage."""
    name: str
    description: str
    current: int
    total: int
    started_at: float
    substages: List[str] = None


class ProgressDisplay:
    """
    Display progress information in terminal.
    """

    def __init__(self, use_rich: bool = True):
        """
        Initialize progress display.

        Args:
            use_rich: Whether to use rich library for fancy output
        """
        self.use_rich = use_rich and self._check_rich_available()
        self.current_stage: Optional[StageProgress] = None

        if self.use_rich:
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )

    def _check_rich_available(self) -> bool:
        """Check if rich library is available."""
        try:
            import rich
            return True
        except ImportError:
            return False

    def start_stage(self, name: str, description: str, total: int = 100):
        """
        Start a new stage.

        Args:
            name: Stage name
            description: Stage description
            total: Total steps in stage
        """
        self.current_stage = StageProgress(
            name=name,
            description=description,
            current=0,
            total=total,
            started_at=time.time()
        )

        if self.use_rich:
            self.task_id = self.progress.add_task(description, total=total)
            self.progress.start()
        else:
            print(f"\n[{name}] {description}")
            self._print_bar(0, total)

    def update(self, current: int, message: Optional[str] = None):
        """
        Update progress.

        Args:
            current: Current step
            message: Optional status message
        """
        if self.current_stage is None:
            return

        self.current_stage.current = current

        if self.use_rich:
            self.progress.update(
                self.task_id,
                completed=current,
                description=message or self.current_stage.description
            )
        else:
            self._print_bar(current, self.current_stage.total)
            if message:
                print(f"  {message}")

    def advance(self, amount: int = 1, message: Optional[str] = None):
        """
        Advance progress by amount.

        Args:
            amount: Steps to advance
            message: Optional status message
        """
        if self.current_stage is None:
            return

        new_current = min(
            self.current_stage.current + amount,
            self.current_stage.total
        )
        self.update(new_current, message)

    def finish_stage(self, success: bool = True):
        """
        Finish current stage.

        Args:
            success: Whether stage completed successfully
        """
        if self.current_stage is None:
            return

        elapsed = time.time() - self.current_stage.started_at
        status = "completed" if success else "FAILED"

        if self.use_rich:
            self.progress.stop()
            self.console.print(
                f"  [{status}] in {elapsed:.1f}s",
                style="green" if success else "red"
            )
        else:
            print(f"  [{status}] in {elapsed:.1f}s")

        self.current_stage = None

    def _print_bar(self, current: int, total: int, width: int = 40):
        """Print ASCII progress bar."""
        filled = int(width * current / max(total, 1))
        bar = "█" * filled + "░" * (width - filled)
        percentage = 100 * current / max(total, 1)
        sys.stdout.write(f"\r  [{bar}] {percentage:5.1f}%")
        sys.stdout.flush()
        if current >= total:
            print()

    def print_metrics(self, metrics: dict):
        """
        Print metrics in a formatted way.

        Args:
            metrics: Dictionary of metric names to values
        """
        if self.use_rich:
            from rich.table import Table
            table = Table(title="Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for name, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(name, f"{value:.6f}")
                else:
                    table.add_row(name, str(value))

            self.console.print(table)
        else:
            print("\nMetrics:")
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.6f}")
                else:
                    print(f"  {name}: {value}")


class PipelineProgressTracker:
    """
    Track progress across all pipeline stages.
    """

    STAGE_NAMES = [
        ("video_loading", "Loading videos"),
        ("optical_flow", "Computing optical flow"),
        ("temporal_sync", "Synchronizing timelines"),
        ("feature_extraction", "Extracting features"),
        ("camera_calibration", "Calibrating cameras"),
        ("triangulation", "Triangulating points"),
        ("gaussian_init", "Initializing Gaussians"),
        ("reconstruction", "Optimizing reconstruction"),
        ("physics_estimation", "Estimating physics"),
        ("output", "Saving results")
    ]

    def __init__(self):
        """Initialize pipeline tracker."""
        self.display = ProgressDisplay()
        self.current_stage_idx = -1
        self.stage_results = {}

    def start_pipeline(self, n_stages: Optional[int] = None):
        """Start tracking pipeline progress."""
        n = n_stages or len(self.STAGE_NAMES)
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION")
        print("=" * 60)
        print(f"\nTotal stages: {n}")

    def start_stage(self, stage_idx: int):
        """Start a stage."""
        if stage_idx < len(self.STAGE_NAMES):
            name, desc = self.STAGE_NAMES[stage_idx]
        else:
            name, desc = f"stage_{stage_idx}", "Processing"

        self.current_stage_idx = stage_idx
        print(f"\n[Stage {stage_idx + 1}/{len(self.STAGE_NAMES)}] {desc}")
        self.stage_start_time = time.time()

    def finish_stage(self, success: bool = True, metrics: Optional[dict] = None):
        """Finish current stage."""
        elapsed = time.time() - self.stage_start_time
        status = "OK" if success else "FAILED"

        print(f"  Status: {status} ({elapsed:.1f}s)")

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        self.stage_results[self.current_stage_idx] = {
            'success': success,
            'duration': elapsed,
            'metrics': metrics
        }

    def finish_pipeline(self, success: bool = True):
        """Finish pipeline tracking."""
        total_duration = sum(
            r.get('duration', 0)
            for r in self.stage_results.values()
        )

        print("\n" + "=" * 60)
        print(f"PIPELINE {'COMPLETED' if success else 'FAILED'}")
        print("=" * 60)
        print(f"\nTotal time: {total_duration:.1f}s")
        print(f"Stages completed: {len(self.stage_results)}/{len(self.STAGE_NAMES)}")


def create_training_callback(display: ProgressDisplay):
    """
    Create a callback for training loop progress.

    Args:
        display: ProgressDisplay instance

    Returns:
        Callback function
    """
    def callback(epoch: int, total_epochs: int, loss: float, metrics: dict = None):
        display.update(
            epoch,
            f"Epoch {epoch}/{total_epochs} | Loss: {loss:.6f}"
        )

        if metrics and epoch % 100 == 0:
            display.print_metrics(metrics)

    return callback
