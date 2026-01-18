#!/usr/bin/env python3
"""
Main entry point for the fluid reconstruction pipeline.

Usage:
    python run_pipeline.py                    # Interactive mode
    python run_pipeline.py video1 video2 video3  # Direct mode
    python run_pipeline.py --help             # Show help
"""

import argparse
import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import PipelineConfig
from src.pipeline import FluidReconstructionPipeline
from src.ui import FileSelector, main_menu


def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon Mac with Metal Performance Shaders
        return "mps"
    else:
        return "cpu"


def get_device_info(device: str) -> str:
    """Get human-readable device info."""
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    elif device == "mps":
        return "Apple Metal (MPS)"
    else:
        return f"CPU ({platform.processor() or 'unknown'})"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Physics-Integrated Neural Fluid Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_pipeline.py

  Single video (uses monocular depth):
    python run_pipeline.py video.mp4

  Multi-view (best quality):
    python run_pipeline.py video1.mp4 video2.mp4 video3.mp4

  With custom output directory:
    python run_pipeline.py video.mp4 -o results/

  Quick test:
    python run_pipeline.py --test video.mp4
        """
    )

    parser.add_argument(
        'videos',
        nargs='*',
        help='Input video files (1-3 videos, more views = better quality)'
    )

    parser.add_argument(
        '-o', '--output',
        default='outputs',
        help='Output directory (default: outputs)'
    )

    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use: auto (detect best), cuda, mps (Apple Silicon), cpu (default: auto)'
    )

    parser.add_argument(
        '--max-resolution',
        type=int,
        default=1080,
        help='Maximum frame resolution (default: 1080)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Number of training epochs (default: 5000)'
    )

    parser.add_argument(
        '--n-gaussians',
        type=int,
        default=10000,
        help='Initial number of Gaussians (default: 10000, use fewer for faster testing)'
    )

    parser.add_argument(
        '--no-physics',
        action='store_true',
        help='Skip physics estimation stage'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint directory'
    )

    parser.add_argument(
        '--test',
        type=str,
        help='Run quick test with single video'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Force interactive mode'
    )

    return parser.parse_args()


def run_interactive():
    """Run in interactive mode."""
    action, config = main_menu()

    if action == 'quit':
        print("Goodbye!")
        return

    elif action == 'run':
        run_pipeline(config)

    elif action == 'resume':
        resume_pipeline(config['checkpoint_dir'])

    elif action == 'test':
        run_test(config['video_path'])


def run_pipeline(config: dict):
    """Run the full pipeline."""
    # Create pipeline config
    pipeline_config = PipelineConfig(
        output_dir=Path(config.get('output_dir', 'outputs')),
        device=config.get('device', 'cuda'),
        max_resolution=config.get('max_resolution', 1080),
        initial_n_gaussians=config.get('n_gaussians', 10000)
    )
    pipeline_config.optimization.n_epochs = config.get('n_epochs', 5000)

    # Create and run pipeline
    enable_physics = config.get('enable_physics', True)
    pipeline = FluidReconstructionPipeline(pipeline_config, enable_physics=enable_physics)

    video_paths = [Path(p) for p in config['video_paths']]
    result = pipeline.run(video_paths)

    # Print summary
    print_result_summary(result)


def resume_pipeline(checkpoint_dir: str):
    """Resume pipeline from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_path}")
        return

    pipeline = FluidReconstructionPipeline()
    result = pipeline.resume(checkpoint_path)

    print_result_summary(result)


def run_test(video_path: str):
    """Run quick test with single video."""
    from src.pipeline import quick_test

    video = Path(video_path)
    if not video.exists():
        print(f"Error: Video not found: {video}")
        return

    print(f"\nRunning quick test with: {video.name}")
    print("(Using same video for all 3 views - for testing only)\n")

    result = quick_test(str(video), n_epochs=100)
    print_result_summary(result)


def print_result_summary(result):
    """Print pipeline result summary."""
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)

    print(f"\nSuccess: {result.success}")
    print(f"Total duration: {result.total_duration:.1f}s")

    print("\nStage Results:")
    for stage_name, stage_result in result.stage_results.items():
        status = "OK" if stage_result.status.name == "COMPLETED" else stage_result.status.name
        duration = f"{stage_result.duration_seconds:.1f}s"
        print(f"  {stage_name}: {status} ({duration})")

    if result.success:
        output_dir = result.outputs.get('video_paths', [''])[0]
        if isinstance(output_dir, Path):
            output_dir = output_dir.parent
        print(f"\nOutputs saved to: outputs/")


def main():
    """Main entry point."""
    args = parse_args()

    # Check for test mode
    if args.test:
        run_test(args.test)
        return

    # Check for resume mode
    if args.resume:
        resume_pipeline(args.resume)
        return

    # Check for direct mode (videos provided)
    if args.videos and 1 <= len(args.videos) <= 3 and not args.interactive:
        n_videos = len(args.videos)
        if n_videos == 1:
            print(f"\nSingle video mode: Using monocular depth estimation")
            print("(For better quality, use 2-3 videos from different angles)\n")
        elif n_videos == 2:
            print(f"\nTwo video mode: Using stereo reconstruction\n")
        else:
            print(f"\nMulti-view mode: Using triangulation from 3 views\n")

        # Auto-detect device if needed
        device = args.device
        if device == 'auto':
            device = get_best_device()

        print(f"Device: {get_device_info(device)}")

        # Warn about slower CPU processing
        if device == 'cpu':
            print("Note: Running on CPU will be slower. Consider using --max-resolution 480 for faster processing.\n")

        config = {
            'video_paths': args.videos,
            'output_dir': args.output,
            'device': device,
            'max_resolution': args.max_resolution,
            'n_epochs': args.epochs,
            'n_gaussians': args.n_gaussians,
            'enable_physics': not args.no_physics
        }
        run_pipeline(config)
        return

    # Interactive mode
    if args.videos and len(args.videos) > 3:
        print(f"Warning: Maximum 3 videos supported, got {len(args.videos)}. Switching to interactive mode.")

    run_interactive()


if __name__ == "__main__":
    main()
