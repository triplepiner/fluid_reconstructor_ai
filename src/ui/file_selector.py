"""
File selection interface for video input.

Provides both GUI and terminal-based file selection.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import sys


class FileSelector:
    """
    Select video files for processing.

    Supports both GUI (tkinter) and terminal-based selection.
    """

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

    def __init__(self, use_gui: bool = True):
        """
        Initialize file selector.

        Args:
            use_gui: Whether to use GUI dialogs
        """
        self.use_gui = use_gui and self._check_gui_available()

    def _check_gui_available(self) -> bool:
        """Check if GUI is available."""
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            return True
        except:
            return False

    def select_videos(self, n_videos: int = 3) -> List[Path]:
        """
        Select video files.

        Args:
            n_videos: Number of videos to select

        Returns:
            List of selected video paths
        """
        if self.use_gui:
            return self._select_gui(n_videos)
        else:
            return self._select_terminal(n_videos)

    def _select_gui(self, n_videos: int) -> List[Path]:
        """Select videos using GUI dialog."""
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        filetypes = [
            ("Video files", " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS)),
            ("All files", "*.*")
        ]

        videos = []
        for i in range(n_videos):
            title = f"Select Video {i + 1} of {n_videos}"
            path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes
            )

            if not path:
                root.destroy()
                raise ValueError(f"Video selection cancelled at video {i + 1}")

            videos.append(Path(path))

        root.destroy()

        # Confirm selection
        self._print_selection(videos)

        return videos

    def _select_terminal(self, n_videos: int) -> List[Path]:
        """Select videos using terminal input."""
        print("\n" + "=" * 50)
        print("VIDEO FILE SELECTION")
        print("=" * 50)
        print(f"\nPlease enter paths to {n_videos} video files.")
        print(f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        print()

        videos = []
        for i in range(n_videos):
            while True:
                path_str = input(f"Video {i + 1}/{n_videos}: ").strip()

                if not path_str:
                    print("  Error: Path cannot be empty")
                    continue

                path = Path(path_str).expanduser().resolve()

                if not path.exists():
                    print(f"  Error: File not found: {path}")
                    continue

                if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    print(f"  Error: Unsupported format: {path.suffix}")
                    continue

                videos.append(path)
                print(f"  Added: {path.name}")
                break

        # Confirm selection
        self._print_selection(videos)

        confirm = input("\nProceed with these videos? [Y/n]: ").strip().lower()
        if confirm in ('n', 'no'):
            return self._select_terminal(n_videos)

        return videos

    def _print_selection(self, videos: List[Path]):
        """Print selected videos."""
        print("\n" + "-" * 50)
        print("SELECTED VIDEOS:")
        print("-" * 50)
        for i, v in enumerate(videos):
            print(f"  {i + 1}. {v.name}")
            print(f"     Path: {v}")


class ConfigurationMenu:
    """
    Terminal menu for configuration options.
    """

    def __init__(self):
        """Initialize configuration menu."""
        self.config = {}

    def run(self) -> dict:
        """
        Run configuration menu.

        Returns:
            Configuration dictionary
        """
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print("Press Enter to accept default values.\n")

        # Output directory
        default_output = "outputs"
        output = input(f"Output directory [{default_output}]: ").strip()
        self.config['output_dir'] = output if output else default_output

        # Device
        default_device = "cuda"
        device = input(f"Device (cuda/cpu) [{default_device}]: ").strip()
        self.config['device'] = device if device else default_device

        # Max resolution
        default_res = "1080"
        res = input(f"Max resolution [{default_res}]: ").strip()
        self.config['max_resolution'] = int(res) if res else int(default_res)

        # Training iterations
        default_epochs = "5000"
        epochs = input(f"Training iterations [{default_epochs}]: ").strip()
        self.config['n_epochs'] = int(epochs) if epochs else int(default_epochs)

        # Physics estimation
        physics = input("Enable physics estimation? [Y/n]: ").strip().lower()
        self.config['enable_physics'] = physics not in ('n', 'no')

        print("\n" + "-" * 50)
        print("CONFIGURATION SUMMARY:")
        print("-" * 50)
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        return self.config


def main_menu() -> Tuple[str, dict]:
    """
    Display main menu and get user choice.

    Returns:
        Tuple of (action, config)
    """
    print("\n" + "=" * 60)
    print("   PHYSICS-INTEGRATED FLUID RECONSTRUCTION")
    print("=" * 60)
    print()
    print("[1] Select videos and run pipeline")
    print("[2] Resume from checkpoint")
    print("[3] Run quick test")
    print("[4] View help")
    print("[q] Quit")
    print()

    choice = input("Enter choice: ").strip().lower()

    if choice == '1':
        selector = FileSelector()
        videos = selector.select_videos(3)

        menu = ConfigurationMenu()
        config = menu.run()
        config['video_paths'] = [str(v) for v in videos]

        return 'run', config

    elif choice == '2':
        checkpoint_dir = input("Checkpoint directory: ").strip()
        return 'resume', {'checkpoint_dir': checkpoint_dir}

    elif choice == '3':
        video_path = input("Video path for test: ").strip()
        return 'test', {'video_path': video_path}

    elif choice == '4':
        print_help()
        return main_menu()

    elif choice in ('q', 'quit', 'exit'):
        return 'quit', {}

    else:
        print("Invalid choice. Please try again.")
        return main_menu()


def print_help():
    """Print help information."""
    print("\n" + "=" * 60)
    print("HELP")
    print("=" * 60)
    print("""
This tool reconstructs 3D fluid dynamics from multi-view video.

REQUIREMENTS:
- 3 video files of the same fluid scene from different angles
- Videos can have different FPS, resolution, and start times
- GPU with at least 8GB VRAM recommended

WORKFLOW:
1. Select 3 video files
2. Configure options (or use defaults)
3. Wait for processing
4. View results in output directory

OUTPUT:
- 3D Gaussian representation of fluid
- Velocity, pressure, density fields
- Estimated viscosity
- Visualization videos

For more information, see the README.
""")
