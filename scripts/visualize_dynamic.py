#!/usr/bin/env python3
"""
Enhanced visualization for Dynamic Gaussian Splatting output.

Features:
- Dynamic video animation showing fluid motion over time
- Viser interactive 3D viewer
- Proper handling of DynamicGaussianCloud
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gaussian_splatting.gaussian import GaussianCloud
from src.gaussian_splatting.dynamic_gaussians import DynamicGaussianCloud
from src.gaussian_splatting.rasterizer import GaussianRasterizer
from src.config import Camera, CameraIntrinsics, CameraExtrinsics


def load_dynamic_gaussians(input_path: Path, n_timesteps: int = 30, device: str = "cpu"):
    """
    Load DynamicGaussianCloud from saved state dict.
    """
    print(f"Loading from {input_path}...")
    state_dict = torch.load(input_path, map_location=device, weights_only=False)

    # Check if it's a DynamicGaussianCloud
    if 'base_gaussians._positions' in state_dict:
        n_gaussians = state_dict['base_gaussians._positions'].shape[0]
        print(f"  Found DynamicGaussianCloud with {n_gaussians:,} Gaussians")

        # Create base GaussianCloud
        base = GaussianCloud(n_gaussians=n_gaussians, sh_degree=3, device=device)

        # Load base gaussian parameters
        base_state = {}
        for k, v in state_dict.items():
            if k.startswith('base_gaussians.'):
                new_key = k.replace('base_gaussians.', '')
                base_state[new_key] = v
        base.load_state_dict(base_state)

        # Determine temporal mode from state dict
        if 'velocity_mlp.0.weight' in state_dict:
            temporal_mode = "velocity"
        elif 'position_offsets' in state_dict:
            temporal_mode = "trajectory"
        else:
            temporal_mode = "per_frame"

        print(f"  Temporal mode: {temporal_mode}")

        # Create DynamicGaussianCloud
        dynamic = DynamicGaussianCloud(
            base,
            n_timesteps=n_timesteps,
            temporal_mode=temporal_mode,
            device=device
        )

        # Load full state
        dynamic.load_state_dict(state_dict, strict=False)

        return dynamic, True
    else:
        # Static GaussianCloud
        n_gaussians = state_dict['_positions'].shape[0]
        print(f"  Found static GaussianCloud with {n_gaussians:,} Gaussians")

        gaussians = GaussianCloud(n_gaussians=n_gaussians, device=device)
        gaussians.load_state_dict(state_dict)

        return gaussians, False


def create_orbit_camera(
    center: torch.Tensor,
    radius: float,
    angle: float,
    elevation: float,
    resolution: int,
    fov: float = 60.0
) -> Camera:
    """Create a camera orbiting around a center point."""
    # Camera position
    cam_x = center[0].item() + radius * np.cos(angle) * np.cos(elevation)
    cam_y = center[1].item() + radius * np.sin(elevation)
    cam_z = center[2].item() + radius * np.sin(angle) * np.cos(elevation)
    cam_pos = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32)

    # Look at center
    forward = center - cam_pos
    forward = forward / forward.norm()

    # Up vector
    up = torch.tensor([0., 1., 0.], dtype=torch.float32)

    # Right vector
    right = torch.cross(forward, up)
    if right.norm() < 1e-6:
        up = torch.tensor([0., 0., 1.], dtype=torch.float32)
        right = torch.cross(forward, up)
    right = right / right.norm()

    # Recompute up
    up = torch.cross(right, forward)
    up = up / up.norm()

    # Rotation matrix (camera to world -> world to camera)
    R_c2w = torch.stack([right, up, -forward], dim=1)
    R = R_c2w.T
    t = -R @ cam_pos

    # Intrinsics
    focal = resolution / (2 * np.tan(np.radians(fov / 2)))
    intrinsics = CameraIntrinsics(fx=focal, fy=focal, cx=resolution/2, cy=resolution/2)
    extrinsics = CameraExtrinsics(R=R, t=t)

    return Camera(intrinsics=intrinsics, extrinsics=extrinsics, width=resolution, height=resolution)


def print_gaussian_stats(gaussians):
    """Print statistics about Gaussians."""
    if hasattr(gaussians, 'base_gaussians'):
        g = gaussians.base_gaussians
    else:
        g = gaussians

    positions = g.positions.detach()
    opacities = g.opacities.detach()
    scales = g.scales.detach()

    print("\n=== Gaussian Statistics ===")
    print(f"Count: {g.n_gaussians:,}")
    print(f"\nPositions:")
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    print(f"  Center: ({positions.mean(dim=0)[0]:.2f}, {positions.mean(dim=0)[1]:.2f}, {positions.mean(dim=0)[2]:.2f})")

    print(f"\nOpacities (after sigmoid): [{opacities.min():.4f}, {opacities.max():.4f}], mean={opacities.mean():.4f}")
    print(f"Scales (after exp): [{scales.min():.6f}, {scales.max():.6f}], mean={scales.mean():.6f}")

    # Count visible Gaussians
    visible = (opacities > 0.01).sum().item()
    print(f"Visible Gaussians (opacity > 0.01): {visible:,} ({100*visible/g.n_gaussians:.1f}%)")


def render_frame(rasterizer, gaussians, camera, is_dynamic: bool, timestep: float = 0):
    """Render a single frame."""
    if is_dynamic:
        g = gaussians.get_gaussians_at_time(timestep)
    else:
        g = gaussians

    with torch.no_grad():
        result = rasterizer(g, camera, return_depth=True, return_alpha=True)

    return result


def create_dynamic_video(
    gaussians,
    is_dynamic: bool,
    output_path: Path,
    n_frames: int = 60,
    n_timesteps: int = 30,
    resolution: int = 512,
    orbit: bool = True
):
    """
    Create video showing dynamic fluid animation.

    If orbit=True: Camera rotates around the scene
    If orbit=False: Camera is fixed, showing fluid motion over time
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return False

    device = "cpu"

    # Get scene bounds
    if hasattr(gaussians, 'base_gaussians'):
        positions = gaussians.base_gaussians.positions.detach()
    else:
        positions = gaussians.positions.detach()

    center = positions.mean(dim=0)
    extent = (positions.max(dim=0).values - positions.min(dim=0).values).max().item()
    radius = extent * 2.0

    print(f"\nScene center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"Scene extent: {extent:.2f}, camera radius: {radius:.2f}")

    # Create rasterizer
    rasterizer = GaussianRasterizer(
        image_height=resolution,
        image_width=resolution,
        use_gsplat=True,
        background_color=(0.1, 0.1, 0.15),  # Dark blue-gray background
        device=device
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, 30, (resolution, resolution))

    print(f"\nCreating video: {output_path}")
    print(f"  Frames: {n_frames}, Resolution: {resolution}x{resolution}")
    print(f"  Mode: {'Orbit + Time' if orbit else 'Fixed camera + Time'}")

    for i in range(n_frames):
        # Calculate timestep for fluid animation
        if is_dynamic:
            timestep = (i / n_frames) * (n_timesteps - 1)
        else:
            timestep = 0

        # Calculate camera angle
        if orbit:
            angle = 2 * np.pi * i / n_frames
        else:
            angle = np.pi / 4  # Fixed 45-degree angle

        elevation = 0.3  # Slight elevation

        camera = create_orbit_camera(center, radius, angle, elevation, resolution)

        # Render
        result = render_frame(rasterizer, gaussians, camera, is_dynamic, timestep)
        image = result['image'].cpu().numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Add frame info overlay
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"t={timestep:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        video.write(frame_bgr)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames} (t={timestep:.1f})")

    video.release()
    print(f"  Saved to {output_path}")
    return True


def create_multiview_grid(
    gaussians,
    is_dynamic: bool,
    output_path: Path,
    n_views: int = 8,
    resolution: int = 256,
    timestep: float = 0
):
    """Create a grid of views from different angles."""
    import matplotlib.pyplot as plt

    device = "cpu"

    # Get scene bounds
    if hasattr(gaussians, 'base_gaussians'):
        positions = gaussians.base_gaussians.positions.detach()
    else:
        positions = gaussians.positions.detach()

    center = positions.mean(dim=0)
    extent = (positions.max(dim=0).values - positions.min(dim=0).values).max().item()
    radius = extent * 2.0

    rasterizer = GaussianRasterizer(
        image_height=resolution,
        image_width=resolution,
        use_gsplat=True,
        background_color=(0.1, 0.1, 0.15),
        device=device
    )

    # Render from multiple angles
    images = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        elevation = 0.3

        camera = create_orbit_camera(center, radius, angle, elevation, resolution)
        result = render_frame(rasterizer, gaussians, camera, is_dynamic, timestep)
        image = result['image'].cpu().numpy()
        image = np.clip(image, 0, 1)
        images.append(image)

    # Create grid
    n_cols = 4
    n_rows = (n_views + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_views > 1 else [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.set_title(f"View {i} ({int(np.degrees(2 * np.pi * i / n_views))}°)")
        ax.axis('off')

    # Hide empty axes
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved multiview grid to {output_path}")


def run_viser_viewer(gaussians, is_dynamic: bool, n_timesteps: int = 30, port: int = 8080):
    """
    Run interactive Viser viewer for the Gaussian cloud.
    """
    try:
        import viser
        import viser.transforms as tf
    except ImportError:
        print("\nViser not installed. Install with: pip install viser")
        print("Then run this script again with --viser flag")
        return False

    print(f"\n=== Starting Viser Interactive Viewer ===")
    print(f"Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop\n")

    server = viser.ViserServer(host="0.0.0.0", port=port)

    # Get positions and colors
    if hasattr(gaussians, 'base_gaussians'):
        g = gaussians.base_gaussians
    else:
        g = gaussians

    positions = g.positions.detach().cpu().numpy()
    opacities = g.opacities.detach().cpu().numpy().squeeze()

    # Get colors from SH DC component
    SH_C0 = 0.28209479177387814
    colors_sh = g._features_dc.detach().cpu().numpy()
    colors = np.clip(SH_C0 * colors_sh + 0.5, 0, 1)

    # Filter by opacity for better visualization
    visible_mask = opacities > 0.05
    vis_positions = positions[visible_mask]
    vis_colors = colors[visible_mask]
    vis_opacities = opacities[visible_mask]

    print(f"Displaying {vis_positions.shape[0]:,} visible Gaussians (opacity > 0.05)")

    # Add point cloud
    server.scene.add_point_cloud(
        "/gaussians",
        points=vis_positions,
        colors=(vis_colors * 255).astype(np.uint8),
        point_size=0.02,
        point_shape="circle"
    )

    # Add axes helper
    server.scene.add_frame("/world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Add time slider if dynamic
    if is_dynamic:
        time_slider = server.gui.add_slider(
            "Time", min=0, max=n_timesteps - 1, step=1, initial_value=0
        )

        @time_slider.on_update
        def _on_time_change(event):
            t = time_slider.value
            g_t = gaussians.get_gaussians_at_time(float(t))

            new_positions = g_t.positions.detach().cpu().numpy()
            new_positions = new_positions[visible_mask]

            server.scene.add_point_cloud(
                "/gaussians",
                points=new_positions,
                colors=(vis_colors * 255).astype(np.uint8),
                point_size=0.02,
                point_shape="circle"
            )

    # Add controls
    server.gui.add_markdown("### Controls")
    server.gui.add_markdown("- **Left-click + drag**: Rotate view")
    server.gui.add_markdown("- **Right-click + drag**: Pan")
    server.gui.add_markdown("- **Scroll**: Zoom")

    # Keep server running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping Viser server...")

    return True


def export_ply_sequence(
    gaussians,
    is_dynamic: bool,
    output_dir: Path,
    n_timesteps: int = 30,
    opacity_threshold: float = 0.01
):
    """Export PLY files for each timestep (for Blender animation)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting PLY sequence to {output_dir}/")

    SH_C0 = 0.28209479177387814

    for t in range(n_timesteps):
        if is_dynamic:
            g = gaussians.get_gaussians_at_time(float(t))
        else:
            g = gaussians if not hasattr(gaussians, 'base_gaussians') else gaussians.base_gaussians

        positions = g.positions.detach().cpu().numpy()
        opacities = g.opacities.detach().cpu().numpy().squeeze()
        colors_sh = g._features_dc.detach().cpu().numpy()
        colors = np.clip(SH_C0 * colors_sh + 0.5, 0, 1)

        # Filter by opacity
        mask = opacities > opacity_threshold
        positions = positions[mask]
        colors = colors[mask]

        # Write PLY
        ply_path = output_dir / f"frame_{t:04d}.ply"
        n = len(positions)

        header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

        with open(ply_path, 'w') as f:
            f.write(header)
            colors_uint8 = (colors * 255).astype(np.uint8)
            for i in range(n):
                f.write(f"{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} ")
                f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")

        if (t + 1) % 10 == 0:
            print(f"  Exported frame {t+1}/{n_timesteps} ({n:,} points)")

    print(f"  Done! Import sequence in Blender for animation.")


def main():
    parser = argparse.ArgumentParser(description="Visualize Dynamic Gaussian Splatting output")
    parser.add_argument("input", type=str, nargs="?", default="outputs/final_gaussians.pt",
                        help="Path to final_gaussians.pt")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/visualizations",
                        help="Output directory")
    parser.add_argument("--timesteps", "-t", type=int, default=30,
                        help="Number of timesteps for animation")
    parser.add_argument("--resolution", "-r", type=int, default=512,
                        help="Render resolution")
    parser.add_argument("--n-frames", type=int, default=90,
                        help="Number of video frames")

    # Output options
    parser.add_argument("--video", action="store_true", help="Create orbit video")
    parser.add_argument("--video-fixed", action="store_true", help="Create fixed-camera video")
    parser.add_argument("--grid", action="store_true", help="Create multiview grid")
    parser.add_argument("--ply-sequence", action="store_true", help="Export PLY sequence")
    parser.add_argument("--viser", action="store_true", help="Launch Viser interactive viewer")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--all", action="store_true", help="Generate all outputs (except Viser)")

    args = parser.parse_args()

    if args.all:
        args.video = args.grid = True

    # Default to video if nothing specified
    if not any([args.video, args.video_fixed, args.grid, args.ply_sequence, args.viser]):
        args.video = True
        args.grid = True

    # Load gaussians
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    gaussians, is_dynamic = load_dynamic_gaussians(input_path, args.timesteps)
    print_gaussian_stats(gaussians)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    if args.grid:
        create_multiview_grid(
            gaussians, is_dynamic,
            output_dir / "multiview_grid.png",
            n_views=8,
            resolution=args.resolution
        )

    if args.video:
        create_dynamic_video(
            gaussians, is_dynamic,
            output_dir / "orbit_animation.mp4",
            n_frames=args.n_frames,
            n_timesteps=args.timesteps,
            resolution=args.resolution,
            orbit=True
        )

    if args.video_fixed:
        create_dynamic_video(
            gaussians, is_dynamic,
            output_dir / "fluid_animation.mp4",
            n_frames=args.n_frames,
            n_timesteps=args.timesteps,
            resolution=args.resolution,
            orbit=False
        )

    if args.ply_sequence:
        export_ply_sequence(
            gaussians, is_dynamic,
            output_dir / "ply_sequence",
            n_timesteps=args.timesteps
        )

    if args.viser:
        run_viser_viewer(gaussians, is_dynamic, args.timesteps, args.port)

    print(f"\nAll outputs saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
