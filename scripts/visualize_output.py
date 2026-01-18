#!/usr/bin/env python3
"""
Visualization and export script for Gaussian Splatting output.

Exports to:
- PLY point cloud (viewable in MeshLab, CloudCompare, Blender)
- Rendered images from multiple viewpoints
- Video flythrough
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import struct

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gaussian_splatting.gaussian import GaussianCloud
from src.gaussian_splatting.rasterizer import GaussianRasterizer, create_camera_from_matrices


def export_to_ply(gaussians: GaussianCloud, output_path: Path):
    """
    Export Gaussians to PLY format.

    The PLY file contains:
    - Position (x, y, z)
    - Color (r, g, b) from DC spherical harmonics
    - Scale (sx, sy, sz)
    - Opacity
    """
    n = gaussians.n_gaussians

    # Get data as numpy
    positions = gaussians.positions.detach().cpu().numpy()
    scales = gaussians.scales.detach().cpu().numpy()
    opacities = gaussians.opacities.detach().cpu().numpy().squeeze()

    # Get colors from DC component of SH (first coefficient)
    # SH to RGB: color = SH_C0 * dc + 0.5 (where SH_C0 = 0.28209479177387814)
    SH_C0 = 0.28209479177387814
    colors_sh = gaussians._features_dc.detach().cpu().numpy()
    colors = np.clip(SH_C0 * colors_sh + 0.5, 0, 1)
    colors_uint8 = (colors * 255).astype(np.uint8)

    # Write PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float scale_x
property float scale_y
property float scale_z
property float opacity
end_header
"""

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))

        for i in range(n):
            # Position
            f.write(struct.pack('<fff', *positions[i]))
            # Color
            f.write(struct.pack('<BBB', *colors_uint8[i]))
            # Scale
            f.write(struct.pack('<fff', *scales[i]))
            # Opacity
            f.write(struct.pack('<f', opacities[i]))

    print(f"Exported {n} Gaussians to {output_path}")


def export_to_simple_ply(gaussians: GaussianCloud, output_path: Path, opacity_threshold: float = 0.01):
    """
    Export to simple PLY (just colored points) for maximum compatibility.
    """
    # Filter by opacity
    opacities = gaussians.opacities.detach().cpu().numpy().squeeze()
    mask = opacities > opacity_threshold

    positions = gaussians.positions.detach().cpu().numpy()[mask]

    # Get colors
    SH_C0 = 0.28209479177387814
    colors_sh = gaussians._features_dc.detach().cpu().numpy()[mask]
    colors = np.clip(SH_C0 * colors_sh + 0.5, 0, 1)
    colors_uint8 = (colors * 255).astype(np.uint8)

    n = positions.shape[0]

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

    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(n):
            f.write(f"{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} ")
            f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")

    print(f"Exported {n} points (opacity > {opacity_threshold}) to {output_path}")


def render_views(gaussians: GaussianCloud, output_dir: Path, n_views: int = 8, resolution: int = 512):
    """
    Render the Gaussians from multiple viewpoints around the scene.
    """
    import matplotlib.pyplot as plt

    device = "cpu"  # Use CPU for compatibility
    gaussians = gaussians.to(device)

    # Get scene bounds
    positions = gaussians.positions.detach()
    center = positions.mean(dim=0)
    extent = (positions.max(dim=0).values - positions.min(dim=0).values).max().item()
    radius = extent * 1.5

    # Create rasterizer
    rasterizer = GaussianRasterizer(
        image_height=resolution,
        image_width=resolution,
        use_gsplat=False,
        device=device
    )

    # Create cameras around the scene
    from src.config import Camera, CameraIntrinsics, CameraExtrinsics

    # Focal length for 60 degree FOV
    fov = 60
    focal = resolution / (2 * np.tan(np.radians(fov / 2)))

    intrinsics = CameraIntrinsics(
        fx=focal, fy=focal,
        cx=resolution / 2, cy=resolution / 2
    )

    images = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views

        # Camera position on a circle around the center
        cam_pos = torch.tensor([
            center[0].item() + radius * np.cos(angle),
            center[1].item() + radius * 0.3,  # Slightly above
            center[2].item() + radius * np.sin(angle)
        ], dtype=torch.float32)

        # Look at center
        forward = center - cam_pos
        forward = forward / forward.norm()

        # Up vector
        up = torch.tensor([0., 1., 0.], dtype=torch.float32)

        # Right vector
        right = torch.cross(forward, up)
        right = right / right.norm()

        # Recompute up
        up = torch.cross(right, forward)

        # Rotation matrix (camera to world)
        R_c2w = torch.stack([right, up, -forward], dim=1)  # (3, 3)
        R = R_c2w.T  # world to camera
        t = -R @ cam_pos

        extrinsics = CameraExtrinsics(R=R, t=t)
        camera = Camera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            width=resolution,
            height=resolution
        )

        # Render
        with torch.no_grad():
            result = rasterizer(gaussians, camera)
            image = result['image'].cpu().numpy()
            image = np.clip(image, 0, 1)
            images.append(image)

        # Save individual image
        plt.imsave(output_dir / f"view_{i:02d}.png", image)
        print(f"Rendered view {i+1}/{n_views}")

    # Create a grid of all views
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img)
        ax.set_title(f"View {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "views_grid.png", dpi=150)
    plt.close()

    print(f"Saved view grid to {output_dir / 'views_grid.png'}")

    return images


def create_video(gaussians: GaussianCloud, output_path: Path, n_frames: int = 60, resolution: int = 512):
    """
    Create a video flythrough around the scene.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Skipping video creation.")
        print("Install with: pip install opencv-python")
        return

    device = "cpu"
    gaussians = gaussians.to(device)

    # Get scene bounds
    positions = gaussians.positions.detach()
    center = positions.mean(dim=0)
    extent = (positions.max(dim=0).values - positions.min(dim=0).values).max().item()
    radius = extent * 1.5

    # Create rasterizer
    rasterizer = GaussianRasterizer(
        image_height=resolution,
        image_width=resolution,
        use_gsplat=False,
        device=device
    )

    from src.config import Camera, CameraIntrinsics, CameraExtrinsics

    fov = 60
    focal = resolution / (2 * np.tan(np.radians(fov / 2)))
    intrinsics = CameraIntrinsics(fx=focal, fy=focal, cx=resolution/2, cy=resolution/2)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, 30, (resolution, resolution))

    print(f"Creating video with {n_frames} frames...")

    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames

        cam_pos = torch.tensor([
            center[0].item() + radius * np.cos(angle),
            center[1].item() + radius * 0.3,
            center[2].item() + radius * np.sin(angle)
        ], dtype=torch.float32)

        forward = center - cam_pos
        forward = forward / forward.norm()
        up = torch.tensor([0., 1., 0.], dtype=torch.float32)
        right = torch.cross(forward, up)
        right = right / right.norm()
        up = torch.cross(right, forward)

        R_c2w = torch.stack([right, up, -forward], dim=1)
        R = R_c2w.T
        t = -R @ cam_pos

        extrinsics = CameraExtrinsics(R=R, t=t)
        camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics, width=resolution, height=resolution)

        with torch.no_grad():
            result = rasterizer(gaussians, camera)
            image = result['image'].cpu().numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image_bgr)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    video.release()
    print(f"Saved video to {output_path}")


def print_stats(gaussians: GaussianCloud):
    """Print statistics about the Gaussian cloud."""
    print("\n=== Gaussian Cloud Statistics ===")
    print(f"Number of Gaussians: {gaussians.n_gaussians:,}")

    positions = gaussians.positions.detach()
    print(f"\nPosition bounds:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    scales = gaussians.scales.detach()
    print(f"\nScale range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"Mean scale: {scales.mean():.6f}")

    opacities = gaussians.opacities.detach()
    print(f"\nOpacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    print(f"Mean opacity: {opacities.mean():.3f}")
    print(f"Gaussians with opacity > 0.5: {(opacities > 0.5).sum().item():,}")
    print(f"Gaussians with opacity > 0.1: {(opacities > 0.1).sum().item():,}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Gaussian Splatting output")
    parser.add_argument("input", type=str, nargs="?", default="outputs/final_gaussians.pt",
                        help="Path to final_gaussians.pt")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--ply", action="store_true", help="Export to PLY format")
    parser.add_argument("--simple-ply", action="store_true", help="Export to simple PLY (points only)")
    parser.add_argument("--render", action="store_true", help="Render views")
    parser.add_argument("--video", action="store_true", help="Create flythrough video")
    parser.add_argument("--all", action="store_true", help="Do all exports")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    parser.add_argument("--n-views", type=int, default=8, help="Number of views to render")
    parser.add_argument("--n-frames", type=int, default=60, help="Number of video frames")

    args = parser.parse_args()

    if args.all:
        args.ply = args.simple_ply = args.render = args.video = True

    # Default to simple PLY if nothing specified
    if not any([args.ply, args.simple_ply, args.render, args.video]):
        args.simple_ply = True
        args.render = True

    # Load Gaussians
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    print(f"Loading Gaussians from {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle both GaussianCloud and DynamicGaussianCloud state dicts
    if 'base_gaussians._positions' in state_dict:
        # DynamicGaussianCloud format - extract base_gaussians
        base_state = {}
        for k, v in state_dict.items():
            if k.startswith('base_gaussians.'):
                new_key = k.replace('base_gaussians.', '')
                base_state[new_key] = v
        state_dict = base_state

    # Determine number of Gaussians from state dict
    n_gaussians = state_dict['_positions'].shape[0]

    gaussians = GaussianCloud(n_gaussians=n_gaussians, device="cpu")
    gaussians.load_state_dict(state_dict)

    print_stats(gaussians)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export PLY
    if args.ply:
        export_to_ply(gaussians, output_dir / "gaussians.ply")

    if args.simple_ply:
        export_to_simple_ply(gaussians, output_dir / "gaussians_simple.ply")

    # Render views
    if args.render:
        render_views(gaussians, output_dir, n_views=args.n_views, resolution=args.resolution)

    # Create video
    if args.video:
        create_video(gaussians, output_dir / "flythrough.mp4",
                     n_frames=args.n_frames, resolution=args.resolution)

    print(f"\nAll outputs saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
