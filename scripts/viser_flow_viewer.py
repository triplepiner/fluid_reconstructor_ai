#!/usr/bin/env python3
"""
Interactive Viser viewer for dynamic Gaussian fluid flow.
Shows animated fluid motion with playback controls.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gaussian_splatting.gaussian import GaussianCloud
from src.gaussian_splatting.dynamic_gaussians import DynamicGaussianCloud


def main():
    print("=" * 60)
    print("DYNAMIC GAUSSIAN FLUID FLOW VIEWER")
    print("=" * 60)

    # Load DynamicGaussianCloud
    print("\nLoading model...")
    input_path = Path("outputs/final_gaussians.pt")

    state = torch.load(input_path, map_location='cpu', weights_only=False)
    base_state = {k.replace('base_gaussians.', ''): v
                  for k, v in state.items() if k.startswith('base_gaussians.')}

    n_gaussians = base_state['_positions'].shape[0]

    base = GaussianCloud(n_gaussians=n_gaussians, sh_degree=3, device='cpu')
    base.load_state_dict(base_state)

    n_timesteps = 30
    dynamic = DynamicGaussianCloud(base, n_timesteps=n_timesteps,
                                   temporal_mode="velocity", device='cpu')
    dynamic.load_state_dict(state, strict=False)

    print(f"  Loaded {n_gaussians:,} Gaussians")
    print(f"  Timesteps: {n_timesteps}")

    # Get base colors
    SH_C0 = 0.28209479177387814
    colors_sh = base._features_dc.detach().numpy()
    base_colors = np.clip(SH_C0 * colors_sh + 0.5, 0, 1)

    # Subsample for performance
    n_display = min(30000, n_gaussians)
    indices = np.random.choice(n_gaussians, n_display, replace=False)
    display_colors = base_colors[indices]

    # Pre-compute positions for all timesteps
    print("\nPre-computing fluid positions...")
    all_positions = []
    for t in range(n_timesteps):
        g = dynamic.get_gaussians_at_time(float(t))
        pos = g.positions.detach().numpy()[indices]
        all_positions.append(pos)
        if (t + 1) % 10 == 0:
            print(f"  Frame {t + 1}/{n_timesteps}")

    # Compute velocities for color-coding
    print("Computing velocity field...")
    velocities = []
    for t in range(n_timesteps - 1):
        vel = all_positions[t + 1] - all_positions[t]
        velocities.append(vel)
    velocities.append(velocities[-1])

    # Compute max velocity for normalization
    max_vel = max(np.linalg.norm(v, axis=1).max() for v in velocities)
    print(f"  Max velocity: {max_vel:.4f}")

    # Start Viser
    import viser

    print("\n" + "=" * 60)
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print("=" * 60)
    print("\n  >>> Open in your browser: http://localhost:8080 <<<")
    print("\nControls:")
    print("  - Left-click + drag: Rotate view")
    print("  - Right-click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Use GUI controls to play/pause animation")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # State
    current_time = 0
    is_playing = False
    playback_speed = 1.0
    color_by_velocity = True
    point_size = 0.04

    def velocity_to_color(vel_mag, vel_max):
        """Convert velocity magnitude to color (blue->cyan->green->yellow->red)."""
        v = np.clip(vel_mag / (vel_max + 1e-8), 0, 1)
        colors = np.zeros((len(v), 3))

        # Blue to cyan
        mask = v < 0.25
        t = v[mask] / 0.25
        colors[mask] = np.column_stack([np.zeros_like(t), t, np.ones_like(t)])

        # Cyan to green
        mask = (v >= 0.25) & (v < 0.5)
        t = (v[mask] - 0.25) / 0.25
        colors[mask] = np.column_stack([np.zeros_like(t), np.ones_like(t), 1 - t])

        # Green to yellow
        mask = (v >= 0.5) & (v < 0.75)
        t = (v[mask] - 0.5) / 0.25
        colors[mask] = np.column_stack([t, np.ones_like(t), np.zeros_like(t)])

        # Yellow to red
        mask = v >= 0.75
        t = (v[mask] - 0.75) / 0.25
        colors[mask] = np.column_stack([np.ones_like(t), 1 - t, np.zeros_like(t)])

        return colors

    def update_display(t_idx):
        """Update the point cloud display."""
        t_idx = int(t_idx) % n_timesteps
        positions = all_positions[t_idx]

        if color_by_velocity:
            vel_mag = np.linalg.norm(velocities[t_idx], axis=1)
            vel_colors = velocity_to_color(vel_mag, max_vel)
            # Blend velocity color with base color
            final_colors = 0.6 * vel_colors + 0.4 * display_colors
        else:
            final_colors = display_colors

        final_colors = (np.clip(final_colors, 0, 1) * 255).astype(np.uint8)

        server.scene.add_point_cloud(
            "/fluid",
            points=positions,
            colors=final_colors,
            point_size=point_size,
            point_shape="circle"
        )

    # Add origin frame
    server.scene.add_frame("/origin", wxyz=(1, 0, 0, 0), position=(0, 0, 0),
                           axes_length=1.0, axes_radius=0.02)

    # Initial display
    update_display(0)

    # GUI
    server.gui.add_markdown("## Fluid Flow Viewer")
    server.gui.add_markdown(f"**Points:** {n_display:,} of {n_gaussians:,}")
    server.gui.add_markdown(f"**Frames:** {n_timesteps}")

    server.gui.add_markdown("---")
    server.gui.add_markdown("### Playback")

    play_btn = server.gui.add_button("▶ Play")
    time_slider = server.gui.add_slider("Frame", min=0, max=n_timesteps - 1,
                                         step=1, initial_value=0)
    speed_slider = server.gui.add_slider("Speed", min=0.2, max=3.0,
                                          step=0.1, initial_value=1.0)

    server.gui.add_markdown("---")
    server.gui.add_markdown("### Display")

    size_slider = server.gui.add_slider("Point Size", min=0.01, max=0.1,
                                         step=0.005, initial_value=0.04)
    vel_checkbox = server.gui.add_checkbox("Color by velocity", initial_value=True)

    server.gui.add_markdown("---")
    server.gui.add_markdown("### Legend")
    server.gui.add_markdown("🔵 Slow → 🟢 Medium → 🔴 Fast")

    @play_btn.on_click
    def toggle_play(_):
        nonlocal is_playing
        is_playing = not is_playing
        play_btn.name = "⏸ Pause" if is_playing else "▶ Play"

    @time_slider.on_update
    def on_time_change(_):
        nonlocal current_time
        if not is_playing:
            current_time = time_slider.value
            update_display(current_time)

    @speed_slider.on_update
    def on_speed_change(_):
        nonlocal playback_speed
        playback_speed = speed_slider.value

    @size_slider.on_update
    def on_size_change(_):
        nonlocal point_size
        point_size = size_slider.value
        update_display(current_time)

    @vel_checkbox.on_update
    def on_vel_change(_):
        nonlocal color_by_velocity
        color_by_velocity = vel_checkbox.value
        update_display(current_time)

    # Animation loop
    last_update = time.time()
    frame_interval = 1.0 / 15.0  # 15 FPS base

    try:
        while True:
            now = time.time()

            if is_playing:
                if (now - last_update) > (frame_interval / playback_speed):
                    current_time = (current_time + 1) % n_timesteps
                    time_slider.value = current_time
                    update_display(current_time)
                    last_update = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
