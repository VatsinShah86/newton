"""Animated visualization of end-effector trajectory from teleop delta data.

Loads teleop.csv (10 fps), integrates delta_pos / delta_rot (rotation vectors)
from origin, and animates the movement in real time:
  - ghost trail of the full trajectory drawn faintly in the background
  - rolling colour-coded trail grows behind the moving tip
  - current coordinate frame (RGB = X/Y/Z axes) follows the tip
  - faded ghost frames shown at every GHOST_INTERVAL steps in the trail
  - time counter in the title
"""

import csv
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation

FPS = 10
TRAIL_LEN = 20        # how many past positions to keep in the rolling trail
GHOST_INTERVAL = 8    # draw a faded past frame every N steps in the trail


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, dpos, drot = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            dpos.append([float(row["delta_pos_x"]),
                         float(row["delta_pos_y"]),
                         float(row["delta_pos_z"])])
            drot.append([float(row["delta_rot_x"]),
                         float(row["delta_rot_y"]),
                         float(row["delta_rot_z"])])
    return np.array(times), np.array(dpos), np.array(drot)


def integrate(dpos: np.ndarray, drot: np.ndarray) -> tuple[np.ndarray, list[Rotation]]:
    """Integrate deltas into absolute poses.

    delta_pos: applied in world frame.
    delta_rot: rotation vector, composed as body-frame right-multiply
               (matches how oculus_teleop computes it: prev_rot.inv() * rot).
    """
    n = len(dpos)
    positions = np.zeros((n + 1, 3))
    rotations = [Rotation.identity()]
    pos = np.zeros(3)
    rot = Rotation.identity()
    for i in range(n):
        pos = pos + dpos[i]
        rot = rot * Rotation.from_rotvec(drot[i])
        positions[i + 1] = pos
        rotations.append(rot)
    return positions, rotations


def frame_segments(pos: np.ndarray, rot: Rotation, length: float):
    """Return (xs, ys, zs, color) for each of the three frame axes."""
    segs = []
    for axis_idx, color in enumerate(("r", "g", "b")):
        local = np.zeros(3)
        local[axis_idx] = length
        tip = pos + rot.apply(local)
        segs.append(([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]], color))
    return segs


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "teleop.csv")

    times, dpos, drot = load_csv(csv_path)
    positions, rotations = integrate(dpos, drot)
    n_steps = len(positions)   # includes t=0 origin

    # Axis length = 30 % of the largest spatial span so frames are always visible
    span = max((positions.max(axis=0) - positions.min(axis=0)).max(), 1e-5)
    axis_len = span * 0.30

    # Fixed axis limits
    mid = (positions.max(axis=0) + positions.min(axis=0)) / 2
    half = span / 2 + span * 0.30   # 30 % padding

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    # Static ghost of the full trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color="lightgray", linewidth=0.8, alpha=0.35, zorder=1)
    ax.scatter(*positions[0], color="green", s=70, marker="^", zorder=6,
               label="Start")
    ax.scatter(*positions[-1], color="red", s=70, marker="*", zorder=6,
               label=f"End  t={times[-1]:.1f}s")

    # --- mutable artists ---
    trail_line, = ax.plot([], [], [], color="steelblue", linewidth=2.0,
                          alpha=0.9, zorder=3)
    tip_dot = ax.scatter([], [], [], color="orange", s=60, zorder=5,
                         depthshade=False)

    # Three lines for the live coordinate frame (X=red, Y=green, Z=blue)
    live_axes = [ax.plot([], [], [], linewidth=2.5, zorder=5)[0] for _ in range(3)]

    # Ghost frame lines — reused for each ghost drawn in the trail
    # We pre-allocate a pool: one set per ghost slot
    max_ghosts = TRAIL_LEN // GHOST_INTERVAL + 1
    ghost_lines = [[ax.plot([], [], [], linewidth=1.2, alpha=0.35, zorder=2)[0]
                    for _ in range(3)]
                   for _ in range(max_ghosts)]

    title = ax.set_title("")

    ax.legend(loc="upper left", fontsize=9)
    # Custom legend for frame axes
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="green", marker="^", linestyle="None", label="Start"),
        Line2D([0], [0], color="red",   marker="*", linestyle="None", label="End"),
        Line2D([0], [0], color="r",  linewidth=2, label="X axis"),
        Line2D([0], [0], color="g",  linewidth=2, label="Y axis"),
        Line2D([0], [0], color="b",  linewidth=2, label="Z axis"),
    ], loc="upper left", fontsize=8)

    def _set_frame(lines, pos, rot, length, alpha=1.0):
        for line, (xs, ys, zs, color) in zip(lines, frame_segments(pos, rot, length)):
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            line.set_color(color)
            line.set_alpha(alpha)

    def _hide_frame(lines):
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

    def update(step: int):
        # Rolling trail
        start = max(0, step - TRAIL_LEN)
        trail = positions[start:step + 1]
        trail_line.set_data(trail[:, 0], trail[:, 1])
        trail_line.set_3d_properties(trail[:, 2])

        # Tip dot
        p = positions[step]
        tip_dot._offsets3d = ([p[0]], [p[1]], [p[2]])

        # Live frame at tip
        _set_frame(live_axes, p, rotations[step], axis_len)

        # Ghost frames along the visible trail
        ghost_idx = 0
        for past in range(start, step, GHOST_INTERVAL):
            if ghost_idx >= max_ghosts:
                break
            alpha = 0.15 + 0.25 * (past - start) / max(step - start, 1)
            _set_frame(ghost_lines[ghost_idx], positions[past],
                       rotations[past], axis_len * 0.6, alpha=alpha)
            ghost_idx += 1
        # Hide unused ghost slots
        for i in range(ghost_idx, max_ghosts):
            _hide_frame(ghost_lines[i])

        t = times[step - 1] if step > 0 else 0.0
        title.set_text(f"Teleop end-effector   t = {t:.2f} s   "
                       f"step {step}/{n_steps - 1}")
        return trail_line, tip_dot, *live_axes, title

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_steps,
        interval=int(1000 / FPS),
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
