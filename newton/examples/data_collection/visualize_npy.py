import argparse
import os
import tempfile

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    default_mpl_dir = os.path.expanduser("~/.config/matplotlib")
    if not os.path.isdir(default_mpl_dir) or not os.access(default_mpl_dir, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")

import matplotlib.pyplot as plt

# Define the base directory where the npy files are located
# According to the analyzed script, data is in surrol/assets/data
# Adjust this if your DATA_DIR_PATH is different.
BASE_DIR = os.path.join(os.path.dirname(__file__))


def _resolve_point_cloud_path(folder_name, file_number):
    preferred_path = os.path.join(BASE_DIR, "data", folder_name, "pc", f"pc_{file_number}.npy")
    legacy_path = os.path.join(BASE_DIR, "data", folder_name, f"pc_{file_number}.npy")

    if os.path.exists(preferred_path):
        return preferred_path
    if os.path.exists(legacy_path):
        return legacy_path
    return preferred_path


def _set_equal_aspect_2d(ax, x_values, y_values):
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, 1e-6)

    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
    ax.set_aspect("equal", adjustable="box")

def visualize_seg(file_path):
    """Visualizes a segmentation .npy file."""
    try:
        data = np.load(file_path)
        print(f"Loaded segmentation data from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Unique values: {np.unique(data)}")

        plt.imshow(data, cmap='gray')
        plt.title(f"Segmentation Mask: {os.path.basename(file_path)}")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error loading or visualizing {file_path}: {e}")

def visualize_depth(file_path):
    """Visualizes a depth .npy file."""
    try:
        data = np.load(file_path)
        print(f"Loaded depth data from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {data.min()}, Max value: {data.max()}")

        plt.imshow(data, cmap='viridis')
        plt.title(f"Depth Map: {os.path.basename(file_path)}")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error loading or visualizing {file_path}: {e}")


def visualize_pc(file_path):
    """Visualizes a point cloud .npy file.

    Accepts either shape (N, 3) or (H, W, 3). Zero vectors are treated as
    invalid pixels and filtered out before plotting.
    """
    try:
        data = np.load(file_path)
        print(f"Loaded point cloud data from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

        # Normalise to (N, 3)
        if data.ndim == 3 and data.shape[2] == 3:
            data = data.reshape(-1, 3)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape (N, 3) or (H, W, 3), got {data.shape}")

        # Remove invalid (zero) points
        valid = (data != 0).any(axis=1)
        data = data[valid]
        print(f"Valid points: {data.shape[0]}")
        if data.shape[0] == 0:
            print("No valid points to display.")
            return

        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as import_error:
            print("3D plotting backend is unavailable. Falling back to orthographic 2D projections.")
            print(f"Underlying import error: {import_error}")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            projections = [
                (0, 1, "X", "Y"),
                (0, 2, "X", "Z"),
                (1, 2, "Y", "Z"),
            ]
            for ax, (x_idx, y_idx, x_label, y_label) in zip(axes, projections, strict=True):
                x_values = data[:, x_idx]
                y_values = data[:, y_idx]
                ax.scatter(x_values, y_values, s=1)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"{x_label}{y_label} Projection")
                _set_equal_aspect_2d(ax, x_values, y_values)

            fig.suptitle(f"Point Cloud Projections: {os.path.basename(file_path)}")
            plt.tight_layout()
            plt.show()
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Point Cloud: {os.path.basename(file_path)}")

        plt.show()
    except Exception as e:
        print(f"Error loading or visualizing {file_path}: {e}")

def visualize_act(file_path):
    """Confirm action sizes"""
    try:
        data = np.load(file_path)
        print(f"Loaded action data from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

    except Exception as e:
        print(f"Error loading or visualizing {file_path}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files (segmentation, depth, point cloud, or actions).")
    parser.add_argument("path", type=str, help="Path to a .npy file, or a folder name (legacy: requires file_number too).")
    parser.add_argument("file_number", type=int, nargs="?", help="File number for the legacy folder/number addressing scheme.")

    args = parser.parse_args()

    if args.path.endswith(".npy") or os.path.isfile(args.path):
        file_path = args.path
    else:
        if args.file_number is None:
            parser.error("file_number is required when using the legacy folder/number scheme.")
        file_path = _resolve_point_cloud_path(args.path, args.file_number)

    visualize_pc(file_path)
    
