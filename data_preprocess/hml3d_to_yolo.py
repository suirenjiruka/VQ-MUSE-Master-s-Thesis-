import numpy as np
import json
import os
import random
from os.path import join as pjoin
from .keypoint_normalize import normalize

# SMPL22 → 13-joint YOLO format
# Output order: [nose≈head, l_shoulder, r_shoulder, l_elbow, r_elbow,
#                l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
SMPL22_TO_13 = [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

WINDOW_SIZE = 7

# 8 evenly-spaced azimuths matching the Blender camera setup
CAMERA_AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
CAMERA_ELEVATION = 10.0  # degrees downward tilt

# Fixed virtual camera parameters — simulate a real fixed camera
# Person starts at world origin; initial pelvis projects to image center.
# SCALE_FACTOR controls apparent size: 150 px/m → standing person ≈ 35% of frame height.
IMAGE_W      = 1280
IMAGE_H      = 720
SCALE_FACTOR = 150  # pixels per world-unit (meter)

# Index of L/R hip in the 13-joint output (used to find initial pelvis position)
_LHIP_IDX = 7  # SMPL22_TO_13[7] = joint 1
_RHIP_IDX = 8  # SMPL22_TO_13[8] = joint 2


def project_to_yolo_xyn_with_angle(positions_3d, azimuth_deg=0, elevation_deg=10.0):
    """
    Orthographic projection of [T, 22, 3] HumanML3D joints → [T, 13, 2] in image space.

    Uses a FIXED virtual camera at the given azimuth/elevation.
    The initial pelvis position maps to the image center, and all frames share the
    same pixel-per-metre scale (SCALE_FACTOR).  This gives hip_mid and scale in the
    global descriptor consistent physical meaning across sequences (unlike per-sequence
    min/max normalisation, which collapses every clip into its own bounding box).

    Returns [T, 13, 2]; values can exceed [0,1] if the person walks far from centre.
    """
    joints_13 = positions_3d[:, SMPL22_TO_13, :]  # [T, 13, 3]

    theta = np.radians(azimuth_deg)
    phi   = np.radians(elevation_deg)

    # Camera basis vectors
    cam_right = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    cam_up    = np.array([-np.sin(theta) * np.sin(phi),
                           np.cos(phi),
                          -np.cos(theta) * np.sin(phi)])

    # Orthographic projection (world → camera plane)
    px =  (joints_13 @ cam_right)   # [T, 13]  image-x: positive = right
    py = -(joints_13 @ cam_up)      # [T, 13]  image-y: positive = down (top=0)

    # Anchor: project the initial pelvis (mean of L/R hip at t=0) to image centre.
    # This means the person always starts roughly centred, but retains consistent
    # scale and meaningful relative translation across frames.
    init_pelvis_3d = 0.5 * (
        positions_3d[0, SMPL22_TO_13[_LHIP_IDX]] +
        positions_3d[0, SMPL22_TO_13[_RHIP_IDX]]
    )  # [3]
    init_px =  init_pelvis_3d @ cam_right
    init_py = -(init_pelvis_3d @ cam_up)

    # Map to pixel space with fixed scale
    px_pixel = (px - init_px) * SCALE_FACTOR + IMAGE_W / 2   # [T, 13]
    py_pixel = (py - init_py) * SCALE_FACTOR + IMAGE_H / 2   # [T, 13]

    # Normalise to [0, 1] by image dimensions (may exceed range for large translations)
    x_norm = px_pixel / IMAGE_W
    y_norm = py_pixel / IMAGE_H

    return np.stack([x_norm, y_norm], axis=-1)  # [T, 13, 2]


def joints_to_yolo_json(joints_path, output_path, azimuth_deg=None):
    """
    Convert a single HumanML3D new_joints .npy file to YOLO-format JSON.
    Input:  [T, 22, 3] world-space joint positions
    Output: [num_windows, 7, 15, 2] — same format as Yolo_test.py / PR-VIPE

    azimuth_deg: camera azimuth in degrees. Randomly sampled if None.
    """
    if azimuth_deg is None:
        azimuth_deg = random.choice(CAMERA_AZIMUTHS)

    positions     = np.load(joints_path)                                              # [T, 22, 3]
    keypoints_2d  = project_to_yolo_xyn_with_angle(positions, azimuth_deg,
                                                    CAMERA_ELEVATION)                 # [T, 13, 2]
    keypoints_2d  = [normalize(frame, True) for frame in keypoints_2d]               # each → [15, 2]

    T           = len(keypoints_2d)
    num_windows = T // WINDOW_SIZE

    output = []
    for i in range(num_windows):
        window = keypoints_2d[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]  # [7, 15, 2]
        output.append(window)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def batch_convert(joints_dir, output_dir, mid_list_path=None, seed=None):
    """
    Batch convert HumanML3D new_joints .npy files → YOLO-format JSON files.
    Each motion receives a randomly assigned camera azimuth angle.
    An azimuth log is saved to <output_dir>/_azimuth_log.json for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    if mid_list_path:
        with open(mid_list_path) as f:
            mids  = [l.strip() for l in f.readlines()]
        files = [f"{mid}.npy" for mid in mids]
    else:
        files = [f for f in os.listdir(joints_dir) if f.endswith(".npy")]

    print(f"Converting {len(files)} files with random camera azimuths ...")
    angle_log = {}
    for fname in files:
        npy_path = pjoin(joints_dir, fname)
        if not os.path.exists(npy_path):
            print(f"  skip (not found): {fname}")
            continue
        out_path = pjoin(output_dir, fname.replace(".npy", ".json"))
        azimuth  = random.choice(CAMERA_AZIMUTHS)
        angle_log[fname.replace(".npy", "")] = azimuth
        try:
            joints_to_yolo_json(npy_path, out_path, azimuth_deg=azimuth)
        except Exception as e:
            print(f"  error {fname}: {e}")

    log_path = pjoin(output_dir, "_azimuth_log.json")
    with open(log_path, "w") as f:
        json.dump(angle_log, f, indent=2)
    print(f"Done. Azimuth log saved to {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints_dir", required=True, help="HumanML3D new_joints directory")
    parser.add_argument("--output",     required=True, help="output JSON directory")
    parser.add_argument("--mid_list",   default=None,  help="optional txt list of motion IDs")
    parser.add_argument("--seed",       type=int, default=42,
                        help="random seed for azimuth assignment reproducibility")
    args = parser.parse_args()

    batch_convert(args.joints_dir, args.output, args.mid_list, args.seed)
