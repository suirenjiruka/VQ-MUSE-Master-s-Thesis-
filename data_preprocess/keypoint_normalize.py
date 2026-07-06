import json
import os
import numpy as np

LHIP = 7
RHIP = 8
L_shoulder = 1
R_shoulder = 2
image_sizes = [720, 1280]

def normalize(pose2d, norm: bool, height= image_sizes[0], width= image_sizes[1]):
    # pose2D: [keypointts, 2D (x, y)]
    if not isinstance(pose2d, np.ndarray):
        pose2d = np.asarray(pose2d, dtype=float) 

    #normalize
    if norm:
        pose2d = pose2d.copy()
        pose2d[:, 0] = pose2d[:, 0] * width
        pose2d[:, 1] = (1 - pose2d[:, 1]) * height


    # set the pelvis as origin (0, 0)
    hip_mid = 0.5 * (pose2d[LHIP] + pose2d[RHIP])
    p = pose2d - hip_mid[None, :]
    # scale thhe size, make the distance between hip and shoulder be 0.5
    pts = np.stack([p[R_shoulder], p[L_shoulder], p[RHIP], p[LHIP]], axis=0)
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    max_dist = np.nanmax(dists)  # handle NaN if missing
    if not np.isfinite(max_dist) or max_dist < 1e-6:  # handle infinity or 0
        scale = 1.0
    else:
        # paper normalize to 0.5
        scale = 0.5 / max_dist
    p *= scale
    hip_mid /= [width, height]
    #加入額外一個表示位移的parameter [2, 2] = [[x, y], [z, z]]，normalize前 pelvis x, y, z cordinate(normalized by image width and height)，z用scale程度表示
    cord = np.array([hip_mid, [scale, scale]])
    p = np.concatenate([p, cord], axis = 0) 
    return p.tolist()


def batch_normal(dir_path, output_dir):
    files = os.listdir(dir_path)
    for file in files:
        path = os.path.join(dir_path, file)
        if os.path.isfile(path):
            with open(path, "r")as jfile:
                data = json.load(jfile)
                data = [[normalize(frame, True) for frame in d] for d in data] 

            output_path = os.path.join(output_dir, file)
            with open(output_path, "w")as jfile:
                json.dump(data, jfile, indent=2)

if __name__== '__main__':
    batch_normal("/mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/eval_joints", "/mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/normal_joints") #