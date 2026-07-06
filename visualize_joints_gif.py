import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Correct joint order from SMPL22_TO_13 = [15,16,17,18,19,20,21,1,2,4,5,7,8]
# 0:head  1:lsho  2:rsho  3:lelb  4:relb  5:lwri  6:rwri  7:lhip  8:rhip  9:lkne 10:rkne 11:lank 12:rank
JOINT_NAMES = ['head','lsho','rsho','lelb','relb','lwri','rwri','lhip','rhip','lkne','rkne','lank','rank']
SKELETON = [
    (0,1),(0,2),    # head - shoulders
    (1,2),          # shoulder bar
    (1,3),(3,5),    # left arm: lsho-lelb-lwri
    (2,4),(4,6),    # right arm: rsho-relb-rwri
    (1,7),(2,8),    # torso sides
    (7,8),          # hip bar
    (7,9),(9,11),   # left leg
    (8,10),(10,12), # right leg
]

IMAGE_W, IMAGE_H = 1280, 720

DATA_DIR = '//wsl.localhost/Ubuntu/home/imlab/HumanML3D/repo/HumanML3D/yolo_format_13js'
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

files   = sorted(os.listdir(DATA_DIR))[:5]

for fname in files:
    mid = fname.replace('.json', '')
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    arr = np.array(data)                   # [num_windows, 7, 15, 2]
    centered = arr[:, :, :13, :]           # [nw, 7, 13, 2]  pelvis-centred, scaled
    global_desc = arr[:, :, 13:, :]        # [nw, 7,  2, 2]  [[pelvis_x/W, pelvis_y_cart/H], [scale, scale]]

    frames_c = centered.reshape(-1, 13, 2) # [T, 13, 2]
    frames_g = global_desc.reshape(-1, 2, 2)

    pelvis_norm = frames_g[:, 0, :]        # [T, 2]  pelvis in Cartesian-normalised coords
    scale       = frames_g[:, 1, 0]        # [T]

    # Restore absolute Cartesian pixel positions
    # centered = (abs - pelvis_px) * scale  →  abs = centered/scale + pelvis_px
    pelvis_px   = pelvis_norm * np.array([IMAGE_W, IMAGE_H])          # [T, 2]
    abs_joints  = frames_c / scale[:, None, None] + pelvis_px[:, None, :]  # [T, 13, 2]

    x_all = abs_joints[:, :, 0]
    y_all = abs_joints[:, :, 1]  # already Cartesian (y up), no need to negate

    pad  = 50
    xlim = (x_all.min() - pad, x_all.max() + pad)
    ylim = (y_all.min() - pad, y_all.max() + pad)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    title = ax.set_title(f'{mid}  frame 0/{len(abs_joints)}', fontsize=9)

    lines = [ax.plot([], [], 'b-', linewidth=2, alpha=0.8)[0] for _ in SKELETON]
    dots  = ax.scatter([], [], c='red', s=40, zorder=5)

    def update(t, abs_joints=abs_joints, lines=lines, dots=dots, title=title, mid=mid):
        kps = abs_joints[t]
        x, y = kps[:, 0], kps[:, 1]
        for line, (a, b) in zip(lines, SKELETON):
            line.set_data([x[a], x[b]], [y[a], y[b]])
        dots.set_offsets(np.stack([x, y], axis=1))
        title.set_text(f'{mid}  frame {t}/{len(abs_joints)}')
        return lines + [dots, title]

    ani = animation.FuncAnimation(fig, update, frames=len(abs_joints), interval=80, blit=True)
    out_path = os.path.join(OUT_DIR, f'joint_{mid}.gif')
    ani.save(out_path, writer='pillow', fps=12)
    plt.close()
    print(f'Saved {out_path}')

print('Done.')