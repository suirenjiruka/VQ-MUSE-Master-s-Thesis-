"""VAMotion demo backend.

Loads best.tar + frozen VQ + T5 once, exposes /generate and /edit that return
22-joint world positions [F][22][3].  The heavy mesh rendering is NOT done here —
the browser (Three.js) renders the mesh in real time, which is what makes this fast.

Run (in the WSL / conda env that has torch + the project deps):
    pip install flask                       # once
    cd /mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/Application
    CKPT=best.tar python server.py          # serves on http://localhost:5000

Env overrides:  TRANS_CFG (default configs/train_vamotion_hml.yaml), CKPT (default best.tar), PORT (5000)
"""
import os, sys, time, uuid, threading
from os.path import join as pjoin

import numpy as np
import torch
from flask import Flask, request, jsonify, Response

# make the project (master_project) importable regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # relative paths in configs resolve from the project root

from configs.load_config import load_config
from utils.visualize_motion_editing_hml import (
    load_mean_std, load_vq_model, load_trans_model, denorm_motion, motion_to_joints,
    resolve_trans_ckpt_path,
)

TRANS_CFG = os.environ.get("TRANS_CFG", pjoin(ROOT, "configs", "train_vamotion_hml.yaml"))
CKPT      = os.environ.get("CKPT", "best.tar")
PORT      = int(os.environ.get("PORT", "5000"))
USE_EMA   = os.environ.get("USE_EMA", "1") != "0"   # eval 設定 use_ema:True；用 USE_EMA=0 可載原始權重對照


def apply_ema(model, ckpt_path, device):
    """把 checkpoint 裡的 EMA 影子權重覆蓋到模型（對齊 eval 的 use_ema:True）。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    shadow = ck.get("ema", {}).get("shadow", {}) if isinstance(ck.get("ema"), dict) else {}
    if not shadow:
        print("[boot] ckpt 無 EMA shadow，沿用原始 training_model 權重", flush=True)
        return 0
    n = 0
    with torch.no_grad():
        for name, pp in model.named_parameters():
            if name in shadow:
                pp.data.copy_(shadow[name].to(pp.device)); n += 1
    print(f"[boot] 已套用 EMA 權重到 {n} 個參數", flush=True)
    return n


print(f"[boot] loading model … cfg={TRANS_CFG} ckpt={CKPT} use_ema={USE_EMA}", flush=True)
cfg = load_config(TRANS_CFG)
device = torch.device(cfg.exp.device if (cfg.exp.device == "cpu" or torch.cuda.is_available()) else "cpu")
mean, std = load_mean_std(cfg)
vq_cfg = load_config(pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name))
vq_model = load_vq_model(cfg, vq_cfg, device)
trans = load_trans_model(cfg, vq_cfg, CKPT, device)          # also sets cfg.vq (loads raw training_model)
if USE_EMA:
    apply_ema(trans, resolve_trans_ckpt_path(cfg, CKPT), device)
trans.eval()
if hasattr(trans, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
    trans.set_vq_codebook(vq_model.quantizer.codebook)
UNIT = cfg.data.unit_length
DIM  = cfg.data.dim_pose
MAXL = cfg.data.max_motion_length
FPS  = int(getattr(cfg.data, "fps", 20))
print(f"[boot] ready on {device} · unit={UNIT} · dim={DIM} · fps={FPS}", flush=True)

# --- body render mode ---
# MESH_MODE=lbs      : position-driven LBS(快速 fallback)。用 22 關節位置直接把 SMPL 皮膚變形貼合，
#                      無 per-frame 優化。① SMPL pose blendshape 補回關節體積 ④ 沿骨軸伸縮消關節縫隙。
#                      四肢 twist 不從位置猜(forearm 等 roll 不可觀測，硬解反而扭曲)→ 用乾淨 swing。
# MESH_MODE=own(預設): minimum-twist IK + 官方 SMPL skinning；固定精瘦身形，不拉扯表面。
# MESH_MODE=fit      : visualize 的 joints2smpl SMPLify(保留對照)。
# MESH_MODE=fast     : rot6d 直接前向(免優化，會扭曲，不建議)。
# MESH_MODE=metaball : 回傳關節點，前端 metaball 皮膚(不碰 SMPL)。
MESH_MODE   = os.environ.get("MESH_MODE", "own").lower()
OWN_ITERS   = int(os.environ.get("OWN_ITERS", "48"))
OWN_BUDGET  = float(os.environ.get("OWN_BUDGET", "2.8")) # fitting time budget in seconds
SMPL_ITERS  = int(os.environ.get("SMPL_ITERS", "8"))    # fit 模式 warm-start 後迭代
SMPL_STRIDE = int(os.environ.get("SMPL_STRIDE", "1"))
WARM        = os.environ.get("WARM", "1") != "0"        # 用動作旋轉當初始姿勢(warm-start)
LBS_POSE_BS = os.environ.get("LBS_POSE_BS", "1") != "0" # ① pose blendshape(補關節體積，實測有效)
LBS_STRETCH = os.environ.get("LBS_STRETCH", "1") != "0" # ④ 沿骨軸伸縮(消關節縫隙)
SMOOTH_WIN  = int(os.environ.get("SMOOTH_WIN", "2"))    # 關節時間平滑半徑(減少抖動；0=關)
TAUBIN_ITERS= int(os.environ.get("TAUBIN_ITERS", "2"))  # 網格 Taubin 平滑迭代(減少表面扭曲/重疊觀感；0=關)
IK_SMOOTH_WIN = int(os.environ.get("IK_SMOOTH_WIN", "3"))
NEED_SMPL = MESH_MODE in ("lbs", "own", "fit", "fast")
MESH_OK = False
FACES = np.zeros((0, 3), np.int64)
VIS_CFG = VISUAL_CFG = SMPL_MODEL = LBS = None
SMPL_DIR = None
fit_vertices = proxy_vertices = _init_pose_fns = None
if NEED_SMPL:
    try:
        import smplx as _smplx
        from smplx.lbs import lbs as _raw_lbs
        # warm-start 用的旋轉轉換(與 motion_to_proxy_vertices 同慣例)
        from utils.motion_process_bvh import recover_root_rot_pos as _rrp
        from utils.common.quaternion import qinv as _qinv
        from utils.rotation_conversions import (
            quaternion_to_matrix as _q2m, matrix_to_quaternion as _m2q,
            rotation_6d_to_matrix as _r6m, matrix_to_axis_angle as _m2aa,
            axis_angle_to_matrix as _aa2m,
        )
        _init_pose_fns = (_rrp, _qinv, _q2m, _r6m, _m2aa)
        VISUAL_CFG = load_config(pjoin(ROOT, "utils", "visual_config.yaml"))
        SMPL_DIR = VISUAL_CFG.SMPL_MODEL_DIR
        _faces_model = _smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=1)
        FACES = np.asarray(_faces_model.faces, dtype=np.int64)
        raw_model = None
        if MESH_MODE in ("lbs", "own"):
            # position-driven LBS 綁定資產(rest 模板 / skinning 權重 / rest 關節 / 骨架樹 / pose 修形基底)
            v_template = _faces_model.v_template.to(device).float()                     # [6890, 3]
            shape_values = list(getattr(VISUAL_CFG, "SMPL_BODY_BETAS", [0.0] * 10))
            if len(shape_values) != int(_faces_model.num_betas):
                raise ValueError(
                    f"SMPL_BODY_BETAS needs {_faces_model.num_betas} values, got {len(shape_values)}"
                )
            shape_betas = torch.as_tensor(shape_values, device=device, dtype=v_template.dtype)
            v_template = v_template + torch.einsum(
                "vcn,n->vc", _faces_model.shapedirs.to(device).float(), shape_betas
            )
            J_regressor = _faces_model.J_regressor.to(device).float()                   # [24, 6890]
            lbs_w = _faces_model.lbs_weights.to(device).float()                         # [6890, 24]
            parents = _faces_model.parents.to(device).long()                            # [24]
            J_rest = J_regressor @ v_template                                           # [24, 3] rest 關節
            if MESH_MODE == "own":
                male_path = pjoin(SMPL_DIR, "smplx", "SMPLX_MALE.npz")
                if not os.path.isfile(male_path):
                    raise FileNotFoundError(f"Male SMPL-X core model not found: {male_path}")
                with np.load(male_path, allow_pickle=True) as male_data:
                    FACES = np.asarray(male_data["f"], dtype=np.int64)
                    male_v_template = torch.as_tensor(
                        male_data["v_template"], device=device, dtype=torch.float32
                    )
                    male_shapedirs = torch.as_tensor(
                        male_data["shapedirs"][:, :, :10], device=device, dtype=torch.float32
                    )
                    male_posedirs_np = male_data["posedirs"]
                    male_posedirs = torch.as_tensor(
                        male_posedirs_np.reshape(-1, male_posedirs_np.shape[-1]).T,
                        device=device,
                        dtype=torch.float32,
                    ).contiguous()
                    J_regressor = torch.as_tensor(
                        male_data["J_regressor"], device=device, dtype=torch.float32
                    )
                    lbs_w = torch.as_tensor(
                        male_data["weights"], device=device, dtype=torch.float32
                    )
                    parents = torch.as_tensor(
                        male_data["kintree_table"][0], device=device, dtype=torch.long
                    ).clone()
                parents[0] = -1
                male_shape_values = list(getattr(VISUAL_CFG, "SMPL_MALE_BETAS", [0.0] * 10))
                if len(male_shape_values) != 10:
                    raise ValueError(
                        f"SMPL_MALE_BETAS needs 10 values, got {len(male_shape_values)}"
                    )
                male_betas = torch.as_tensor(
                    male_shape_values, device=device, dtype=torch.float32
                )
                female_path = pjoin(SMPL_DIR, "smplx", "SMPLX_FEMALE_LOCKED.npz")
                if not os.path.isfile(female_path):
                    raise FileNotFoundError(
                        f"Corresponding female SMPL-X core model not found: {female_path}"
                    )
                with np.load(female_path, allow_pickle=True) as female_data:
                    female_v_template = torch.as_tensor(
                        female_data["v_template"], device=device, dtype=torch.float32
                    )
                gender_alpha = float(
                    getattr(VISUAL_CFG, "SMPL_MALE_GENDER_EXTRAPOLATION", 0.75)
                )
                v_template = (
                    male_v_template
                    + gender_alpha * (male_v_template - female_v_template)
                    + torch.einsum(
                    "vcn,n->vc", male_shapedirs, male_betas
                    )
                )
                J_rest = J_regressor @ v_template
                raw_model = dict(
                    v_template=v_template,
                    shapedirs=male_shapedirs,
                    posedirs=male_posedirs,
                    J_regressor=J_regressor,
                    lbs_w=lbs_w,
                    parents=parents,
                    betas=torch.zeros_like(male_betas),
                )
            NJ = parents.shape[0]
            dir_children = [[] for _ in range(NJ)]                                      # 骨架樹(僅 22 HML 關節連骨)
            for j in range(1, NJ):
                if j < 22:
                    dir_children[int(parents[j])].append(j)
            rest_dir = torch.zeros(NJ, 3, device=device)                               # 每根骨 rest 方向(單位)
            is_leaf = torch.zeros(NJ, dtype=torch.bool, device=device)
            for k in range(NJ):
                ch = dir_children[k]
                if ch:
                    d = (J_rest[ch] - J_rest[k]).mean(0)
                    rest_dir[k] = d / d.norm().clamp(min=1e-8)
                else:
                    is_leaf[k] = True
            posedirs = _faces_model.posedirs.to(device).float()                        # [207, 20670] SMPL 姿勢修形基底
            LBS = dict(v_template=v_template, lbs_w=lbs_w, parents=parents, J_rest=J_rest,
                       NJ=NJ, dir_children=dir_children, rest_dir=rest_dir, is_leaf=is_leaf,
                       posedirs=posedirs, raw_model=raw_model)
        if MESH_MODE in ("fit", "fast"):
            from utils.smpl import SMPL as _SMPL
            from utils.visualize_motion_editing_hml import (
                joints_to_fitted_smpl_vertices as _fit, motion_to_proxy_vertices as _proxy,
            )
            VIS_CFG = load_config(pjoin(ROOT, "configs", "visualize_motion_editing_hml.yaml"))
            VIS_CFG.smpl_fit_iters = SMPL_ITERS
            VIS_CFG.smpl_fit_sample_stride = SMPL_STRIDE
            VIS_CFG.max_frames = MAXL
            SMPL_MODEL = _SMPL(VISUAL_CFG, model_path=pjoin(SMPL_DIR, "smpl")).eval().to(device)
            fit_vertices, proxy_vertices = _fit, _proxy
        MESH_OK = True
        detail = {"lbs": f"pose_bs={int(LBS_POSE_BS)} stretch={int(LBS_STRETCH)}",
                  "own": "minimum-twist IK + male SMPL-X core"}.get(
                      MESH_MODE, f"iters={SMPL_ITERS} stride={SMPL_STRIDE}"
                  )
        print(f"[boot] SMPL mesh ON · mode={MESH_MODE} warm={WARM} · faces={FACES.shape[0]} · {detail}", flush=True)
    except Exception as e:
        MESH_OK = False
        print(f"[boot] SMPL mesh OFF ({type(e).__name__}: {e}) → 回傳關節點，前端改用 metaball", flush=True)
else:
    print("[boot] mesh mode = metaball（回傳關節點，前端 metaball 皮膚）", flush=True)

# 每種 batch(幀數)建一次 smplx 模型並快取，避免每次請求重建
_SMPL_CACHE = {}
def _smpl_for(F):
    m = _SMPL_CACHE.get(F)
    if m is None:
        m = _smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", ext="pkl",
                          batch_size=F).to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _SMPL_CACHE[F] = m
    return m


# ===== position-driven LBS(品質強化版) =====
def _shortest_arc(a, b):
    """最短弧旋轉矩陣，把單位向量 a 轉到 b。a,b:[F,3] → [F,3,3](Rodrigues)。"""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    s2 = (v * v).sum(-1).clamp(min=1e-8)                # sin^2
    Fn = a.shape[0]
    K = torch.zeros(Fn, 3, 3, device=a.device)
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2];  K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] = v[:, 0]
    eye = torch.eye(3, device=a.device).expand(Fn, 3, 3)
    R = eye + K + torch.bmm(K, K) * ((1 - c) / s2).view(Fn, 1, 1)
    flip = (c < -0.999)                                 # a≈-b 時公式退化 → 繞正交軸 180°
    if flip.any():
        R180 = torch.diag(torch.tensor([1., -1., -1.], device=a.device)).expand(Fn, 3, 3)
        R = torch.where(flip.view(Fn, 1, 1), R180, R)
    return R


def lbs_vertices(target_joints):
    """position-driven LBS(品質強化版)：用 22 關節位置直接把 SMPL 皮膚變形貼合(無 per-frame 優化)。
    ① pose blendshape 補關節體積 ④ 沿骨軸伸縮消縫隙。四肢用乾淨 swing(不從位置猜 twist → 不扭曲)。
    target_joints:[F,22,3](含全域位移) → vertices [F,6890,3]。"""
    vt, W, parents = LBS["v_template"], LBS["lbs_w"], LBS["parents"]
    J_rest, NJ, dir_children = LBS["J_rest"], LBS["NJ"], LBS["dir_children"]
    rest_dir, is_leaf, posedirs = LBS["rest_dir"], LBS["is_leaf"], LBS["posedirs"]
    F = target_joints.shape[0]
    T = torch.zeros(F, NJ, 3, device=device)
    T[:, :22] = target_joints                           # 每根骨『目標位置』= 對應關節(root 為絕對位置 → 位移保留)
    if NJ > 22: T[:, 22] = target_joints[:, 20]         # 手→腕(沿用腕位置)
    if NJ > 23: T[:, 23] = target_joints[:, 21]
    eye = torch.eye(3, device=device)
    R = torch.zeros(F, NJ, 3, 3, device=device)
    scale = torch.ones(F, NJ, device=device)            # ④ 每根骨沿骨軸伸縮比例
    for k in range(NJ):
        if is_leaf[k]:                                   # 葉關節(如腕/腳掌)→ 繼承父骨旋轉
            R[:, k] = R[:, int(parents[k])] if k > 0 else eye.expand(F, 3, 3)
            continue
        ch = dir_children[k]
        if len(ch) >= 2:                                # ≥2 子關節 → Kabsch 全旋轉(含 yaw，軀幹/骨盆朝向正確)
            Pr = J_rest[ch] - J_rest[k]                 # [n,3]
            Pc = T[:, ch] - T[:, k:k+1]                 # [F,n,3]
            H = torch.einsum('ij,fik->fjk', Pr, Pc)
            U, _, Vh = torch.linalg.svd(H)
            Ut, V = U.transpose(1, 2), Vh.transpose(1, 2)
            dsign = torch.sign(torch.det(torch.bmm(V, Ut)))
            D = eye.expand(F, 3, 3).clone(); D[:, 2, 2] = dsign
            R[:, k] = torch.bmm(torch.bmm(V, D), Ut)
        else:                                           # 單子關節(四肢)→ swing 對齊骨向(不猜 twist → 不扭曲)
            c = ch[0]
            b1 = T[:, c] - T[:, k]                       # 目前主骨向量
            d = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            R[:, k] = _shortest_arc(rest_dir[k].expand(F, 3), d)
            if LBS_STRETCH:                              # ④ 沿骨軸伸縮比例(消關節縫隙)
                rest_len = (J_rest[c] - J_rest[k]).norm().clamp(min=1e-8)
                scale[:, k] = (b1.norm(dim=-1) / rest_len).clamp(0.5, 2.0)
    # ① pose blendshape：用『局部』旋轉(相對父骨)算 SMPL 姿勢修形，補回關節彎折處的體積
    if LBS_POSE_BS:
        R_local = R.clone()
        for k in range(1, NJ):
            R_local[:, k] = torch.bmm(R[:, int(parents[k])].transpose(1, 2), R[:, k])
        pose_feat = (R_local[:, 1:] - eye).reshape(F, (NJ - 1) * 9)          # [F,207]
        v_posed = vt.unsqueeze(0) + torch.matmul(pose_feat, posedirs).view(F, -1, 3)  # [F,6890,3]
    else:
        v_posed = vt.unsqueeze(0).expand(F, -1, -1)
    # LBS: v' = Σ_k W[:,k] (R_k S_k (v_posed - J_rest_k) + T_k)
    acc = torch.zeros(F, vt.shape[0], 3, device=device)
    for k in range(NJ):
        off = v_posed - J_rest[k]                        # [F,V,3]
        if LBS_STRETCH and not is_leaf[k] and len(dir_children[k]) == 1:
            dr = rest_dir[k]                             # 沿 rest 骨軸伸縮(不改變橫剖面，不會變胖)
            along = torch.einsum('fvj,j->fv', off, dr).unsqueeze(-1) * dr
            off = off + (scale[:, k].view(F, 1, 1) - 1) * along
        rot = torch.einsum('fij,fvj->fvi', R[:, k], off) + T[:, k:k+1]        # [F,V,3]
        acc += W[:, k].view(1, -1, 1) * rot
    return acc


def smooth_rotation_matrices(rotations, radius):
    """Temporally smooth rotations in quaternion space without deforming vertices."""
    frame_count, joint_count = rotations.shape[:2]
    if radius <= 0 or frame_count < 3:
        return rotations

    quaternions = _m2q(rotations)
    aligned = [quaternions[0]]
    max_step = 0.30
    for frame_idx in range(1, frame_count):
        current = quaternions[frame_idx]
        sign = torch.where(
            (current * aligned[-1]).sum(dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(current[..., :1]),
            torch.ones_like(current[..., :1]),
        )
        current = current * sign
        previous = aligned[-1]
        dot = (current * previous).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        alpha = torch.clamp(max_step / angle.clamp(min=1e-6), max=1.0)
        current = previous + alpha * (current - previous)
        current = current / current.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        aligned.append(current)
    quaternions = torch.stack(aligned, dim=0)

    kernel_size = 2 * radius + 1
    offsets = torch.arange(
        kernel_size, device=rotations.device, dtype=rotations.dtype
    ) - radius
    sigma = max(0.8, radius * 0.6)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    channels = joint_count * 4
    signal = quaternions.permute(1, 2, 0).reshape(1, channels, frame_count)
    signal = torch.nn.functional.pad(signal, (radius, radius), mode="replicate")
    weights = kernel.view(1, 1, kernel_size).expand(channels, 1, -1)
    smoothed = torch.nn.functional.conv1d(signal, weights, groups=channels)
    smoothed = smoothed.reshape(joint_count, 4, frame_count).permute(2, 0, 1)
    smoothed = smoothed / smoothed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return _q2m(smoothed)


def orientation_basis(right, up):
    """Build a stable right/up/forward frame from two observed body axes."""
    right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    up = up - (up * right).sum(dim=-1, keepdim=True) * right
    up = up / up.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    forward = torch.cross(right, up, dim=-1)
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.stack([right, up, forward], dim=-1)


def clamp_local_rotation(rotation, max_angle):
    """Limit local joint bend while preserving its rotation axis."""
    axis_angle = _m2aa(rotation)
    angle = axis_angle.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_angle / angle.clamp(min=1e-8), max=1.0)
    return _aa2m(axis_angle * scale)


def ik_smpl_vertices(target_joints):
    """Recover minimum-twist SMPL rotations from the 22 HML joint positions.

    Unlike the fast LBS path, this never stretches or directly deforms mesh
    vertices. Joint directions only determine valid local SMPL rotations; the
    official SMPL skinning function produces the final surface.
    """
    rest_joints = LBS["J_rest"]
    parents = LBS["parents"]
    children = LBS["dir_children"]
    rest_dir = LBS["rest_dir"]
    joint_count = int(LBS["NJ"])
    frame_count = target_joints.shape[0]
    eye = torch.eye(3, device=device, dtype=target_joints.dtype)

    target = torch.zeros(
        frame_count, joint_count, 3, device=device, dtype=target_joints.dtype
    )
    target[:, :22] = target_joints
    if joint_count > 22:
        target[:, 22] = target_joints[:, 20]
    if joint_count > 23:
        target[:, 23] = target_joints[:, 21]

    local_rot = eye.view(1, 1, 3, 3).repeat(frame_count, joint_count, 1, 1)
    global_rot = local_rot.clone()

    for joint_idx in range(joint_count):
        if joint_idx == 0:
            parent_rot = eye.expand(frame_count, 3, 3)
        else:
            parent_rot = global_rot[:, int(parents[joint_idx])]

        child_ids = children[joint_idx]
        if joint_idx == 12:
            # Head yaw is not observable from a 22-joint skeleton. Anchor the
            # neck/head frame to the shoulder axis and upper-spine direction
            # instead of letting a noisy head point twist the entire head.
            rest_basis = orientation_basis(
                rest_joints[17] - rest_joints[16],
                rest_joints[12] - rest_joints[9],
            )
            target_basis = orientation_basis(
                target[:, 17] - target[:, 16],
                target[:, 12] - target[:, 9],
            )
            desired_global = torch.matmul(target_basis, rest_basis.transpose(-1, -2))
            local = torch.bmm(parent_rot.transpose(1, 2), desired_global)
        elif len(child_ids) >= 2:
            rest_offsets = rest_joints[child_ids] - rest_joints[joint_idx]
            world_offsets = target[:, child_ids] - target[:, joint_idx:joint_idx + 1]
            parent_offsets = torch.einsum(
                "fij,fnj->fni", parent_rot.transpose(1, 2), world_offsets
            )
            covariance = torch.einsum("ni,fnj->fij", rest_offsets, parent_offsets)
            u, _, vh = torch.linalg.svd(covariance)
            v = vh.transpose(1, 2)
            ut = u.transpose(1, 2)
            correction = eye.expand(frame_count, 3, 3).clone()
            correction[:, 2, 2] = torch.sign(torch.det(torch.bmm(v, ut)))
            local = torch.bmm(torch.bmm(v, correction), ut)
        elif len(child_ids) == 1:
            child_idx = child_ids[0]
            world_bone = target[:, child_idx] - target[:, joint_idx]
            world_bone = world_bone / world_bone.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            parent_bone = torch.einsum(
                "fij,fj->fi", parent_rot.transpose(1, 2), world_bone
            )
            local = _shortest_arc(
                rest_dir[joint_idx].expand(frame_count, 3), parent_bone
            )
        else:
            # End effectors have no observable twist from joint positions.
            # Identity inherits the parent orientation and avoids arbitrary
            # wrist, foot, and head roll.
            local = eye.expand(frame_count, 3, 3)

        local_rot[:, joint_idx] = local
        global_rot[:, joint_idx] = torch.bmm(parent_rot, local)

    # Root orientation still carries the intended lean and heading. Constrain
    # only the local torso chain so small joint noise cannot create a segmented,
    # snake-like spine or an abruptly tilted neck.
    for joint_idx, max_angle in ((3, 0.18), (6, 0.18), (9, 0.22), (12, 0.25)):
        local_rot[:, joint_idx] = clamp_local_rotation(
            local_rot[:, joint_idx], max_angle
        )

    local_rot = smooth_rotation_matrices(local_rot, IK_SMOOTH_WIN)

    raw_model = LBS.get("raw_model")
    if raw_model is not None:
        betas = raw_model["betas"].view(1, 10).expand(frame_count, -1)
        vertices, joints = _raw_lbs(
            betas,
            local_rot,
            raw_model["v_template"],
            raw_model["shapedirs"],
            raw_model["posedirs"],
            raw_model["J_regressor"],
            raw_model["parents"],
            raw_model["lbs_w"],
            pose2rot=False,
        )
        transl = target_joints[:, 0] - joints[:, 0]
        return vertices + transl[:, None]

    shape_values = list(getattr(VISUAL_CFG, "SMPL_BODY_BETAS", [0.0] * 10))
    betas = torch.as_tensor(
        shape_values, device=device, dtype=target_joints.dtype
    ).view(1, 10).expand(frame_count, -1)
    smpl = _smpl_for(frame_count)
    zero_transl = torch.zeros(frame_count, 3, device=device, dtype=target_joints.dtype)
    initial = smpl(
        global_orient=local_rot[:, :1], body_pose=local_rot[:, 1:],
        betas=betas, transl=zero_transl, pose2rot=False,
    )
    transl = target_joints[:, 0] - initial.joints[:, 0]
    output = smpl(
        global_orient=local_rot[:, :1], body_pose=local_rot[:, 1:],
        betas=betas, transl=transl, pose2rot=False,
    )
    return output.vertices


def motion_init_pose(denorm):
    """從動作 263 特徵的 root 旋轉 + 21 個 rot6d 算出每幀 SMPL 72 維 axis-angle 初始姿勢(warm-start)。"""
    _rrp, _qinv, _q2m, _r6m, _m2aa = _init_pose_fns
    m = denorm                                             # [1, F, 263]
    root_quat, _ = _rrp(m)                                 # [1, F, 4]
    go = _m2aa(_q2m(_qinv(root_quat.reshape(-1, 4))))      # [F, 3]  global orient (同 proxy 慣例)
    rs = 4 + 21 * 3
    rot6d = m[..., rs:rs + 21 * 6].reshape(-1, 21, 6)      # [F, 21, 6]
    bp = _m2aa(_r6m(rot6d))                                # [F, 21, 3]
    Fn = bp.shape[0]
    body = torch.zeros(Fn, 23, 3, device=bp.device, dtype=bp.dtype)
    body[:, :21] = bp
    return torch.cat([go, body.reshape(Fn, 69)], dim=-1)   # [F, 72]


def fit_smpl_own(target_joints, denorm, iters=None):
    """自寫 skeleton→SMPL 擬合：Adam 優化 pose+shape+每幀 transl 貼合 22 關節。
    target_joints:[F,22,3](含全域位移)。回傳 vertices [F,6890,3]。"""
    iters = OWN_ITERS if iters is None else iters
    F = target_joints.shape[0]
    smpl = _smpl_for(F)
    tgt = target_joints.detach()
    init72 = motion_init_pose(denorm).detach() if WARM else torch.zeros(F, 72, device=device)
    go = init72[:, :3].clone().requires_grad_(True)         # global orient
    bp = init72[:, 3:].clone().requires_grad_(True)         # body pose (69)
    transl = tgt[:, 0].clone().requires_grad_(True)         # 每幀 transl，用 root 關節初始化 → 真正位移
    betas = torch.zeros(1, 10, device=device, requires_grad=True)   # 全序列共用體型
    opt = torch.optim.Adam([{"params": [go, bp, transl], "lr": 0.05},
                            {"params": [betas], "lr": 0.01}])
    for _ in range(iters):
        out = smpl(global_orient=go, body_pose=bp, betas=betas.expand(F, 10), transl=transl)
        j = out.joints[:, :22]                              # SMPL 前 22 關節 == HML 22 關節(amass idx 恆等)
        loss = ((j - tgt) ** 2).sum(-1).mean() * 100.0      # 關節位置(含位移)
        if F > 1:                                           # 時序平滑(pose / transl / orient 速度)
            loss = loss + 8.0 * ((bp[1:] - bp[:-1]) ** 2).mean() \
                        + 8.0 * ((transl[1:] - transl[:-1]) ** 2).mean() \
                        + 0.5 * ((go[1:] - go[:-1]) ** 2).mean()
        loss = loss + 1e-3 * (bp ** 2).mean() + 1e-3 * (betas ** 2).mean()   # 弱正則
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        verts = smpl(global_orient=go, body_pose=bp, betas=betas.expand(F, 10), transl=transl).vertices
    return verts


def fit_smpl_high_quality(target_joints, denorm, iters=None):
    """Constrained SMPL fitting with a fixed lean body and stable motion."""
    iters = OWN_ITERS if iters is None else iters
    frame_count = target_joints.shape[0]
    smpl = _smpl_for(frame_count)
    tgt = target_joints.detach()

    init72 = motion_init_pose(denorm).detach() if WARM else torch.zeros(
        frame_count, 72, device=device, dtype=tgt.dtype
    )
    go_init = init72[:, :3].clone()
    bp_init = init72[:, 3:].clone()

    shape_values = list(getattr(VISUAL_CFG, "SMPL_BODY_BETAS", [0.0] * 10))
    if len(shape_values) != 10:
        raise ValueError(f"SMPL_BODY_BETAS needs 10 values, got {len(shape_values)}")
    betas = torch.as_tensor(shape_values, device=device, dtype=tgt.dtype).view(1, 10)
    frame_betas = betas.expand(frame_count, -1)

    # Initialize translation from the shaped pelvis instead of assuming that
    # the SMPL pelvis is exactly at the template origin.
    with torch.no_grad():
        init_out = smpl(
            global_orient=go_init,
            body_pose=bp_init,
            betas=frame_betas,
            transl=torch.zeros_like(tgt[:, 0]),
        )
        transl_init = tgt[:, 0] - init_out.joints[:, 0]

    # Optimize bounded residuals, not unrestricted axis angles. Even if the
    # joint objective is imperfect, the fitter cannot twist far away from the
    # valid pose produced by the motion model.
    go_delta = torch.zeros_like(go_init, requires_grad=True)
    bp_delta = torch.zeros_like(bp_init, requires_grad=True)
    opt = torch.optim.Adam([go_delta, bp_delta], lr=0.05)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    completed_iters = 0
    for step in range(iters):
        go = go_init + 0.20 * torch.tanh(go_delta)
        bp = bp_init + 0.30 * torch.tanh(bp_delta)
        out = smpl(global_orient=go, body_pose=bp, betas=frame_betas, transl=transl_init)
        joints = out.joints[:, :22]
        loss = 100.0 * ((joints - tgt) ** 2).sum(-1).mean()

        # Prefer minimal corrections inside the already bounded search space.
        loss = loss + 0.10 * (bp_delta ** 2).mean()
        loss = loss + 0.05 * (go_delta ** 2).mean()

        # Penalize acceleration, not velocity: this removes visible jitter
        # without damping intentional walking/running speed.
        if frame_count > 2:
            bp_acc = bp[2:] - 2.0 * bp[1:-1] + bp[:-2]
            go_acc = go[2:] - 2.0 * go[1:-1] + go[:-2]
            loss = loss + 4.0 * (bp_acc ** 2).mean()
            loss = loss + 1.0 * (go_acc ** 2).mean()
        elif frame_count > 1:
            loss = loss + 0.25 * ((bp[1:] - bp[:-1]) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([go_delta, bp_delta], max_norm=1.0)
        opt.step()
        completed_iters = step + 1

        # Spend the available latency budget on quality. CUDA work is
        # asynchronous, so synchronize only every four iterations.
        if completed_iters >= 8 and completed_iters % 4 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if time.perf_counter() - started >= OWN_BUDGET:
                break

    with torch.no_grad():
        go = go_init + 0.20 * torch.tanh(go_delta)
        bp = bp_init + 0.30 * torch.tanh(bp_delta)
        final_out = smpl(
            global_orient=go,
            body_pose=bp,
            betas=frame_betas,
            transl=transl_init,
        )
        verts = final_out.vertices
        joint_rmse = ((final_out.joints[:, :22] - tgt) ** 2).sum(-1).mean().sqrt()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(
        f"[mesh-own] frames={frame_count} iters={completed_iters}/{iters} "
        f"fit={elapsed:.2f}s joint_rmse={joint_rmse.item():.4f}",
        flush=True,
    )
    return verts


def _ground(a):
    """a:(F,N,3) → 落地(最低點 y=0) + 起始幀質心 XZ 置中，保留位移。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    a[..., 1] -= a[..., 1].min()
    c0 = a[0].mean(axis=0)
    a[..., 0] -= c0[0]; a[..., 2] -= c0[2]
    return a


# ===== 渲染後製：減少抖動(時間) + 表面扭曲/重疊觀感(空間) =====
def smooth_joints_time(joints, win):
    """關節位置沿時間軸高斯平滑 → 減少逐幀抖動。joints:[L,22,3] → [L,22,3](win=半徑,0=關)。"""
    L = joints.shape[0]
    if win <= 0 or L < 2 * win + 1:
        return joints
    k = 2 * win + 1
    g = torch.arange(k, device=joints.device, dtype=torch.float) - win
    ker = torch.exp(-(g ** 2) / (2 * (win * 0.6 + 1e-6) ** 2))
    ker = (ker / ker.sum()).view(1, 1, k)
    x = joints.permute(1, 2, 0).contiguous().view(-1, 1, L)       # [66,1,L]
    x = torch.nn.functional.pad(x, (win, win), mode="replicate")
    y = torch.nn.functional.conv1d(x, ker)                        # [66,1,L]
    return y.view(joints.shape[1], 3, L).permute(2, 0, 1).contiguous()


_MESH_ADJ = None
def _mesh_adj(N):
    """由 FACES 建無向鄰接(雙向邊 index + 每點度數)，快取一次。"""
    global _MESH_ADJ
    if _MESH_ADJ is None:
        f = torch.as_tensor(FACES, device=device, dtype=torch.long)
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = torch.cat([e, e.flip(1)], 0)                          # 雙向
        deg = torch.zeros(N, device=device).index_add_(
            0, e[:, 0], torch.ones(e.shape[0], device=device)).clamp(min=1.0)
        _MESH_ADJ = (e[:, 0].contiguous(), e[:, 1].contiguous(), deg)
    return _MESH_ADJ


def taubin_smooth(V, iters, lam=0.53, mu=-0.55):
    """Taubin λ|μ 網格平滑(收縮免疫)：平順化 LBS 表面、淡化關節扭曲/重疊觀感。V:[F,N,3] → [F,N,3]。"""
    if iters <= 0:
        return V
    idx_i, idx_j, deg = _mesh_adj(V.shape[1])
    dinv = (1.0 / deg).view(1, -1, 1)
    def lap(P):
        nb = torch.zeros_like(P)
        nb.index_add_(1, idx_i, P[:, idx_j])
        return nb * dinv - P
    for _ in range(iters):
        V = V + lam * lap(V)
        V = V + mu * lap(V)
    return V

# session cache: motion id -> {"mids": exact generated token list (per-scale), "len": frames}
# 保存生成當下的『原始 token』本身，edit 時直接回餵當 source，不做 decode→encode 重量化（無精度漂移）。
CACHE = {}
LOCK = threading.Lock()


@torch.no_grad()
def _run(text, length, source_id, p):
    """One generate/edit call. Returns (motion_id, joints[list])."""
    with LOCK:
        has_source = source_id is not None
        if has_source:
            if source_id not in CACHE:
                raise KeyError(f"unknown source_id {source_id}")
            entry = CACHE[source_id]
            source_code_idx = [s.to(device) for s in entry["mids"]]   # 原始 token（list of per-scale idx），與畫面動作完全對齊
            Ts = entry["len"]
            src_len = torch.tensor([Ts], device=device).long()
            source_m_lens = src_len // UNIT
            m_len = src_len.clone()                       # edit keeps source length
        else:
            source_code_idx, source_m_lens = None, None
            L = int(np.clip(length, 40, MAXL))
            m_len = torch.tensor([L], device=device).long()

        has_src_t = torch.tensor([1 if has_source else 0], device=device).long()
        kwargs = dict(
            timesteps=int(p["time_steps"]),
            cond_scale=float(p["cond_scale"]),
            source_cond_scale=float(p.get("source_scale", 1.0)),
            source_m_lens=source_m_lens,
            temperature=float(p["temperature"]),
            topk_filter_thres=float(p["topk"]),
            gsample=bool(p["gumbel"]),
        )
        # 動態 source in-painting：edit 時鎖住這比例的 source token 當畫布 → 輸出貼著來源動作(0=關閉)
        keep_ratio = float(p.get("source_keep_ratio", 0.0)) if has_source else 0.0
        mids = trans.generate(source_code_idx, [text], m_len // UNIT, has_src_t, t_drop=0,
                              source_keep_ratio=keep_ratio, **kwargs)

        # The VQ decoder returns its fixed 196-frame canvas; the requested
        # duration is carried separately in m_len, so crop before rendering.
        pred = vq_model.forward_decoder(mids, m_len.clone())[0, :int(m_len.item())]
        pred = pred.detach().cpu().numpy()                                                # [L, 263] normalized
        mid = uuid.uuid4().hex[:8]
        # 保存這次產出的『原始 token』本身（含被編輯後的結果），供之後的 edit 直接當 source
        CACHE[mid] = {"mids": [s.detach().cpu() for s in mids], "len": int(m_len.item())}
        denorm = denorm_motion(pred, mean, std, pred.shape[0], device)                    # [1, L, 263]
        joints_t = motion_to_joints(denorm, cfg.data.joint_num)                           # [L, 22, 3]
        joints_t = smooth_joints_time(joints_t, SMOOTH_WIN)                               # 減少抖動(時間平滑)
        if MESH_OK:
            if MESH_MODE == "lbs":
                verts_t = lbs_vertices(joints_t)                                          # position-driven LBS(最快 + 真位移)
                verts_t = taubin_smooth(verts_t, TAUBIN_ITERS)                            # 平順表面、淡化扭曲/重疊觀感
            elif MESH_MODE == "own":
                verts_t = ik_smpl_vertices(joints_t)
            elif MESH_MODE == "own_legacy":
                verts_t = fit_smpl_own(joints_t, denorm)                                  # 自寫擬合(精準 + 真位移)
            elif MESH_MODE == "fit":
                init_pose = motion_init_pose(denorm) if WARM else None
                verts_t, _ = fit_vertices(joints_t, VISUAL_CFG, VIS_CFG, device, init_pose=init_pose)
            else:
                verts_t, _, _ = proxy_vertices(denorm, joints_t, SMPL_MODEL, VIS_CFG, device)
            return mid, "mesh", _ground(verts_t.detach().cpu().numpy())
        return mid, "joints", _ground(joints_t.detach().cpu().numpy())


def _pack(mid, kind, arr):
    """二進位回傳（避免數十 MB JSON）：float32 [tag,F,V,Fc,fps] + faces + verts。"""
    F, V = int(arr.shape[0]), int(arr.shape[1])
    if kind == "mesh":
        head = np.array([1.0, F, V, FACES.shape[0], FPS], np.float32)
        blob = np.concatenate([head, FACES.astype(np.float32).ravel(), arr.ravel()])
    else:
        head = np.array([0.0, F, V, 0, FPS], np.float32)
        blob = np.concatenate([head, arr.ravel()])
    resp = Response(blob.astype(np.float32).tobytes(), mimetype="application/octet-stream")
    resp.headers["X-Motion-Id"] = mid
    return resp


app = Flask(__name__)

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Expose-Headers"] = "X-Motion-Id"   # 讓瀏覽器讀得到動作 id
    return resp

@app.route("/health")
def health():
    return jsonify(ok=True, device=str(device), ckpt=CKPT, use_ema=USE_EMA,
                   mesh=MESH_OK, mesh_mode=(MESH_MODE if MESH_OK else None), fps=FPS, cached=len(CACHE))

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return ("", 204)
    d = request.get_json(force=True)
    mid, kind, arr = _run(d["text"], int(d.get("length", 120)), None, d["params"])
    return _pack(mid, kind, arr)

@app.route("/edit", methods=["POST", "OPTIONS"])
def edit():
    if request.method == "OPTIONS":
        return ("", 204)
    d = request.get_json(force=True)
    mid, kind, arr = _run(d["text"], 0, d["source_id"], d["params"])
    return _pack(mid, kind, arr)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=False)
