"""VQMotion demo backend.

Loads the VQMotion checkpoint, frozen HRVQ-VAE, and T5 encoder once. The
`own` path converts generated 22-joint motion to fixed-shape male SMPL-X
vertices; the browser renders the returned surface with Three.js.

Run (in the WSL / conda env that has torch + the project deps):
    pip install flask                       # once
    cd /mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/Application
    CKPT=best.tar python server.py          # serves on http://localhost:5000

Env overrides: TRANS_CFG, CKPT, PORT, USE_EMA, MESH_MODE (own or joints).
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
    resolve_trans_ckpt_path, load_trusted_local_checkpoint,
)

TRANS_CFG = os.environ.get("TRANS_CFG", pjoin(ROOT, "configs", "train_vqmotion_hml.yaml"))
CKPT      = os.environ.get("CKPT", "best.tar")
PORT      = int(os.environ.get("PORT", "5000"))
USE_EMA   = os.environ.get("USE_EMA", "1") != "0"   # eval 設定 use_ema:True；用 USE_EMA=0 可載原始權重對照


def apply_ema(model, ckpt_path, device):
    """把 checkpoint 裡的 EMA 影子權重覆蓋到模型（對齊 eval 的 use_ema:True）。"""
    ck = load_trusted_local_checkpoint(ckpt_path, device)
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
DEFAULT_DELTA_BETA = float(getattr(trans, "delta_beta", 0.3))
if hasattr(trans, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
    trans.set_vq_codebook(vq_model.quantizer.codebook)
UNIT = cfg.data.unit_length
DIM  = cfg.data.dim_pose
MAXL = cfg.data.max_motion_length
FPS  = int(getattr(cfg.data, "fps", 20))
print(f"[boot] ready on {device} · unit={UNIT} · dim={DIM} · fps={FPS}", flush=True)

# --- body render mode ---
# own: official Application path: minimum-twist IK + male SMPL-X skinning.
# joints: return the generated 22-joint skeleton without loading SMPL assets.
MESH_MODE = os.environ.get("MESH_MODE", "own").lower()
if MESH_MODE not in {"own", "joints"}:
    raise ValueError("MESH_MODE must be 'own' or 'joints'")
TORSO_POSE_CORRECTIVE_SCALE = float(
    np.clip(float(os.environ.get("TORSO_POSE_CORRECTIVE_SCALE", "0.35")), 0.0, 1.0)
)
TORSO_PROFILE_FLATTEN = float(
    np.clip(float(os.environ.get("TORSO_PROFILE_FLATTEN", "1.0")), 0.0, 1.5)
)
VLINE_TAPER = float(
    np.clip(float(os.environ.get("VLINE_TAPER", "0.05")), 0.0, 0.10)
)
ATHLETIC_SHOULDER_GAIN = float(
    np.clip(float(os.environ.get("ATHLETIC_SHOULDER_GAIN", "0.025")), 0.0, 0.05)
)
SMOOTH_WIN = int(os.environ.get("SMOOTH_WIN", "0"))
IK_SMOOTH_WIN = int(os.environ.get("IK_SMOOTH_WIN", "1"))
IK_JITTER_FILTER = os.environ.get("IK_JITTER_FILTER", "1") != "0"
POSE_REFINE_ITERS = max(0, int(os.environ.get("POSE_REFINE_ITERS", "4")))
POSE_REFINE_BUDGET = max(
    0.0, float(os.environ.get("POSE_REFINE_BUDGET", "0.45"))
)
NEED_SMPL = MESH_MODE == "own"
MESH_OK = False
FACES = np.zeros((0, 3), np.int64)
VISUAL_CFG = LBS = None
SMPL_DIR = None
if NEED_SMPL:
    try:
        from smplx.lbs import (
            batch_rigid_transform as _batch_rigid_transform,
            lbs as _raw_lbs,
        )
        from utils.rotation_conversions import (
            quaternion_to_matrix as _q2m, matrix_to_quaternion as _m2q,
            matrix_to_axis_angle as _m2aa,
            axis_angle_to_matrix as _aa2m,
        )
        VISUAL_CFG = load_config(pjoin(ROOT, "utils", "visual_config.yaml"))
        SMPL_DIR = VISUAL_CFG.SMPL_MODEL_DIR
        male_path = pjoin(SMPL_DIR, "smplx", "SMPLX_MALE.npz")
        if not os.path.isfile(male_path):
            raise FileNotFoundError(f"Male SMPL-X model not found: {male_path}")
        with np.load(male_path, allow_pickle=True) as male_data:
            FACES = np.asarray(male_data["f"], dtype=np.int64)
            male_v_template = torch.as_tensor(
                male_data["v_template"], device=device, dtype=torch.float32
            )
            male_shapedirs = torch.as_tensor(
                male_data["shapedirs"][:, :, :10], device=device, dtype=torch.float32
            )
            posedirs_np = male_data["posedirs"]
            male_posedirs = torch.as_tensor(
                posedirs_np.reshape(-1, posedirs_np.shape[-1]).T,
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

        shape_values = list(getattr(VISUAL_CFG, "SMPL_MALE_BETAS", [0.0] * 10))
        if len(shape_values) != 10:
            raise ValueError(f"SMPL_MALE_BETAS needs 10 values, got {len(shape_values)}")
        male_betas = torch.as_tensor(shape_values, device=device, dtype=torch.float32)
        v_template = male_v_template + torch.einsum(
            "vcn,n->vc", male_shapedirs, male_betas
        )

        # Joint positions do not constrain the torso surface. Keep the accepted
        # athletic profile and damp torso pose correctives without a female
        # template or any per-request shape optimization.
        torso_weight = lbs_w[:, [0, 3, 6, 9, 12]].sum(dim=-1)
        torso_blend = ((torso_weight - 0.20) / 0.60).clamp(0.0, 1.0)
        torso_blend = torso_blend * torso_blend * (3.0 - 2.0 * torso_blend)
        frontness = ((v_template[:, 2] + 0.01) / 0.12).clamp(0.0, 1.0)
        frontness = frontness * frontness * (3.0 - 2.0 * frontness)
        abdomen = torch.exp(-0.5 * ((v_template[:, 1] + 0.28) / 0.13) ** 2)
        chest = torch.exp(-0.5 * ((v_template[:, 1] + 0.02) / 0.10) ** 2)
        profile_offset = TORSO_PROFILE_FLATTEN * (
            0.028 * abdomen + 0.010 * chest
        ) * frontness * torso_blend
        v_template[:, 2] = v_template[:, 2] - profile_offset

        vline = torch.exp(-0.5 * ((v_template[:, 1] + 0.31) / 0.09) ** 2)
        v_template[:, 0] = v_template[:, 0] * (
            1.0 - VLINE_TAPER * vline * torso_blend
        )
        upper_weight = lbs_w[:, [9, 12, 13, 14, 16, 17]].sum(dim=-1)
        upper_blend = ((upper_weight - 0.15) / 0.65).clamp(0.0, 1.0)
        upper_blend = upper_blend * upper_blend * (3.0 - 2.0 * upper_blend)
        shoulder_height = torch.exp(
            -0.5 * ((v_template[:, 1] - 0.055) / 0.105) ** 2
        )
        v_template[:, 0] = v_template[:, 0] * (
            1.0 + ATHLETIC_SHOULDER_GAIN * upper_blend * shoulder_height
        )
        corrective_scale = 1.0 - (
            1.0 - TORSO_POSE_CORRECTIVE_SCALE
        ) * torso_blend
        male_posedirs = male_posedirs * corrective_scale.repeat_interleave(
            3
        ).unsqueeze(0)

        J_rest = J_regressor @ v_template
        joint_count = parents.shape[0]
        children = [[] for _ in range(joint_count)]
        for joint_idx in range(1, joint_count):
            if joint_idx < 22:
                children[int(parents[joint_idx])].append(joint_idx)
        rest_dir = torch.zeros(joint_count, 3, device=device)
        for joint_idx, child_ids in enumerate(children):
            if child_ids:
                direction = (J_rest[child_ids] - J_rest[joint_idx]).mean(0)
                rest_dir[joint_idx] = direction / direction.norm().clamp(min=1e-8)

        raw_model = dict(
            v_template=v_template,
            shapedirs=male_shapedirs,
            posedirs=male_posedirs,
            J_regressor=J_regressor,
            lbs_w=lbs_w,
            parents=parents,
            betas=torch.zeros_like(male_betas),
        )
        LBS = dict(
            parents=parents,
            J_rest=J_rest,
            children=children,
            rest_dir=rest_dir,
            raw_model=raw_model,
        )
        MESH_OK = True
        print(
            "[boot] SMPL mesh ON · male-only minimum-twist IK "
            f"· faces={FACES.shape[0]} · torso_corrective="
            f"{TORSO_POSE_CORRECTIVE_SCALE:.2f} · pose_refine="
            f"{POSE_REFINE_ITERS}",
            flush=True,
        )
    except Exception as e:
        raise RuntimeError("Male SMPL-X mesh initialization failed") from e
else:
    print("[boot] mesh mode = joints", flush=True)


# ===== minimum-twist IK and fixed-male SMPL-X skinning =====
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


def smooth_rotation_matrices(rotations, radius):
    """Temporally smooth rotations in quaternion space without deforming vertices."""
    frame_count, joint_count = rotations.shape[:2]
    if radius <= 0 or frame_count < 3:
        return rotations

    quaternions = _m2q(rotations)
    aligned = [quaternions[0]]
    max_step = 0.50
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

    Joint directions only determine valid local SMPL rotations; the official
    SMPL-X skinning function produces the final surface without bone stretching.
    """
    rest_joints = LBS["J_rest"]
    parents = LBS["parents"]
    children = LBS["children"]
    rest_dir = LBS["rest_dir"]
    joint_count = int(parents.shape[0])
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

    raw_model = LBS["raw_model"]
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
    base_vertices = vertices + transl[:, None]
    base_joints = joints + transl[:, None]
    if POSE_REFINE_ITERS > 0:
        return refine_male_pose(
            target_joints,
            local_rot,
            base_vertices,
            base_joints,
            raw_model,
        )
    return base_vertices


def refine_male_pose(
    target_joints,
    base_rotations,
    base_vertices,
    base_joints,
    raw_model,
):
    """Apply a bounded pose-only refinement and keep the fixed male shape."""
    frame_count = target_joints.shape[0]
    if frame_count < 3:
        return base_vertices

    # These joints have observable child-bone directions in the HML skeleton.
    # End effectors remain frozen because their twist cannot be recovered from
    # joint positions and is the main source of visibly distorted limbs.
    refined_joint_ids = torch.as_tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19],
        device=target_joints.device,
        dtype=torch.long,
    )
    max_degrees = torch.as_tensor(
        [2.0, 3.0, 3.0, 2.0, 4.0, 4.0, 2.0, 3.0, 3.0,
         2.5, 2.0, 2.5, 2.5, 3.0, 3.0, 4.0, 4.0],
        device=target_joints.device,
        dtype=target_joints.dtype,
    )
    max_radians = torch.deg2rad(max_degrees).view(1, -1, 1)
    target = target_joints.detach()
    base_rotations = base_rotations.detach()
    rest_joints = LBS["J_rest"].detach().view(
        1, -1, 3
    ).expand(frame_count, -1, -1)
    parents = raw_model["parents"]

    control_stride = 2
    control_count = max(2, (frame_count - 1 + control_stride - 1) // control_stride + 1)
    delta_params = torch.zeros(
        control_count,
        refined_joint_ids.numel(),
        3,
        device=target_joints.device,
        dtype=target_joints.dtype,
        requires_grad=True,
    )
    first_moment = torch.zeros_like(delta_params)
    second_moment = torch.zeros_like(delta_params)
    started = time.perf_counter()

    def interpolate_delta(params):
        signal = params.permute(1, 2, 0).reshape(
            1, refined_joint_ids.numel() * 3, control_count
        )
        signal = torch.nn.functional.interpolate(
            signal,
            size=frame_count,
            mode="linear",
            align_corners=True,
        )
        return signal.reshape(
            refined_joint_ids.numel(), 3, frame_count
        ).permute(2, 0, 1)

    def pose_from_delta(params):
        delta_axis_angle = torch.tanh(interpolate_delta(params)) * max_radians
        delta_rotation = _aa2m(delta_axis_angle)
        candidate = base_rotations.clone()
        candidate[:, refined_joint_ids] = torch.matmul(
            base_rotations[:, refined_joint_ids], delta_rotation
        )
        return candidate, delta_axis_angle

    def posed_joints(candidate):
        joints, _ = _batch_rigid_transform(
            candidate,
            rest_joints,
            parents,
            dtype=target_joints.dtype,
        )
        translation = target[:, :1] - joints[:, :1]
        return joints + translation

    base_visible_joints = base_joints[:, :22]
    base_rmse = torch.linalg.norm(
        base_visible_joints - target, dim=-1
    ).mean()
    target_acceleration = (
        target[2:] - 2.0 * target[1:-1] + target[:-2]
    )

    def acceleration_residual(joints):
        acceleration = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
        return torch.linalg.norm(
            acceleration - target_acceleration, dim=-1
        ).mean()

    base_temporal_error = acceleration_residual(base_visible_joints)
    best_params = delta_params.detach().clone()
    best_rmse = base_rmse.detach().clone()
    best_temporal_error = base_temporal_error.detach().clone()
    best_score = best_rmse + 0.25 * best_temporal_error

    with torch.enable_grad():
        for refine_step in range(1, POSE_REFINE_ITERS + 1):
            candidate, delta_axis_angle = pose_from_delta(delta_params)
            predicted = posed_joints(candidate)[:, :22]

            joint_loss = torch.nn.functional.smooth_l1_loss(
                predicted,
                target,
                beta=0.015,
            )
            pred_bones = predicted[:, 1:] - predicted[:, parents[1:22]]
            target_bones = target[:, 1:] - target[:, parents[1:22]]
            pred_bones = pred_bones / pred_bones.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            target_bones = target_bones / target_bones.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            direction_loss = (
                1.0 - (pred_bones * target_bones).sum(dim=-1)
            ).mean()

            normalized_delta = delta_axis_angle / max_radians
            trust_loss = normalized_delta.square().mean()
            velocity_loss = (
                delta_axis_angle[1:] - delta_axis_angle[:-1]
            ).square().mean()
            if frame_count > 2:
                acceleration_loss = (
                    delta_axis_angle[2:]
                    - 2.0 * delta_axis_angle[1:-1]
                    + delta_axis_angle[:-2]
                ).square().mean()
            else:
                acceleration_loss = delta_axis_angle.new_zeros(())

            loss = (
                4.0 * joint_loss
                + 0.25 * direction_loss
                + 0.02 * trust_loss
                + 2.0 * velocity_loss
                + 4.0 * acceleration_loss
            )
            gradient = torch.autograd.grad(loss, delta_params)[0]
            gradient_norm = torch.linalg.vector_norm(gradient)
            gradient = gradient * torch.clamp(
                0.5 / gradient_norm.clamp_min(1e-8), max=1.0
            )
            with torch.no_grad():
                first_moment.mul_(0.9).add_(gradient, alpha=0.1)
                second_moment.mul_(0.999).addcmul_(
                    gradient, gradient, value=0.001
                )
                first_unbiased = first_moment / (1.0 - 0.9 ** refine_step)
                second_unbiased = second_moment / (
                    1.0 - 0.999 ** refine_step
                )
                delta_params.addcdiv_(
                    first_unbiased,
                    second_unbiased.sqrt().add_(1e-8),
                    value=-0.15,
                )

            with torch.no_grad():
                step_rotations, _ = pose_from_delta(delta_params)
                step_joints = posed_joints(step_rotations)[:, :22]
                step_rmse = torch.linalg.norm(
                    step_joints - target, dim=-1
                ).mean()
                step_temporal_error = acceleration_residual(step_joints)
                step_score = step_rmse + 0.25 * step_temporal_error
                temporally_safe = (
                    step_temporal_error
                    <= base_temporal_error * 1.10 + 5e-4
                )
                if bool(temporally_safe and step_score < best_score):
                    best_params.copy_(delta_params.detach())
                    best_rmse.copy_(step_rmse)
                    best_temporal_error.copy_(step_temporal_error)
                    best_score.copy_(step_score)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if time.perf_counter() - started >= POSE_REFINE_BUDGET:
                break

    with torch.no_grad():
        refined_rotations, _ = pose_from_delta(best_params)
        refined_joints = posed_joints(refined_rotations)
        refined_rmse = torch.linalg.norm(
            refined_joints[:, :22] - target, dim=-1
        ).mean()
        refined_temporal_error = acceleration_residual(
            refined_joints[:, :22]
        )
        accept = (
            refined_rmse + 1e-5 < base_rmse
            and refined_temporal_error
            <= base_temporal_error * 1.10 + 5e-4
        )
        if not bool(accept):
            print(
                "[mesh-refine] rejected; preserving original fixed-male mesh "
                f"rmse={base_rmse.item():.4f}->{refined_rmse.item():.4f} "
                "temporal_error="
                f"{base_temporal_error.item():.4f}->"
                f"{refined_temporal_error.item():.4f}",
                flush=True,
            )
            return base_vertices

        betas = raw_model["betas"].view(1, 10).expand(frame_count, -1)
        vertices, joints = _raw_lbs(
            betas,
            refined_rotations,
            raw_model["v_template"],
            raw_model["shapedirs"],
            raw_model["posedirs"],
            raw_model["J_regressor"],
            raw_model["parents"],
            raw_model["lbs_w"],
            pose2rot=False,
        )
        translation = target[:, :1] - joints[:, :1]
        elapsed = time.perf_counter() - started
        print(
            "[mesh-refine] accepted fixed-male pose refinement "
            f"rmse={base_rmse.item():.4f}->{refined_rmse.item():.4f} "
            "temporal_error="
            f"{base_temporal_error.item():.4f}->"
            f"{refined_temporal_error.item():.4f} "
            f"time={elapsed:.3f}s",
            flush=True,
        )
        return vertices + translation


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


def suppress_joint_jitter(joints):
    """Remove frame-to-frame jitter without damping smooth motion trends.

    The seven-point, second-order Savitzky-Golay kernel preserves constant
    velocity and acceleration. Unlike a Gaussian moving average, it therefore
    does not make deliberate motion slower or more conservative.
    """
    frame_count = joints.shape[0]
    if frame_count < 7:
        return joints
    kernel = joints.new_tensor([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0
    signal = joints.permute(1, 2, 0).reshape(-1, 1, frame_count)
    signal = torch.nn.functional.pad(signal, (3, 3), mode="reflect")
    filtered = torch.nn.functional.conv1d(signal, kernel.view(1, 1, 7))
    return filtered.view(joints.shape[1], 3, frame_count).permute(2, 0, 1).contiguous()


# session cache: motion id -> {"mids": exact generated token list (per-scale), "len": frames}
# 保存生成當下的『原始 token』本身，edit 時直接回餵當 source，不做 decode→encode 重量化（無精度漂移）。
CACHE = {}
LOCK = threading.Lock()


@torch.no_grad()
def _run(text, length, source_id, p):
    """Run one generation/edit request and return its render representation."""
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
            # Clients may choose an edited duration explicitly. A missing or
            # non-positive value preserves the source duration.
            L = Ts if length is None or int(length) <= 0 else int(np.clip(length, 40, MAXL))
            m_len = torch.tensor([L], device=device).long()
        else:
            source_code_idx, source_m_lens = None, None
            L = int(np.clip(length, 40, MAXL))
            m_len = torch.tensor([L], device=device).long()

        has_src_t = torch.tensor([1 if has_source else 0], device=device).long()
        delta_beta = float(p.get("delta_beta", DEFAULT_DELTA_BETA))
        if not np.isfinite(delta_beta) or delta_beta < 0.0:
            raise ValueError("delta_beta must be a finite non-negative value")
        trans.delta_beta = delta_beta
        kwargs = dict(
            timesteps=int(p["time_steps"]),
            cond_scale=float(p["cond_scale"]),
            source_cond_scale=float(p.get("source_scale", 1.0)),
            source_m_lens=source_m_lens,
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
        joints_t = smooth_joints_time(joints_t, SMOOTH_WIN)
        if MESH_OK and IK_JITTER_FILTER:
            joints_t = suppress_joint_jitter(joints_t)
        if MESH_OK:
            verts_t = ik_smpl_vertices(joints_t)
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
                   mesh=MESH_OK, mesh_mode=MESH_MODE,
                   delta_latent_mode=str(getattr(trans, "delta_latent_mode", "unknown")),
                   delta_beta=float(getattr(trans, "delta_beta", DEFAULT_DELTA_BETA)),
                   fps=FPS, cached=len(CACHE))

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
    mid, kind, arr = _run(d["text"], int(d.get("length", 0)), d["source_id"], d["params"])
    return _pack(mid, kind, arr)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=False)
