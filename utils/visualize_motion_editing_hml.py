import os
if os.name != "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import copy
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from os.path import join as pjoin

from configs.load_config import load_config
from utils.fixseeds import fixseed
from utils.motion_process_bvh import recover_pos_from_ric, recover_root_rot_pos
from utils.common.quaternion import qinv
from utils.rotation_conversions import quaternion_to_matrix, rotation_6d_to_matrix
try:
    from utils.smpl import SMPL
except ModuleNotFoundError as e:
    if e.name != "smplx":
        raise
    SMPL = None


HML_KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
]
HML_EDGES = [edge for chain in HML_KINEMATIC_CHAIN for edge in zip(chain[:-1], chain[1:])]
TORSO_EDGES = {(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)}
SMPL_KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [12, 13, 16, 18, 20],
    [12, 14, 17, 19, 21],
]


def import_renderer():
    try:
        import trimesh
        import pyrender
        from pyrender.constants import RenderFlags
        return trimesh, pyrender, RenderFlags
    except Exception as e:
        raise ImportError(
            "SMPL mesh visualization needs trimesh + pyrender + a valid OpenGL backend. "
            "Install the visualization dependencies instead of falling back to joint spheres."
        ) from e


def load_mean_std(cfg):
    candidates = [
        (pjoin(cfg.data.root_dir, 'meta_data', 'mean.npy'), pjoin(cfg.data.root_dir, 'meta_data', 'std.npy')),
        (pjoin(cfg.data.root_dir, 'HumanML3D', 'Mean.npy'), pjoin(cfg.data.root_dir, 'HumanML3D', 'Std.npy')),
        (pjoin(cfg.data.root_dir, 'HumanML3D', 'mean.npy'), pjoin(cfg.data.root_dir, 'HumanML3D', 'std.npy')),
        (pjoin(os.path.dirname(cfg.data.feat_dir), 'Mean.npy'), pjoin(os.path.dirname(cfg.data.feat_dir), 'Std.npy')),
        (pjoin(os.path.dirname(cfg.data.feat_dir), 'mean.npy'), pjoin(os.path.dirname(cfg.data.feat_dir), 'std.npy')),
    ]
    for mean_path, std_path in candidates:
        if os.path.exists(mean_path) and os.path.exists(std_path):
            return np.load(mean_path), np.load(std_path)
    raise FileNotFoundError("Cannot find mean/std normalization files.")


def load_vq_model(cfg, vq_cfg, device):
    from SnapMogen_model.vq.rvq_model import HRVQVAE

    vq_model = HRVQVAE(vq_cfg, vq_cfg.data.dim_pose, vq_cfg.model.down_t, vq_cfg.model.stride_t,
                       vq_cfg.model.width, vq_cfg.model.depth, vq_cfg.model.dilation_growth_rate,
                       vq_cfg.model.vq_act, vq_cfg.model.use_attn, vq_cfg.model.vq_norm)
    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model', cfg.vq_ckpt),
                      map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    return vq_model.to(device).eval()


def resolve_trans_ckpt_path(cfg, ckpt_name):
    model_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'VA_motion', 'model')
    ckpt_name = str(ckpt_name)
    if os.path.isabs(ckpt_name) or ckpt_name.startswith(("/", "\\")):
        candidates = [ckpt_name]
    else:
        candidates = [pjoin(model_dir, ckpt_name)]

    for path in candidates:
        if os.path.exists(path):
            return path

    available = []
    if os.path.isdir(model_dir):
        available = sorted(
            os.path.relpath(pjoin(root, name), model_dir).replace("\\", "/")
            for root, _, files in os.walk(model_dir)
            for name in files
            if name.endswith(".tar")
        )
    raise FileNotFoundError(
        f"Cannot find transformer checkpoint '{ckpt_name}'. Tried: {candidates}. "
        f"Available in {model_dir}: {available}"
    )


def load_trans_model(cfg, vq_cfg, ckpt_name, device):
    from model import VAMotion

    cfg.vq = vq_cfg.quantizer
    cfg.vq.nb_code = vq_cfg.quantizer.nb_code
    ckpt_path = resolve_trans_ckpt_path(cfg, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # old checkpoints do not have the task token, keep visual results faithful to the ckpt
    if "task_embed.weight" not in ckpt["training_model"]:
        cfg.model.use_task_token = False

    trans = VAMotion(cfg=cfg, device=device, full_length=cfg.data.max_motion_length // cfg.data.unit_length)
    missing, unexpected = trans.load_state_dict(ckpt["training_model"], strict=False)
    old_prefixes = (
        "text_delta_encoder.",
        "condition_encoder.",
        "edit_map_head.",
        "part_gate_mlp.",
        "part_cond_mlp.",
        "part_text_mlp.",
        "part_source_mlp.",
        "edit_loc_head.",
        "task_embed.",
        "text_delta_scale",
        "source_delta_scale",
    )
    old_keys = {"part_joint_mask"}
    ignored_unexpected = [k for k in unexpected if k in old_keys or k.startswith(old_prefixes)]
    unexpected = [k for k in unexpected if k not in old_keys and not k.startswith(old_prefixes)]
    assert len(unexpected) == 0, f"unexpected keys from ckpt: {unexpected[:5]}"
    if missing:
        print(f"missing keys from ckpt: {missing[:5]}")
    if ignored_unexpected:
        print(f"ignored old checkpoint keys: {ignored_unexpected[:5]} ({len(ignored_unexpected)} total)")
    print(f"Loaded transformer: {ckpt_path}, epoch={ckpt.get('epoch', '?')}")
    return trans.to(device).eval()


def build_dataset(cfg, mean, std, split, motionfix_start_id):
    from utils.get_opt import get_opt
    from dataset.HumanML3D_dataset import HML3DMotionEditDataset

    opt_path = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'Comp_v6_KLD005', 'opt.txt')
    opt = get_opt(opt_path, cfg.exp.device)
    opt.root_dir = cfg.data.root_dir
    opt.max_motion_length = cfg.data.max_motion_length
    opt.unit_length = cfg.data.unit_length
    opt.motion_dir = pjoin(cfg.data.root_dir, "HumanML3D", "new_joint_vecs")
    opt.text_dir = pjoin(cfg.data.root_dir, "HumanML3D", "texts")
    split_file = pjoin(cfg.data.root_dir, "HumanML3D", f"{split}.txt")
    return HML3DMotionEditDataset(opt, mean, std, split_file, motionfix_start_id=motionfix_start_id)


def select_indices(dataset, mode, start_index, num_samples, random_select=False, seed=0):
    mode = {"all": "all", "gen": "generation", "edit": "editing",
            "both": "all", "generation": "generation", "editing": "editing"}.get(mode, mode)
    selected = []
    for idx, name in enumerate(dataset.name_list):
        task_type = "editing" if dataset.data_dict[name][4] else "generation"
        if mode != "all" and task_type != mode:
            continue
        selected.append(idx)
    if random_select:
        rng = np.random.default_rng(int(seed))
        if len(selected) <= num_samples:
            return selected
        return rng.choice(selected, size=num_samples, replace=False).tolist()
    return selected[start_index:start_index + num_samples]


def denorm_motion(motion, mean, std, length, device):
    if not torch.is_tensor(motion):
        motion = torch.from_numpy(motion)
    mean_t = torch.from_numpy(mean[:motion.shape[-1]]).float()
    std_t = torch.from_numpy(std[:motion.shape[-1]]).float()
    motion = motion[:length].float().cpu() * std_t + mean_t
    return motion.unsqueeze(0).to(device)


def motion_to_joints(motion_denorm, joints_num=22):
    joints = recover_pos_from_ric(motion_denorm, joints_num=joints_num - 1, hml3d=True)
    return joints[0]


def similarity_align_points(points, source_joints, target_joints, use_scale=True, eps=1e-8):
    src_mean = source_joints.mean(dim=1, keepdim=True)
    tgt_mean = target_joints.mean(dim=1, keepdim=True)
    src_centered = source_joints - src_mean
    tgt_centered = target_joints - tgt_mean

    cov = torch.matmul(src_centered.transpose(1, 2), tgt_centered)
    u, s, vh = torch.linalg.svd(cov)
    r = torch.matmul(u, vh)

    det = torch.linalg.det(r)
    if torch.any(det < 0):
        fix = torch.ones((r.shape[0], 3), device=r.device, dtype=r.dtype)
        fix[det < 0, -1] = -1
        r = torch.matmul(u * fix.unsqueeze(1), vh)
        s = s * fix

    if use_scale:
        src_var = (src_centered ** 2).sum(dim=(1, 2)).clamp_min(eps)
        scale = (s.sum(dim=1) / src_var).view(-1, 1, 1)
    else:
        scale = torch.ones((points.shape[0], 1, 1), device=points.device, dtype=points.dtype)

    aligned_points = scale * torch.matmul(points - src_mean, r) + tgt_mean
    aligned_joints = scale * torch.matmul(source_joints - src_mean, r) + tgt_mean
    return aligned_points, aligned_joints


def mean_joint_error(source_joints, target_joints):
    return torch.linalg.norm(source_joints - target_joints, dim=-1).mean().item()


def warp_vertices_to_joints(vertices, fitted_joints, target_joints, smpl_model, eps=1e-8):
    if not hasattr(smpl_model, "lbs_weights"):
        return vertices
    weights = smpl_model.lbs_weights[:, :target_joints.shape[1]].to(device=vertices.device, dtype=vertices.dtype)
    if smpl_model.lbs_weights.shape[1] >= 24 and target_joints.shape[1] >= 22:
        extra_weights = smpl_model.lbs_weights[:, 22:24].to(device=vertices.device, dtype=vertices.dtype)
        weights[:, 20] = weights[:, 20] + extra_weights[:, 0]
        weights[:, 21] = weights[:, 21] + extra_weights[:, 1]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    joint_delta = target_joints - fitted_joints
    vertex_delta = torch.einsum("vj,tjc->tvc", weights, joint_delta)
    return vertices + vertex_delta


def resample_sequence(sequence, target_len):
    if sequence.shape[0] == target_len:
        return sequence
    flat = sequence.reshape(sequence.shape[0], -1).transpose(0, 1).unsqueeze(0)
    flat = F.interpolate(flat, size=target_len, mode="linear", align_corners=True)
    return flat.squeeze(0).transpose(0, 1).reshape((target_len,) + tuple(sequence.shape[1:]))


def motion_to_proxy_vertices(motion_denorm, target_joints, smpl_model, vis_cfg, device):
    # HML3D 263 layout: root(4) + ric(21*3) + rot6d(21*6) + vel(22*3) + foot(4)
    b, t, _ = motion_denorm.shape
    root_quat, root_pos = recover_root_rot_pos(motion_denorm)
    global_orient = quaternion_to_matrix(qinv(root_quat.reshape(-1, 4)))

    rot_start = 4 + 21 * 3
    rot6d = motion_denorm[..., rot_start:rot_start + 21 * 6].reshape(b * t, 21, 6)
    rot_mats = rotation_6d_to_matrix(rot6d)

    eye = torch.eye(3, device=device, dtype=motion_denorm.dtype).view(1, 1, 3, 3)
    body_pose = eye.repeat(b * t, 23, 1, 1)
    body_pose[:, :21] = rot_mats

    smpl_out = smpl_model(body_pose=body_pose, global_orient=global_orient)
    vertices = smpl_out["vertices"] + root_pos.reshape(-1, 1, 3)
    smpl_joints = smpl_out["smpl"][:, :target_joints.shape[1]] + root_pos.reshape(-1, 1, 3)
    target_flat = target_joints.reshape(b * t, target_joints.shape[1], 3)

    before_error = mean_joint_error(smpl_joints, target_flat)
    if bool(getattr(vis_cfg, "mesh_align_to_joints", True)):
        vertices, smpl_joints = similarity_align_points(
            vertices,
            smpl_joints,
            target_flat,
            use_scale=bool(getattr(vis_cfg, "mesh_alignment_scale", True)),
        )
    after_error = mean_joint_error(smpl_joints, target_flat)
    return vertices.reshape(b, t, vertices.shape[1], 3)[0], before_error, after_error


def joints_to_fitted_smpl_vertices(joints, visual_cfg, vis_cfg, device, label=None):
    from utils.SMPL_handle import joints2smpl
    from utils.rotation2xyz import Rotation2xyz

    total_frames = min(joints.shape[0], int(getattr(vis_cfg, "max_frames", joints.shape[0])))
    joints = joints[:total_frames]
    sample_stride = max(1, int(getattr(vis_cfg, "smpl_fit_sample_stride", 1)))
    sample_ids = torch.arange(0, total_frames, sample_stride, device=joints.device)
    if sample_ids[-1].item() != total_frames - 1:
        sample_ids = torch.cat([sample_ids, torch.tensor([total_frames - 1], device=joints.device)])
    sampled_joints = joints.index_select(0, sample_ids)

    fit_name = f" {label}" if label else ""
    print(f"[smpl-fit]{fit_name}: fitting {sampled_joints.shape[0]}/{total_frames} frames "
          f"(stride={sample_stride}, iters={int(getattr(vis_cfg, 'smpl_fit_iters', 150))})", flush=True)

    scaled_joints = sampled_joints * float(getattr(vis_cfg, "smpl_fit_input_scale", 1.0))
    visual_cfg.smpl_fit_iters = int(getattr(vis_cfg, "smpl_fit_iters", getattr(visual_cfg, "smpl_fit_iters", 150)))
    j2s = joints2smpl(visual_cfg, num_frames=scaled_joints.shape[0], device=device, cuda=(device.type == "cuda"))
    rot2xyz = Rotation2xyz(visual_cfg, device=device)
    motion_tensor, opt_dict = j2s(scaled_joints)
    vertices = rot2xyz(
        torch.as_tensor(motion_tensor, dtype=torch.float32, device=device).clone(),
        mask=None,
        pose_rep='rot6d',
        translation=True,
        glob=True,
        jointstype='vertices',
        betas=(opt_dict["betas"] / 4),
        vertstrans=True,
    )
    fitted_joints = rot2xyz(
        torch.as_tensor(motion_tensor, dtype=torch.float32, device=device).clone(),
        mask=None,
        pose_rep='rot6d',
        translation=True,
        glob=True,
        jointstype='smpl',
        betas=(opt_dict["betas"] / 4),
        vertstrans=True,
    )
    vertices = vertices[0].permute(2, 0, 1).contiguous()
    fitted_joints = fitted_joints[0, :sampled_joints.shape[1]].permute(2, 0, 1).contiguous()
    vertices = vertices / float(getattr(vis_cfg, "smpl_fit_input_scale", 1.0))
    fitted_joints = fitted_joints / float(getattr(vis_cfg, "smpl_fit_input_scale", 1.0))
    if vertices.shape[0] != total_frames:
        vertices = resample_sequence(vertices, total_frames)
        fitted_joints = resample_sequence(fitted_joints, total_frames)
    if bool(getattr(vis_cfg, "smpl_fit_full_frame_align", True)):
        before_error = mean_joint_error(fitted_joints, joints)
        vertices, fitted_joints = similarity_align_points(
            vertices,
            fitted_joints,
            joints,
            use_scale=bool(getattr(vis_cfg, "mesh_alignment_scale", True)),
        )
        after_error = mean_joint_error(fitted_joints, joints)
        print(f"[smpl-fit]{fit_name}: full-frame fit-joint/HML-joint error {before_error:.4f}->{after_error:.4f}", flush=True)
    if bool(getattr(vis_cfg, "mesh_warp_to_skeleton", True)):
        before_warp_error = mean_joint_error(fitted_joints, joints)
        vertices = warp_vertices_to_joints(vertices, fitted_joints, joints, rot2xyz.smpl_model)
        fitted_joints = joints
        print(f"[smpl-fit]{fit_name}: vertex skinning warp applied (joint residual {before_warp_error:.4f})", flush=True)
    overlay_source = getattr(vis_cfg, "skeleton_overlay_source", "mesh")
    overlay_joints = joints if overlay_source == "target" else fitted_joints
    return vertices, overlay_joints


def center_motion(vertices, joints):
    root = joints[:, :1, :].mean(dim=0, keepdim=True)
    root[..., 1] = 0
    return vertices - root, joints - root


def add_skeleton(scene, trimesh, pyrender, joints, bone_color, joint_color, bone_radius, joint_radius,
                 kinematic_chain=None):
    bone_mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.75,
                                                  alphaMode='BLEND', baseColorFactor=bone_color)
    joint_mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.7,
                                                   alphaMode='BLEND', baseColorFactor=joint_color)
    edges = []
    if kinematic_chain is None:
        kinematic_chain = HML_KINEMATIC_CHAIN
    for chain in kinematic_chain:
        edges += list(zip(chain[:-1], chain[1:]))

    for a, b in edges:
        pa, pb = joints[a], joints[b]
        if np.linalg.norm(pb - pa) < 1e-5:
            continue
        cyl = trimesh.creation.cylinder(radius=bone_radius, segment=np.stack([pa, pb]), sections=10)
        scene.add(pyrender.Mesh.from_trimesh(cyl, material=bone_mat))

    for joint in joints:
        sph = trimesh.creation.uv_sphere(radius=joint_radius, count=[10, 10])
        sph.apply_translation(joint)
        scene.add(pyrender.Mesh.from_trimesh(sph, material=joint_mat))


def add_body_mesh(scene, trimesh, pyrender, vertices, faces, color, vis_cfg=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    solid_color = list(color)
    mesh_alpha = float(getattr(vis_cfg, "mesh_alpha", solid_color[3] if len(solid_color) == 4 else 1.0))
    if len(solid_color) == 4:
        solid_color[3] = mesh_alpha
    else:
        solid_color.append(mesh_alpha)
    alpha_mode = 'BLEND' if mesh_alpha < 0.999 else 'OPAQUE'
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=float(getattr(vis_cfg, "mesh_metallic", 0.0)),
                                                  roughnessFactor=float(getattr(vis_cfg, "mesh_roughness", 0.58)),
                                                  alphaMode=alpha_mode, baseColorFactor=solid_color)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    if bool(getattr(vis_cfg, "mesh_wireframe", False)):
        wire_mesh = mesh.copy()
        wire_scale = float(getattr(vis_cfg, "mesh_wireframe_scale", 1.002))
        if abs(wire_scale - 1.0) > 1e-6:
            center = wire_mesh.vertices.mean(axis=0, keepdims=True)
            wire_mesh.vertices = center + (wire_mesh.vertices - center) * wire_scale
        if bool(getattr(vis_cfg, "mesh_wireframe_match_mesh_color", False)):
            wire_factor = float(getattr(vis_cfg, "mesh_wireframe_color_factor", 0.45))
            wire_alpha = float(getattr(vis_cfg, "mesh_wireframe_alpha", 0.14))
            wire_rgb = np.clip(np.asarray(solid_color[:3], dtype=np.float32) * wire_factor, 0.0, 1.0)
            wire_color = [float(wire_rgb[0]), float(wire_rgb[1]), float(wire_rgb[2]), wire_alpha]
        else:
            wire_color = list(getattr(vis_cfg, "mesh_wireframe_color", [0.02, 0.02, 0.025, 0.22]))
        wire_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.55,
            alphaMode='BLEND',
            baseColorFactor=wire_color,
        )
        scene.add(pyrender.Mesh.from_trimesh(wire_mesh, material=wire_mat, smooth=False, wireframe=True))


def _safe_normalize(vec, fallback):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _add_feature_segment(scene, trimesh, pyrender, start, end, radius, material):
    if np.linalg.norm(end - start) < 1e-5:
        return
    segment = trimesh.creation.cylinder(radius=radius, segment=np.stack([start, end]), sections=8)
    scene.add(pyrender.Mesh.from_trimesh(segment, material=material, smooth=True))


def _add_feature_polyline(scene, trimesh, pyrender, points, radius, material, closed=False):
    if closed:
        pairs = zip(points, points[1:] + points[:1])
    else:
        pairs = zip(points[:-1], points[1:])
    for start, end in pairs:
        _add_feature_segment(scene, trimesh, pyrender, start, end, radius, material)


def add_body_features(scene, trimesh, pyrender, joints, vis_cfg, mesh_color=None):
    if joints.shape[0] <= 21:
        return
    head = joints[15]
    neck = joints[12]
    left_shoulder = joints[16]
    right_shoulder = joints[17]

    up = _safe_normalize(head - neck, [0.0, 1.0, 0.0])
    right = _safe_normalize(right_shoulder - left_shoulder, [1.0, 0.0, 0.0])
    forward = _safe_normalize(np.cross(right, up), [0.0, 0.0, 1.0])
    head_scale = max(0.08, float(np.linalg.norm(head - neck)))

    if mesh_color is not None and bool(getattr(vis_cfg, "face_feature_match_mesh_color", False)):
        face_factor = float(getattr(vis_cfg, "face_feature_color_factor", 0.42))
        face_rgb = np.clip(np.asarray(mesh_color[:3], dtype=np.float32) * face_factor, 0.0, 1.0)
        color = (float(face_rgb[0]), float(face_rgb[1]), float(face_rgb[2]), 1.0)
    else:
        color = tuple(getattr(vis_cfg, "face_feature_color", [0.02, 0.02, 0.02, 1.0]))
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.35,
                                                  alphaMode='OPAQUE', baseColorFactor=color)
    line_radius = float(getattr(vis_cfg, "face_line_radius", 0.006))
    contour_radius = float(getattr(vis_cfg, "face_contour_radius", line_radius))
    detail_radius = float(getattr(vis_cfg, "face_detail_radius", min(line_radius * 0.55, head_scale * 0.018)))
    face_center = head - up * head_scale * 0.03 + forward * head_scale * 0.36
    eye_center = face_center + up * head_scale * 0.11

    contour = []
    for theta in np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False):
        contour.append(
            face_center
            + right * np.cos(theta) * head_scale * 0.28
            + up * np.sin(theta) * head_scale * 0.42
            + forward * head_scale * 0.025
        )
    _add_feature_polyline(scene, trimesh, pyrender, contour, contour_radius, material, closed=True)

    for side in (-1.0, 1.0):
        eye_mid = eye_center + right * side * head_scale * 0.16 + forward * head_scale * 0.045
        eye_left = eye_mid - right * head_scale * 0.075
        eye_right = eye_mid + right * head_scale * 0.075
        brow_left = eye_left + up * head_scale * 0.060
        brow_right = eye_right + up * head_scale * 0.070
        _add_feature_segment(scene, trimesh, pyrender, eye_left, eye_right, detail_radius, material)
        _add_feature_segment(scene, trimesh, pyrender, brow_left, brow_right, detail_radius * 0.85, material)

    nose_bridge = face_center + up * head_scale * 0.09 + forward * head_scale * 0.04
    nose_tip = face_center - up * head_scale * 0.08 + forward * head_scale * 0.20
    _add_feature_segment(scene, trimesh, pyrender, nose_bridge, nose_tip, detail_radius * 1.25, material)

    mouth_center = face_center - up * head_scale * 0.22 + forward * head_scale * 0.04
    mouth_left = mouth_center - right * head_scale * 0.14
    mouth_right = mouth_center + right * head_scale * 0.14
    _add_feature_segment(scene, trimesh, pyrender, mouth_left, mouth_right, detail_radius, material)


def add_joint_body_mesh(scene, trimesh, pyrender, joints, color, vis_cfg):
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.72,
                                                  alphaMode='BLEND', baseColorFactor=color)
    limb_radius = float(getattr(vis_cfg, "mesh_limb_radius", 0.055))
    torso_radius = float(getattr(vis_cfg, "mesh_torso_radius", 0.085))
    joint_radius = float(getattr(vis_cfg, "mesh_joint_radius", 0.070))
    head_radius = float(getattr(vis_cfg, "mesh_head_radius", 0.120))

    for a, b in HML_EDGES:
        pa, pb = joints[a], joints[b]
        if np.linalg.norm(pb - pa) < 1e-5:
            continue
        radius = torso_radius if (a, b) in TORSO_EDGES else limb_radius
        cyl = trimesh.creation.cylinder(radius=radius, segment=np.stack([pa, pb]), sections=16)
        scene.add(pyrender.Mesh.from_trimesh(cyl, material=material, smooth=True))

    for idx, joint in enumerate(joints):
        radius = head_radius if idx == 15 else joint_radius
        sph = trimesh.creation.uv_sphere(radius=radius, count=[16, 16])
        sph.apply_translation(joint)
        scene.add(pyrender.Mesh.from_trimesh(sph, material=material, smooth=True))


def add_floor(scene, trimesh, pyrender, center, size, color, floor_y=-0.01):
    floor = trimesh.creation.box(extents=[size, 0.01, size])
    floor.apply_translation([center[0], floor_y, center[2]])
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=1.0, baseColorFactor=color)
    scene.add(pyrender.Mesh.from_trimesh(floor, material=material))


def save_video(out_path, video, fps):
    try:
        import cv2
        h, w = video[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened():
            for frame in video:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return out_path
        writer.release()
    except Exception:
        pass

    try:
        imageio.mimsave(out_path, video, fps=fps)
        return out_path
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(gif_path, video, fps=fps)
        print(f"[fallback] mp4 writer failed: {e}")
        return gif_path


def get_mesh_colors(vis_cfg, num_views):
    if num_views == 3:
        return [tuple(vis_cfg.left_color), tuple(vis_cfg.target_color), tuple(vis_cfg.right_color)]
    return [tuple(vis_cfg.left_color), tuple(vis_cfg.right_color)]


def get_skeleton_colors(vis_cfg, num_views):
    if bool(getattr(vis_cfg, "skeleton_match_mesh_color", True)):
        return get_mesh_colors(vis_cfg, num_views)
    return [tuple(getattr(vis_cfg, "skeleton_color", [0.0, 0.0, 0.0, 1.0])) for _ in range(num_views)]


def project_points_to_screen(points, camera_pose, yfov, width, height):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    camera_points = (np.linalg.inv(camera_pose) @ points_h.T).T[:, :3]
    depth = -camera_points[:, 2]
    valid = depth > 1e-4
    focal = 0.5 * height / np.tan(0.5 * yfov)
    screen = np.zeros((points.shape[0], 2), dtype=np.float32)
    safe_depth = np.maximum(depth, 1e-4)
    screen[:, 0] = width * 0.5 + focal * camera_points[:, 0] / safe_depth
    screen[:, 1] = height * 0.5 - focal * camera_points[:, 1] / safe_depth
    valid &= (screen[:, 0] >= -width * 0.1) & (screen[:, 0] <= width * 1.1)
    valid &= (screen[:, 1] >= -height * 0.1) & (screen[:, 1] <= height * 1.1)
    return screen, valid


def _rgb255(color):
    return tuple(int(np.clip(channel, 0.0, 1.0) * 255) for channel in color[:3])


def add_screen_skeleton_overlay(frame, joints_per_view, camera_pose, yfov, kinematic_chain, vis_cfg, view_colors=None):
    try:
        import cv2
    except Exception:
        return frame

    height, width = frame.shape[:2]
    overlay = frame.copy()
    edges = []
    for chain in kinematic_chain:
        edges += list(zip(chain[:-1], chain[1:]))

    alpha = float(getattr(vis_cfg, "skeleton_screen_alpha", 0.88))
    bone_width = max(1, int(getattr(vis_cfg, "skeleton_screen_bone_width", 2)))
    joint_radius = max(1, int(getattr(vis_cfg, "skeleton_screen_joint_radius", 3)))
    joint_halo = bool(getattr(vis_cfg, "skeleton_screen_joint_halo", True))

    for view_id, joints in enumerate(joints_per_view):
        if view_colors is not None and view_id < len(view_colors):
            bone_color = _rgb255(view_colors[view_id])
            joint_color = bone_color
        else:
            bone_color = _rgb255(getattr(vis_cfg, "skeleton_color", [0.0, 0.0, 0.0, 1.0]))
            joint_color = _rgb255(getattr(vis_cfg, "joint_color", [0.0, 0.0, 0.0, 1.0]))
        screen, valid = project_points_to_screen(joints, camera_pose, yfov, width, height)
        for a, b in edges:
            if a >= len(joints) or b >= len(joints) or not (valid[a] and valid[b]):
                continue
            p0 = tuple(np.round(screen[a]).astype(np.int32))
            p1 = tuple(np.round(screen[b]).astype(np.int32))
            cv2.line(overlay, p0, p1, bone_color, bone_width, cv2.LINE_AA)
        for point, is_valid in zip(screen, valid):
            if not is_valid:
                continue
            center = tuple(np.round(point).astype(np.int32))
            if joint_halo:
                cv2.circle(overlay, center, joint_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, center, joint_radius, joint_color, -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def add_screen_mesh_outline(frame, vertices_per_view, camera_pose, yfov, vis_cfg):
    try:
        import cv2
    except Exception:
        return frame

    height, width = frame.shape[:2]
    overlay = frame.copy()
    color = _rgb255(getattr(vis_cfg, "mesh_outline_color", [0.02, 0.02, 0.025, 1.0]))
    alpha = float(getattr(vis_cfg, "mesh_outline_alpha", 0.7))
    line_width = max(1, int(getattr(vis_cfg, "mesh_outline_width", 2)))

    for vertices in vertices_per_view:
        screen, valid = project_points_to_screen(vertices, camera_pose, yfov, width, height)
        points = screen[valid]
        if points.shape[0] < 3:
            continue
        hull = cv2.convexHull(np.round(points).astype(np.int32))
        cv2.polylines(overlay, [hull], True, color, line_width, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def add_screen_mesh_silhouette(frame, vertices_per_view, faces, camera_pose, yfov, vis_cfg):
    try:
        import cv2
    except Exception:
        return frame
    if faces is None or len(faces) == 0:
        return frame

    height, width = frame.shape[:2]
    overlay = frame.copy()
    color = _rgb255(getattr(vis_cfg, "mesh_silhouette_color", [0.01, 0.01, 0.015, 1.0]))
    alpha = float(getattr(vis_cfg, "mesh_silhouette_alpha", 0.9))
    line_width = max(1, int(getattr(vis_cfg, "mesh_silhouette_width", 2)))
    faces_np = np.asarray(faces, dtype=np.int64)
    view_matrix = np.linalg.inv(camera_pose)

    for vertices in vertices_per_view:
        screen, valid_vertices = project_points_to_screen(vertices, camera_pose, yfov, width, height)
        vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)], axis=1)
        camera_vertices = (view_matrix @ vertices_h.T).T[:, :3]
        tri = camera_vertices[faces_np]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        centers = tri.mean(axis=1)
        visible_face = np.einsum("ij,ij->i", normals, -centers) > 0.0
        valid_faces = valid_vertices[faces_np].all(axis=1)

        edge_faces = {}
        for face_id, face in enumerate(faces_np):
            if not valid_faces[face_id]:
                continue
            face_visible = bool(visible_face[face_id])
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (int(a), int(b)) if a < b else (int(b), int(a))
                edge_faces.setdefault(key, []).append(face_visible)

        for (a, b), adjacent in edge_faces.items():
            if not (valid_vertices[a] and valid_vertices[b]):
                continue
            is_silhouette = len(adjacent) == 1 or (any(adjacent) and not all(adjacent))
            if not is_silhouette:
                continue
            p0 = tuple(np.round(screen[a]).astype(np.int32))
            p1 = tuple(np.round(screen[b]).astype(np.int32))
            cv2.line(overlay, p0, p1, color, line_width, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def render_motion_video(vertices_list, joints_list, faces, caption, labels, out_path, vis_cfg, visual_cfg, use_proxy_mesh=True):
    trimesh, pyrender, RenderFlags = import_renderer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    width, height = int(vis_cfg.width), int(vis_cfg.height)
    renderer = pyrender.OffscreenRenderer(width, height)
    frames = min(max(len(joints) for joints in joints_list), int(vis_cfg.max_frames))
    stride = max(1, int(vis_cfg.frame_stride))
    num_views = len(joints_list)
    colors = get_mesh_colors(vis_cfg, num_views)
    skeleton_colors = get_skeleton_colors(vis_cfg, num_views)
    mesh_mode = getattr(vis_cfg, "mesh_mode", "smpl")
    overlay_source = getattr(vis_cfg, "skeleton_overlay_source", "target")
    skeleton_chain = SMPL_KINEMATIC_CHAIN if mesh_mode == "smpl_fit" and overlay_source == "mesh" else HML_KINEMATIC_CHAIN
    skeleton_mode = str(getattr(vis_cfg, "skeleton_overlay_mode", "screen")).lower()
    draw_3d_skeleton = bool(vis_cfg.use_skeleton_overlay) and skeleton_mode in ("3d", "both")
    draw_screen_skeleton = bool(vis_cfg.use_skeleton_overlay) and skeleton_mode in ("screen", "2d", "both")
    draw_mesh_outline = bool(getattr(vis_cfg, "mesh_screen_outline", True))
    draw_mesh_silhouette = bool(getattr(vis_cfg, "mesh_silhouette", False))
    vertices_list, joints_list = zip(*[center_motion(vertices, joints) for vertices, joints in zip(vertices_list, joints_list)])
    spacing = float(getattr(vis_cfg, "view_spacing", 1.65))
    offsets_x = torch.linspace(-spacing * (num_views - 1) / 2, spacing * (num_views - 1) / 2, num_views,
                               device=joints_list[0].device)
    offsets = [torch.tensor([float(x), 0.0, 0.0], device=joints_list[0].device) for x in offsets_x]
    video = []

    for frame_id in range(0, frames, stride):
        bg_color = list(getattr(vis_cfg, "bg_color", [1, 1, 1, 1]))
        ambient_light = list(getattr(vis_cfg, "ambient_light", [0.45, 0.45, 0.45]))
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

        frame_joints = []
        frame_vertices = []
        floor_candidates = []
        for idx, (vertices, joints, offset) in enumerate(zip(vertices_list, joints_list, offsets)):
            motion_id = min(frame_id, len(joints) - 1)
            cur_vertices = (vertices[motion_id] + offset).detach().cpu().numpy()
            cur_joints = (joints[motion_id] + offset).detach().cpu().numpy()
            frame_joints.append(cur_joints)
            if use_proxy_mesh and mesh_mode in ("smpl", "smpl_fit"):
                frame_vertices.append(cur_vertices)
            floor_candidates.append(float(cur_joints[:, 1].min()))
            if use_proxy_mesh and mesh_mode in ("smpl", "smpl_fit"):
                floor_candidates.append(float(cur_vertices[:, 1].min()))

            if use_proxy_mesh and mesh_mode == "joint":
                add_joint_body_mesh(scene, trimesh, pyrender, cur_joints, colors[idx], vis_cfg)
            elif use_proxy_mesh and mesh_mode in ("smpl", "smpl_fit"):
                add_body_mesh(scene, trimesh, pyrender, cur_vertices, faces, colors[idx], vis_cfg)
                if bool(getattr(vis_cfg, "show_body_features", False)):
                    add_body_features(scene, trimesh, pyrender, cur_joints, vis_cfg, mesh_color=colors[idx])
            if draw_3d_skeleton:
                bone_color = skeleton_colors[idx] if idx < len(skeleton_colors) else tuple(vis_cfg.skeleton_color)
                joint_color = bone_color if bool(getattr(vis_cfg, "skeleton_match_mesh_color", True)) else tuple(vis_cfg.joint_color)
                add_skeleton(scene, trimesh, pyrender, cur_joints, bone_color, joint_color,
                             float(vis_cfg.bone_radius), float(vis_cfg.joint_radius), kinematic_chain=skeleton_chain)

        all_pts = np.concatenate(frame_joints, axis=0)
        center = all_pts.mean(axis=0)
        floor_y = min(floor_candidates) - 0.01
        add_floor(scene, trimesh, pyrender, center, 2.5 + spacing * num_views, tuple(vis_cfg.floor_color), floor_y=floor_y)

        camera_yfov = np.pi / 3.2
        camera = pyrender.PerspectiveCamera(yfov=camera_yfov)
        camera_z = 4.2 + 0.55 * max(0, num_views - 2)
        cam_pose = np.array([
            [1, 0, 0, 0],
            [0, np.cos(-0.35), -np.sin(-0.35), 1.25],
            [0, np.sin(-0.35), np.cos(-0.35), camera_z],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        scene.add(camera, pose=cam_pose)

        key_light = pyrender.PointLight(color=[1, 1, 1], intensity=float(getattr(vis_cfg, "key_light_intensity", 18.0)))
        fill_light = pyrender.PointLight(color=[1, 1, 1], intensity=float(getattr(vis_cfg, "fill_light_intensity", 7.0)))
        rim_light = pyrender.PointLight(color=[1, 1, 1], intensity=float(getattr(vis_cfg, "rim_light_intensity", 12.0)))
        for light, light_pos in (
            (key_light, [-2.2, 2.4, 2.8]),
            (fill_light, [2.4, 1.4, 3.6]),
            (rim_light, [0.0, 2.8, -1.8]),
        ):
            light_pose = np.eye(4)
            light_pose[:3, 3] = light_pos
            scene.add(light, pose=light_pose)

        color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        color = color[..., :3].copy()
        if draw_mesh_outline and len(frame_vertices) > 0:
            color = add_screen_mesh_outline(color, frame_vertices, cam_pose, camera_yfov, vis_cfg)
        if draw_mesh_silhouette and len(frame_vertices) > 0:
            color = add_screen_mesh_silhouette(color, frame_vertices, faces, cam_pose, camera_yfov, vis_cfg)
        if draw_screen_skeleton:
            color = add_screen_skeleton_overlay(
                color, frame_joints, cam_pose, camera_yfov, skeleton_chain, vis_cfg,
                view_colors=skeleton_colors,
            )
        color = add_labels(color, caption, labels, frame_id, frames, vis_cfg)
        video.append(color)

    renderer.delete()
    return save_video(out_path, video, int(vis_cfg.fps))


def add_labels(frame, caption, labels, frame_id, total_frames, vis_cfg=None):
    try:
        import cv2
    except Exception:
        return frame

    frame = frame.copy()
    h, w = frame.shape[:2]
    label_colors = [(35, 55, 150), (35, 125, 65), (150, 65, 35)] if len(labels) == 3 else [(35, 55, 150), (150, 65, 35)]
    for idx, label in enumerate(labels):
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0]
        x = int((idx + 0.5) * w / len(labels) - text_size[0] / 2)
        cv2.putText(frame, label, (max(18, x), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, label_colors[idx], 2, cv2.LINE_AA)
    cv2.putText(frame, f"{frame_id + 1}/{total_frames}", (w - 120, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    if caption:
        text = caption[:130]
        bottom_margin = int(getattr(vis_cfg, "caption_bottom_margin", 160)) if vis_cfg is not None else 160
        bar_height = int(getattr(vis_cfg, "caption_bar_height", 44)) if vis_cfg is not None else 44
        font_scale = float(getattr(vis_cfg, "caption_font_scale", 0.68)) if vis_cfg is not None else 0.68
        thickness = max(1, int(getattr(vis_cfg, "caption_thickness", 2))) if vis_cfg is not None else 2
        bar_bottom = max(58, h - bottom_margin)
        bar_top = max(44, bar_bottom - bar_height)
        cv2.rectangle(frame, (18, bar_top), (w - 18, bar_bottom), (255, 255, 255), -1)
        cv2.putText(frame, text, (28, bar_bottom - 13), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return frame


@torch.no_grad()
def generate_prediction(sample, cfg, vis_cfg, vq_model, trans, mean, std, device, name="sample"):
    caption, src_motion, tgt_motion, m_length, has_source, src_m_length = sample[:6]
    task_type = "editing" if has_source else "generation"
    src_motion_t = torch.from_numpy(src_motion).unsqueeze(0).to(device).float()
    tgt_motion_t = torch.from_numpy(tgt_motion).unsqueeze(0).to(device).float()
    m_length_t = torch.tensor([m_length], device=device).long()
    src_m_length_t = torch.tensor([src_m_length], device=device).long()
    has_source_t = torch.tensor([has_source], device=device).long()

    if has_source:
        source_code_idx, _ = vq_model.encode(src_motion_t[..., :cfg.data.dim_pose], src_m_length_t.clone())
        pair_name = "source_gt_edit"
        labels = ("Source", "GT Target", "Edited Output")
    else:
        source_code_idx = None
        pair_name = "gt_vs_gen"
        labels = ("GT Target", "Generated Output")

    generate_kwargs = dict(
        timesteps=int(vis_cfg.time_steps),
        cond_scale=float(vis_cfg.cond_scale),
        source_cond_scale=float(getattr(vis_cfg, "source_cond_scale", 1.0)),
        source_m_lens=src_m_length_t // cfg.data.unit_length if has_source else None,
        temperature=float(vis_cfg.temperature),
        topk_filter_thres=float(vis_cfg.topkr),
        gsample=bool(vis_cfg.gsample),
    )
    try:
        mids = trans.generate(
            source_code_idx, [caption], m_length_t // cfg.data.unit_length, has_source_t, t_drop=0,
            source_hint_ratio=float(getattr(vis_cfg, "source_hint_ratio", getattr(cfg.inference, "source_hint_ratio", 0.0))),
            **generate_kwargs,
        )
    except TypeError as e:
        if "source_hint_ratio" not in str(e):
            raise
        mids = trans.generate(
            source_code_idx, [caption], m_length_t // cfg.data.unit_length, has_source_t, t_drop=0,
            **generate_kwargs,
        )
    pred_motion = vq_model.forward_decoder(mids, m_length_t.clone())[0].detach().cpu().numpy()
    if has_source:
        motions = [src_motion, tgt_motion, pred_motion]
        lengths = [src_m_length, m_length, m_length]
    else:
        motions = [tgt_motion, pred_motion]
        lengths = [m_length, m_length]
    return caption, motions, lengths, task_type, pair_name, labels, name


def visualize_sample(result, cfg, vis_cfg, visual_cfg, mean, std, smpl_model, faces, device):
    caption, motions, lengths, task_type, pair_name, labels, name = result
    motion_denorms = [denorm_motion(motion, mean, std, length, device) for motion, length in zip(motions, lengths)]
    joints_list = [motion_to_joints(motion_denorm, cfg.data.joint_num) for motion_denorm in motion_denorms]
    mesh_mode = getattr(vis_cfg, "mesh_mode", "smpl")
    align_visual_lengths = bool(getattr(vis_cfg, "align_visual_lengths", True))
    reference_frames = joints_list[1].shape[0] if len(joints_list) == 3 else joints_list[0].shape[0]

    if align_visual_lengths and mesh_mode in ("smpl_fit", "joint", "none"):
        joints_list = [resample_sequence(joints, reference_frames) for joints in joints_list]
    skeleton_joints_list = [joints.clone() for joints in joints_list]

    if mesh_mode == "joint":
        vertices_list = joints_list
        use_proxy_mesh = bool(vis_cfg.use_proxy_mesh)
    elif mesh_mode == "none":
        vertices_list = joints_list
        faces = np.zeros((0, 3), dtype=np.int64)
        use_proxy_mesh = False
    elif mesh_mode == "smpl_fit":
        if smpl_model is None or faces is None or len(faces) == 0:
            raise ValueError("mesh_mode='smpl_fit' requires a loaded SMPL model and non-empty faces.")
        fit_results = [
            joints_to_fitted_smpl_vertices(joints, visual_cfg, vis_cfg, device, label=label)
            for joints, label in zip(joints_list, labels)
        ]
        vertices_list = [vertices for vertices, _ in fit_results]
        joints_list = [overlay_joints for _, overlay_joints in fit_results]
        fitted_frames = vertices_list[0].shape[0]
        joints_list = [resample_sequence(joints, fitted_frames) for joints in joints_list]
        use_proxy_mesh = True
    elif mesh_mode == "smpl":
        if smpl_model is None or faces is None or len(faces) == 0:
            raise ValueError("mesh_mode='smpl' requires a loaded SMPL model and non-empty faces.")
        try:
            vertices_list = []
            align_errors = []
            for motion_denorm, joints in zip(motion_denorms, joints_list):
                vertices, before_error, after_error = motion_to_proxy_vertices(motion_denorm, joints, smpl_model, vis_cfg, device)
                vertices_list.append(vertices)
                align_errors.append((before_error, after_error))
        except Exception as e:
            if not vis_cfg.allow_skeleton_fallback:
                raise
            print(f"[fallback] proxy mesh failed for {name}: {e}")
            vertices_list = joints_list
            faces = np.zeros((0, 3), dtype=np.int64)
            use_proxy_mesh = False
        else:
            err_text = ", ".join(
                f"{label}: {before:.4f}->{after:.4f}"
                for label, (before, after) in zip(labels, align_errors)
            )
            print(f"[mesh-align] {name}: mean SMPL-joint/HML-joint error {err_text}")
            use_proxy_mesh = True
    else:
        raise ValueError(f"Unknown mesh_mode='{mesh_mode}'. Expected 'smpl_fit', 'smpl', 'joint', or 'none'.")

    if align_visual_lengths and mesh_mode == "smpl":
        vertices_list = [resample_sequence(vertices, reference_frames) for vertices in vertices_list]
        joints_list = [resample_sequence(joints, reference_frames) for joints in joints_list]

    safe_name = str(name).replace("/", "_").replace("\\", "_")
    separate_skeleton = bool(getattr(vis_cfg, "save_skeleton_separate", False)) and not bool(getattr(vis_cfg, "skeleton_only", False))
    suffix = "_skeleton" if bool(getattr(vis_cfg, "skeleton_only", False)) else "_mesh" if separate_skeleton else ""
    out_name = f"{safe_name}_{task_type}_{pair_name}{suffix}.mp4"
    out_path = pjoin(vis_cfg.output_dir, out_name)
    saved_path = render_motion_video(vertices_list, joints_list, faces, caption, labels, out_path, vis_cfg, visual_cfg,
                                     use_proxy_mesh=use_proxy_mesh)
    print(f"Saved: {os.path.abspath(saved_path)}", flush=True)
    if separate_skeleton:
        skel_cfg = copy.copy(vis_cfg)
        skel_cfg.mesh_mode = "none"
        skel_cfg.use_proxy_mesh = False
        skel_cfg.use_skeleton_overlay = True
        skel_cfg.skeleton_overlay_mode = "screen"
        skel_cfg.skeleton_overlay_source = "target"
        skel_cfg.mesh_screen_outline = False
        skel_cfg.skeleton_match_mesh_color = True
        skel_cfg.skeleton_screen_alpha = max(float(getattr(skel_cfg, "skeleton_screen_alpha", 0.95)), 0.95)
        skel_cfg.skeleton_screen_bone_width = max(int(getattr(skel_cfg, "skeleton_screen_bone_width", 2)), 2)
        skel_cfg.skeleton_screen_joint_radius = max(int(getattr(skel_cfg, "skeleton_screen_joint_radius", 3)), 3)
        skeleton_out_name = f"{safe_name}_{task_type}_{pair_name}_skeleton.mp4"
        skeleton_out_path = pjoin(vis_cfg.output_dir, skeleton_out_name)
        skeleton_faces = np.zeros((0, 3), dtype=np.int64)
        saved_skeleton_path = render_motion_video(
            skeleton_joints_list, skeleton_joints_list, skeleton_faces, caption, labels,
            skeleton_out_path, skel_cfg, visual_cfg, use_proxy_mesh=False,
        )
        print(f"Saved skeleton: {os.path.abspath(saved_skeleton_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/visualize_motion_editing_hml.yaml')
    parser.add_argument('--mode', type=str, default=None, choices=['all', 'gen', 'edit'])
    parser.add_argument('--skeleton-only', '--skel', action='store_true',
                        help='Fast path: skip SMPL fitting and render color-matched skeleton only.')
    parser.add_argument('--random-select', action='store_true',
                        help='Select samples randomly instead of using start_index.')
    parser.add_argument('--random-seed', action='store_true',
                        help='Use a fresh random seed for sample selection and generation.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the visualization seed.')
    args = parser.parse_args()

    vis_cfg = load_config(args.config)
    # quick task switch, data split uses motionfix_start_id
    if args.mode is not None:
        vis_cfg.mode = args.mode
    if args.random_select:
        vis_cfg.random_select = True
    if args.seed is not None:
        vis_cfg.seed = int(args.seed)
    if args.random_seed:
        vis_cfg.random_select = True
        vis_cfg.seed = int.from_bytes(os.urandom(4), byteorder="little", signed=False)
    if args.skeleton_only:
        vis_cfg.skeleton_only = True
        vis_cfg.mesh_mode = "none"
        vis_cfg.use_proxy_mesh = False
        vis_cfg.use_skeleton_overlay = True
        vis_cfg.skeleton_overlay_mode = "screen"
        vis_cfg.skeleton_overlay_source = "target"
        vis_cfg.mesh_screen_outline = False
        vis_cfg.skeleton_match_mesh_color = True
        vis_cfg.skeleton_screen_alpha = max(float(getattr(vis_cfg, "skeleton_screen_alpha", 0.95)), 0.95)
        vis_cfg.skeleton_screen_bone_width = max(int(getattr(vis_cfg, "skeleton_screen_bone_width", 2)), 2)
        vis_cfg.skeleton_screen_joint_radius = max(int(getattr(vis_cfg, "skeleton_screen_joint_radius", 3)), 3)
    print(
        f"[visualize] output_dir={os.path.abspath(vis_cfg.output_dir)} "
        f"mode={vis_cfg.mode} num_samples={vis_cfg.num_samples} "
        f"random_select={getattr(vis_cfg, 'random_select', False)} "
        f"seed={getattr(vis_cfg, 'seed', '-')} "
        f"mesh_mode={getattr(vis_cfg, 'mesh_mode', 'smpl')} "
        f"skeleton_only={getattr(vis_cfg, 'skeleton_only', False)} "
        f"skeleton_overlay_source={getattr(vis_cfg, 'skeleton_overlay_source', 'mesh')} "
        f"max_frames={vis_cfg.max_frames} frame_stride={vis_cfg.frame_stride} "
        f"smpl_fit_stride={getattr(vis_cfg, 'smpl_fit_sample_stride', '-')} "
        f"smpl_fit_iters={getattr(vis_cfg, 'smpl_fit_iters', '-')}",
        flush=True,
    )

    cfg = load_config(vis_cfg.trans_cfg)
    visual_cfg = load_config(vis_cfg.visual_cfg)
    device = torch.device(cfg.exp.device)
    fixseed(int(vis_cfg.seed))

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    mean, std = load_mean_std(cfg)
    vq_cfg = load_config(pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name))
    vq_model = load_vq_model(cfg, vq_cfg, device)
    trans = load_trans_model(cfg, vq_cfg, vis_cfg.ckpt, device)
    if hasattr(trans, "set_vq_quantizer") and hasattr(vq_model, "quantizer"):
        trans.set_vq_quantizer(vq_model.quantizer)
    elif hasattr(trans, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
        trans.set_vq_codebook(vq_model.quantizer.codebook)
    dataset = build_dataset(cfg, mean, std, vis_cfg.split, int(vis_cfg.motionfix_start_id))

    mesh_mode = getattr(vis_cfg, "mesh_mode", "smpl")
    if mesh_mode in ("smpl", "smpl_fit"):
        if SMPL is None:
            raise ImportError(f"mesh_mode='{mesh_mode}' requires the 'smplx' package. Install smplx to render body meshes.")
        smpl_model = SMPL(visual_cfg, model_path=pjoin(visual_cfg.SMPL_MODEL_DIR, "smpl")).eval().to(device)
        faces = smpl_model.faces
    else:
        smpl_model = None
        faces = np.zeros((0, 3), dtype=np.int64)

    indices = select_indices(
        dataset,
        vis_cfg.mode,
        int(vis_cfg.start_index),
        int(vis_cfg.num_samples),
        random_select=bool(getattr(vis_cfg, "random_select", False)),
        seed=int(vis_cfg.seed),
    )
    if len(indices) == 0:
        print("No samples selected. Check split/mode/motionfix_start_id.")
        return

    os.makedirs(vis_cfg.output_dir, exist_ok=True)
    print(f"[visualize] selected_indices={indices}", flush=True)
    for idx in indices:
        sample = dataset[idx]
        result = generate_prediction(sample, cfg, vis_cfg, vq_model, trans, mean, std, device, name=dataset.name_list[idx])
        visualize_sample(result, cfg, vis_cfg, visual_cfg, mean, std, smpl_model, faces, device)


if __name__ == "__main__":
    main()
