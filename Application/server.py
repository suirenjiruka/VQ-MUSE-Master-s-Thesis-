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
import os, sys, uuid, threading
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
CKPT      = os.environ.get("CKPT", "best_45.tar")
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
# MESH_MODE=own(預設): 自寫 SMPL 擬合(smplx 直接跑，不碰 visualize)。以動作旋轉 warm-start，
#                      Adam 優化 SMPL pose+shape+『每幀 transl』貼合 22 關節 → 精準且會真正位移。
# MESH_MODE=fit      : visualize 的 joints2smpl SMPLify(保留對照)。
# MESH_MODE=fast     : rot6d 直接前向(免優化，會扭曲，不建議)。
# MESH_MODE=metaball : 回傳關節點，前端 metaball 皮膚(不碰 SMPL)。
MESH_MODE   = os.environ.get("MESH_MODE", "own").lower()
OWN_ITERS   = int(os.environ.get("OWN_ITERS", "400"))   # 自寫擬合迭代數(精度優先，可調)
SMPL_ITERS  = int(os.environ.get("SMPL_ITERS", "8"))    # fit 模式 warm-start 後迭代
SMPL_STRIDE = int(os.environ.get("SMPL_STRIDE", "1"))
WARM        = os.environ.get("WARM", "1") != "0"        # 用動作旋轉當初始姿勢(warm-start)
NEED_SMPL = MESH_MODE in ("own", "fit", "fast")
MESH_OK = False
FACES = np.zeros((0, 3), np.int64)
VIS_CFG = VISUAL_CFG = SMPL_MODEL = None
SMPL_DIR = None
fit_vertices = proxy_vertices = _init_pose_fns = None
if NEED_SMPL:
    try:
        import smplx as _smplx
        # warm-start 用的旋轉轉換(與 motion_to_proxy_vertices 同慣例)
        from utils.motion_process_bvh import recover_root_rot_pos as _rrp
        from utils.common.quaternion import qinv as _qinv
        from utils.rotation_conversions import (
            quaternion_to_matrix as _q2m, rotation_6d_to_matrix as _r6m, matrix_to_axis_angle as _m2aa,
        )
        _init_pose_fns = (_rrp, _qinv, _q2m, _r6m, _m2aa)
        VISUAL_CFG = load_config(pjoin(ROOT, "utils", "visual_config.yaml"))
        SMPL_DIR = VISUAL_CFG.SMPL_MODEL_DIR
        _faces_model = _smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=1)
        FACES = np.asarray(_faces_model.faces, dtype=np.int64)
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
        detail = f"iters={OWN_ITERS}" if MESH_MODE == "own" else f"iters={SMPL_ITERS} stride={SMPL_STRIDE}"
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


def _ground(a):
    """a:(F,N,3) → 落地(最低點 y=0) + 起始幀質心 XZ 置中，保留位移。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    a[..., 1] -= a[..., 1].min()
    c0 = a[0].mean(axis=0)
    a[..., 0] -= c0[0]; a[..., 2] -= c0[2]
    return a

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

        pred = vq_model.forward_decoder(mids, m_len.clone())[0].detach().cpu().numpy()   # [L, 263] normalized
        mid = uuid.uuid4().hex[:8]
        # 保存這次產出的『原始 token』本身（含被編輯後的結果），供之後的 edit 直接當 source
        CACHE[mid] = {"mids": [s.detach().cpu() for s in mids], "len": int(m_len.item())}
        denorm = denorm_motion(pred, mean, std, pred.shape[0], device)                    # [1, L, 263]
        joints_t = motion_to_joints(denorm, cfg.data.joint_num)                           # [L, 22, 3]
        if MESH_OK:
            if MESH_MODE == "own":
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
