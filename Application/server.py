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
from flask import Flask, request, jsonify

# make the project (master_project) importable regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # relative paths in configs resolve from the project root

from configs.load_config import load_config
from utils.visualize_motion_editing_hml import (
    load_mean_std, load_vq_model, load_trans_model, denorm_motion, motion_to_joints,
)

TRANS_CFG = os.environ.get("TRANS_CFG", pjoin(ROOT, "configs", "train_vamotion_hml.yaml"))
CKPT      = os.environ.get("CKPT", "best.tar")
PORT      = int(os.environ.get("PORT", "5000"))

print(f"[boot] loading model … cfg={TRANS_CFG} ckpt={CKPT}", flush=True)
cfg = load_config(TRANS_CFG)
device = torch.device(cfg.exp.device if (cfg.exp.device == "cpu" or torch.cuda.is_available()) else "cpu")
mean, std = load_mean_std(cfg)
vq_cfg = load_config(pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name))
vq_model = load_vq_model(cfg, vq_cfg, device)
trans = load_trans_model(cfg, vq_cfg, CKPT, device)          # also sets cfg.vq
if hasattr(trans, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
    trans.set_vq_codebook(vq_model.quantizer.codebook)
UNIT = cfg.data.unit_length
DIM  = cfg.data.dim_pose
MAXL = cfg.data.max_motion_length
FPS  = int(getattr(cfg.data, "fps", 20))
print(f"[boot] ready on {device} · unit={UNIT} · dim={DIM} · fps={FPS}", flush=True)

# session cache: motion id -> normalized 263-dim motion (np [T, 263]) kept so edits can re-encode the source
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
            src = torch.from_numpy(CACHE[source_id]).unsqueeze(0).to(device).float()   # [1, Ts, 263]
            Ts = src.shape[1]
            src_len = torch.tensor([Ts], device=device).long()
            source_code_idx, _ = vq_model.encode(src[..., :DIM], src_len.clone())
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
        try:
            mids = trans.generate(source_code_idx, [text], m_len // UNIT, has_src_t, t_drop=0,
                                  source_hint_ratio=float(p.get("source_hint_ratio", 0.0)), **kwargs)
        except TypeError as e:
            if "source_hint_ratio" not in str(e):
                raise
            mids = trans.generate(source_code_idx, [text], m_len // UNIT, has_src_t, t_drop=0, **kwargs)

        pred = vq_model.forward_decoder(mids, m_len.clone())[0].detach().cpu().numpy()   # [L, 263] normalized
        mid = uuid.uuid4().hex[:8]
        CACHE[mid] = pred
        denorm = denorm_motion(pred, mean, std, pred.shape[0], device)
        joints = motion_to_joints(denorm, cfg.data.joint_num).detach().cpu().numpy()      # [L, 22, 3]
        return mid, joints.astype(np.float32).tolist()


app = Flask(__name__)

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

@app.route("/health")
def health():
    return jsonify(ok=True, device=str(device), ckpt=CKPT, fps=FPS, cached=len(CACHE))

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return ("", 204)
    d = request.get_json(force=True)
    mid, joints = _run(d["text"], int(d.get("length", 120)), None, d["params"])
    return jsonify(id=mid, joints=joints, fps=FPS)

@app.route("/edit", methods=["POST", "OPTIONS"])
def edit():
    if request.method == "OPTIONS":
        return ("", 204)
    d = request.get_json(force=True)
    mid, joints = _run(d["text"], 0, d["source_id"], d["params"])
    return jsonify(id=mid, joints=joints, fps=FPS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=False)
