"""在『能正常跑 SMPL 的環境』量 position-driven LBS 的表面品質(對照真 SMPL ground-truth)。

用法(在 server.py 說明的那個 conda / WSL 環境):
    cd master_project
    python Application/bench_lbs.py

它會：
  1. 先 sanity check：把整個身體轉 90°，確認 SMPL forward 真的會變形(不會就直接報錯，量測無意義)。
  2. 對幾個代表姿勢跑真 SMPL 得到 GT 頂點 + 22 關節，再用 22 關節餵 lbs_vertices，
     比較 plain / +posebs / +stretch / ALL 的『邊長扭曲%』與『頂點誤差mm』(越低越像真人)。

不啟動 flask、不載你的動作模型；只從 server.py 抽出 lbs_vertices 測真實程式碼。
"""
import os, sys, ast
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import smplx
from configs.load_config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUAL_CFG = load_config(os.path.join(ROOT, "utils", "visual_config.yaml"))
SMPL_DIR = VISUAL_CFG.SMPL_MODEL_DIR


def build_model(F):
    return smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", ext="pkl",
                        batch_size=F).to(device)


# --- 從 server.py 抽出 lbs_vertices 及其相依函式(測真實程式碼，不啟 flask) ---
def load_lbs_funcs():
    src = open(os.path.join(HERE, "server.py"), encoding="utf-8").read()
    ns = {"torch": torch, "device": device}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_shortest_arc", "lbs_vertices"}:
            exec(compile(ast.Module([node], []), "server.py", "exec"), ns)
    return ns


def build_lbs_assets(m):
    v_template = m.v_template.to(device).float()
    lbs_w = m.lbs_weights.to(device).float()
    parents = m.parents.to(device).long()
    J_rest = (m.J_regressor.to(device).float()) @ v_template
    NJ = parents.shape[0]
    dir_children = [[] for _ in range(NJ)]
    for j in range(1, NJ):
        if j < 22:
            dir_children[int(parents[j])].append(j)
    rest_dir = torch.zeros(NJ, 3, device=device)
    is_leaf = torch.zeros(NJ, dtype=torch.bool, device=device)
    for k in range(NJ):
        ch = dir_children[k]
        if ch:
            d = (J_rest[ch] - J_rest[k]).mean(0)
            rest_dir[k] = d / d.norm().clamp(min=1e-8)
        else:
            is_leaf[k] = True
    return dict(v_template=v_template, lbs_w=lbs_w, parents=parents, J_rest=J_rest, NJ=NJ,
                dir_children=dir_children, rest_dir=rest_dir, is_leaf=is_leaf,
                posedirs=m.posedirs.to(device).float())


@torch.no_grad()
def main():
    m1 = build_model(1)
    v_rest = m1.v_template.to(device).float()

    # 1) sanity：整個身體轉 90°，頂點必須大幅移動，否則 SMPL forward 壞了
    go = torch.zeros(1, 3, device=device); go[0, 1] = 1.5708
    moved = (m1(global_orient=go, body_pose=torch.zeros(1, 69, device=device)).vertices[0]
             - v_rest).norm(-1).max().item() * 100
    print(f"[sanity] global_orient 90° 頂點最大位移 = {moved:.1f} cm  "
          f"({'OK，SMPL 會變形' if moved > 20 else '!! 幾乎沒動 → 此環境 SMPL forward 壞了，量測無意義'})")
    if moved < 20:
        print("       請換到能正常跑 SMPL 的環境再跑此腳本。")
        return

    ns = load_lbs_funcs()
    faces = m1.faces.astype(np.int64)
    E = torch.from_numpy(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)).long().to(device)

    def edge_distort(vp, vg):
        lp = (vp[:, E[:, 0]] - vp[:, E[:, 1]]).norm(-1)
        lg = (vg[:, E[:, 0]] - vg[:, E[:, 1]]).norm(-1)
        return ((lp - lg).abs() / lg.clamp(min=1e-6)).mean().item() * 100

    def bpose(spec):
        bp = torch.zeros(1, 69, device=device)
        for j, ax, a in spec:
            v = torch.tensor(ax, dtype=torch.float, device=device); v = v / v.norm()
            bp[0, (j - 1) * 3:(j - 1) * 3 + 3] = v * a
        return bp

    poses = {
        "elbows":   bpose([(19, (0, 0, 1), 1.6), (20, (0, 0, 1), 1.2)]),   # R_elbow 彎
        "knees":    bpose([(4, (1, 0, 0), 1.3), (5, (1, 0, 0), 1.3)]),     # 雙膝彎
        "shoulder": bpose([(16, (0, 0, 1), -1.2), (17, (0, 0, 1), 1.2)]),  # 抬肩
        "hips":     bpose([(1, (1, 0, 0), 1.0), (2, (1, 0, 0), 1.0)]),     # 抬腿
    }
    cfgs = {"plain": (0, 0), "+posebs": (1, 0), "+stretch": (0, 1), "ALL": (1, 1)}

    print(f"\n{'pose':10s} | " + " | ".join(f"{c:>8s}" for c in cfgs) + "   (edge distort% ↓)")
    print("-" * 70)
    agg = {c: [] for c in cfgs}
    for pn, bp in poses.items():
        mm = build_model(1)
        out = mm(global_orient=torch.zeros(1, 3, device=device), body_pose=bp,
                 betas=torch.zeros(1, 10, device=device))
        vg, jt = out.vertices, out.joints[:, :22]
        row = []
        for c, (pbs, st) in cfgs.items():
            ns["LBS_POSE_BS"] = bool(pbs); ns["LBS_STRETCH"] = bool(st)
            ns["LBS"] = build_lbs_assets(mm)
            e = edge_distort(ns["lbs_vertices"](jt), vg)
            row.append(e); agg[c].append(e)
        print(f"{pn:10s} | " + " | ".join(f"{d:8.2f}" for d in row))
    print("-" * 70)
    print(f"{'MEAN':10s} | " + " | ".join(f"{np.mean(agg[c]):8.2f}" for c in cfgs))
    print("\n越低越像真人。plain=原始 LBS；比 plain 低才是真的有幫助。")


if __name__ == "__main__":
    main()
