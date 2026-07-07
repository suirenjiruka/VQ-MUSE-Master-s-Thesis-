import torch
import os, argparse
import numpy as np
from os.path import join as pjoin

from configs.load_config import load_config
from SnapMogen_model.vq.rvq_model import HRVQVAE
from SnapMogen_model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from SnapMogen_model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model import VAMotion
from utils.get_opt import get_opt
from evaluator.evaluator import evaluation_vamotion_hml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_cfg', type=str,
                        default='./configs/train_vamotion_hml.yaml',
                        help='VAMotion training config (HumanML3D)')
    parser.add_argument('--eval_cfg', type=str,
                        default='./configs/evaluation_hml.yaml',
                        help='Eval config (time_steps, cond_scales, repeat_time, etc.)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override checkpoint filename (default: eval_cfg.which_ckpt)')
    args = parser.parse_args()

    trans_cfg = load_config(args.trans_cfg)
    eval_cfg  = load_config(args.eval_cfg)

    device = torch.device(eval_cfg.device)
    if eval_cfg.device != 'cpu':
        torch.cuda.set_device(eval_cfg.device)

    # --- VQ model ---
    vq_cfg_path = pjoin(trans_cfg.vq_cfg_dir, 'configs', trans_cfg.vq_name)
    vq_cfg = load_config(vq_cfg_path)
    vq_model = HRVQVAE(vq_cfg,
                       vq_cfg.data.dim_pose,
                       vq_cfg.model.down_t,
                       vq_cfg.model.stride_t,
                       vq_cfg.model.width,
                       vq_cfg.model.depth,
                       vq_cfg.model.dilation_growth_rate,
                       vq_cfg.model.vq_act,
                       vq_cfg.model.use_attn,
                       vq_cfg.model.vq_norm)
    vq_ckpt_path = pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq',
                         vq_cfg.exp.name, 'model', trans_cfg.vq_ckpt)
    vq_ckpt = torch.load(vq_ckpt_path, map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in vq_ckpt else 'model'
    vq_model.load_state_dict(vq_ckpt[model_key])
    print(f'VQ model loaded: {vq_cfg.exp.name}')
    vq_model.to(device).eval()

    # --- VAMotion transformer ---
    trans_cfg.vq = vq_cfg.quantizer
    trans = VAMotion(
        cfg=trans_cfg,
        device=device,
        full_length=trans_cfg.data.max_motion_length // trans_cfg.data.unit_length,
        max_video_len=(trans_cfg.data.max_motion_length // 7) + 2,
        video_joint_num=None,
    )
    model_dir = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, 'VA_motion', 'model')
    ckpt_name = args.ckpt or eval_cfg.which_ckpt
    trans_ckpt = torch.load(pjoin(model_dir, ckpt_name), map_location=device, weights_only=True)
    missing, unexpected = trans.load_state_dict(trans_ckpt['training_model'], strict=False)
    if missing:
        print(f"missing keys from ckpt: {missing[:5]}")
    if unexpected:
        print(f"unexpected keys from ckpt: {unexpected[:5]}")
    if hasattr(trans, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
        trans.set_vq_codebook(vq_model.quantizer.codebook)
    print(f'VAMotion loaded: {ckpt_name} (epoch {trans_ckpt.get("epoch", "?")})')
    trans.to(device).eval()

    # --- HumanML3D evaluator + test dataset ---
    dataset_opt_path = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name,
                             'Comp_v6_KLD005', 'opt.txt')
    hml3d_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(hml3d_opt)
    eval_loader, _ = get_dataset_motion_loader(
        dataset_opt_path, 32, 'test',
        trans_cfg.data.feat2D_dir, trans_cfg.vipe_checkpt, device=device,
    )

    # --- Output log ---
    log_dir = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, 'VA_motion', 'eval')
    os.makedirs(log_dir, exist_ok=True)
    log_name = f'{eval_cfg.ext}_{ckpt_name.replace(".tar", "")}.log'
    out_path = pjoin(log_dir, log_name)
    f = open(out_path, 'w')
    print(f'Logging to {out_path}')

    t_drop  = getattr(eval_cfg, 't_drop',  0)
    v_drop  = getattr(eval_cfg, 'v_drop',  0)
    cal_mm  = getattr(eval_cfg, 'cal_mm',  False)
    topkr   = getattr(eval_cfg, 'topkr',   0.9)
    gsample = getattr(eval_cfg, 'gsample', True)
    temp    = getattr(eval_cfg, 'temperature', 1)

    vs = eval_cfg.video_emphasis

    for cs in eval_cfg.cond_scales:
        for ts in eval_cfg.time_steps:
            fid_list, div_list = [], []
            top1_list, top2_list, top3_list, match_list, mm_list = [], [], [], [], []

            for i in range(eval_cfg.repeat_time):
                header = f'Guidance scale: {cs}, time step: {ts}, repeat: {i}'
                print(header)
                print(header, file=f, flush=True)

                with torch.no_grad():
                    fid, div, Rp, match, mm = evaluation_vamotion_hml(
                        eval_loader, vq_model, trans, i, eval_wrapper,
                        time_steps=ts, cond_scale=cs, video_emphasis=vs,
                        t_drop=t_drop, v_drop=v_drop,
                        unit_length=trans_cfg.data.unit_length,
                        temperature=temp, topkr=topkr, gsample=gsample,
                        cal_mm=cal_mm,
                    )
                fid_list.append(fid)
                div_list.append(div)
                top1_list.append(Rp[0])
                top2_list.append(Rp[1])
                top3_list.append(Rp[2])
                match_list.append(match)
                mm_list.append(mm)

            n = eval_cfg.repeat_time
            ci = lambda arr: np.std(arr) * 1.96 / np.sqrt(n)
            fid_a   = np.array(fid_list)
            div_a   = np.array(div_list)
            top1_a  = np.array(top1_list)
            top2_a  = np.array(top2_list)
            top3_a  = np.array(top3_list)
            match_a = np.array(match_list)
            mm_a    = np.array(mm_list)

            result_header = f'=== Final Result  cs={cs}  ts={ts}  ckpt={ckpt_name} ==='
            msg = (
                f"\tFID:          {np.mean(fid_a):.3f} ± {ci(fid_a):.3f}\n"
                f"\tDiversity:    {np.mean(div_a):.3f} ± {ci(div_a):.3f}\n"
                f"\tTOP1: {np.mean(top1_a):.3f} ± {ci(top1_a):.3f}  "
                f"TOP2: {np.mean(top2_a):.3f} ± {ci(top2_a):.3f}  "
                f"TOP3: {np.mean(top3_a):.3f} ± {ci(top3_a):.3f}\n"
                f"\tMatching:     {np.mean(match_a):.3f} ± {ci(match_a):.3f}\n"
                f"\tMultimodality:{np.mean(mm_a):.3f} ± {ci(mm_a):.3f}\n"
            )
            print(result_header)
            print(msg)
            print(result_header, file=f, flush=True)
            print(msg, file=f, flush=True)

    f.close()
    print('Done.')
