import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join as pjoin

from configs.load_config import load_config
from SnapMogen_model.vq.rvq_model import HRVQVAE
from SnapMogen_model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model import VAMotion
from utils.get_opt import get_opt
from utils.fixseeds import fixseed
from utils.word_vectorizer import WordVectorizer
from utils.metrics import calculate_R_precision, calculate_activation_statistics, calculate_diversity
from utils.metrics import calculate_frechet_distance, euclidean_distance_matrix
from dataset.HumanML3D_dataset import Text2MotionDatasetEval, HML3DMotionEditDataset, collate_fn
from evaluator.evaluator import evaluation_motion_editing_hml
from evaluator.tmr_eval_wrapper import TMREvaluatorWrapper


def load_mean_std(cfg):
    # HML3D / mixed MotionFix data may keep normalization files in different folders
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
    raise FileNotFoundError("Cannot find mean/std normalization files for motion data.")


def load_vq_model(trans_cfg, device):
    # Load the frozen HRVQ-VAE used to tokenize source motion and decode predicted tokens
    vq_cfg_path = pjoin(trans_cfg.vq_cfg_dir, 'configs', trans_cfg.vq_name)
    vq_cfg = load_config(vq_cfg_path)
    vq_model = HRVQVAE(vq_cfg, vq_cfg.data.dim_pose, vq_cfg.model.down_t, vq_cfg.model.stride_t,
                       vq_cfg.model.width, vq_cfg.model.depth, vq_cfg.model.dilation_growth_rate,
                       vq_cfg.model.vq_act, vq_cfg.model.use_attn, vq_cfg.model.vq_norm)

    vq_ckpt_path = pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model', trans_cfg.vq_ckpt)
    vq_ckpt = torch.load(vq_ckpt_path, map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in vq_ckpt else 'model'
    vq_model.load_state_dict(vq_ckpt[model_key])
    print(f'VQ model loaded: {vq_cfg.exp.name}')
    return vq_model.to(device).eval(), vq_cfg


def load_trans_model(trans_cfg, vq_cfg, ckpt_name, device, stage_dir='VA_motion'):
    # Load current motion editing transformer; no video/joint condition is used
    trans_cfg.vq = vq_cfg.quantizer
    trans_cfg.vq.nb_code = vq_cfg.quantizer.nb_code

    model_dir = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, stage_dir, 'model')
    ckpt_path = pjoin(model_dir, ckpt_name)
    trans_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = trans_ckpt['training_model']

    # old ckpt compatibility
    if 'task_embed.weight' not in state:
        trans_cfg.model.use_task_token = False
    trans = VAMotion(cfg=trans_cfg, device=device, full_length=trans_cfg.data.max_motion_length // trans_cfg.data.unit_length)

    state = {k: v for k, v in state.items() if not k.startswith('text_emb.')}
    missing, unexpected = trans.load_state_dict(state, strict=False)
    old_keys = ('text_delta_encoder.', 'condition_encoder.', 'edit_map_head.',
                'part_gate_mlp.', 'part_cond_mlp.', 'part_text_mlp.', 'part_source_mlp.',
                'edit_loc_head.', 'task_embed.', 'text_delta_scale', 'source_delta_scale')
    unexpected = [k for k in unexpected if not k.startswith(old_keys)]
    if missing:
        print(f'missing keys from ckpt: {missing[:5]}')
    assert len(unexpected) == 0, f"unexpected keys from ckpt: {unexpected[:5]}"
    print(f'VAMotion loaded: {ckpt_name} (epoch {trans_ckpt.get("epoch", "?")})')
    return trans.to(device).eval()


def prepare_hml_opt(opt, trans_cfg):
    # get_opt has legacy Linux defaults, so overwrite all paths from training config
    opt.root_dir = trans_cfg.data.root_dir
    opt.data_root = pjoin(trans_cfg.data.root_dir, "HumanML3D")
    opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
    opt.joint_dir = pjoin(opt.data_root, "new_joints")
    opt.text_dir = pjoin(opt.data_root, "texts")
    opt.max_motion_length = trans_cfg.data.max_motion_length
    opt.unit_length = trans_cfg.data.unit_length
    opt.max_text_len = trans_cfg.data.max_text_length
    return opt


def is_editing_id(name, motionfix_start_id):
    raw = name[1:] if name.startswith("M") else name
    return raw.isdigit() and int(raw) >= motionfix_start_id


def build_task_split(trans_cfg, hml3d_opt, split, task_type, motionfix_start_id):
    # Filter mixed split once so each evaluator only reads valid files for its task
    src_split = pjoin(hml3d_opt.data_root, f"{split}.txt")
    out_dir = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, 'VA_motion', 'eval', 'splits')
    os.makedirs(out_dir, exist_ok=True)
    out_split = pjoin(out_dir, f"{split}_{task_type}.txt")

    with open(src_split, 'r') as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]

    if task_type == "generation":
        kept = [name for name in ids if not is_editing_id(name, motionfix_start_id)]
    else:
        kept = [name for name in ids if is_editing_id(name, motionfix_start_id)]

    with open(out_split, 'w', encoding='utf-8') as f:
        f.write("\n".join(kept))

    print(f'{task_type} split: {len(kept)} / {len(ids)} samples -> {out_split}')
    return out_split


def build_generation_loader(trans_cfg, hml3d_opt, mean, std, split, batch_size, num_workers, motionfix_start_id):
    # Generation uses the original HumanML3D eval dataset to keep official text-motion metrics
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w_vectorizer = WordVectorizer(pjoin(project_root, 'glove'), 'our_vab')
    split_file = build_task_split(trans_cfg, hml3d_opt, split, "generation", motionfix_start_id)
    dataset = Text2MotionDatasetEval(hml3d_opt, mean, std, split_file, w_vectorizer)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers,
                        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    return loader


def build_editing_loader(trans_cfg, hml3d_opt, mean, std, split, batch_size, num_workers, motionfix_start_id):
    # Editing uses MotionFix/MotionLab-style source-target pairs only
    split_file = build_task_split(trans_cfg, hml3d_opt, split, "editing", motionfix_start_id)
    dataset = HML3DMotionEditDataset(hml3d_opt, mean, std, split_file, motionfix_start_id=motionfix_start_id)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers,
                        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    return loader


@torch.no_grad()
def evaluation_generation_hml(eval_loader, trans, vq_model, eval_wrapper, device, time_steps=10,
                              cond_scale=4, source_cond_scale=1.0, text_delta_scale=1.0, unit_length=4, temperature=1, topk_filter_thres=0.95,
                              gsample=True):
    trans.eval()
    vq_model.eval()

    text_emb_list, gt_emb_list, pred_emb_list = [], [], []
    r_precision_sum = np.zeros(3)
    matching_sum = 0.0
    nb_sample = 0

    for batch in tqdm(eval_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, _, _ = batch
        pose = pose.to(device).float()
        m_length = m_length.to(device).long()
        bs = pose.shape[0]
        has_source = torch.zeros(bs, device=device, dtype=torch.long)

        # Generate with null source condition, which is the generation branch of the current model
        mids = trans.generate(
            None, clip_text, m_length // unit_length, has_source, t_drop=0,
            timesteps=time_steps, cond_scale=cond_scale, source_cond_scale=source_cond_scale, text_delta_scale=text_delta_scale, temperature=temperature,
            topk_filter_thres=topk_filter_thres, gsample=gsample
        )
        pred_motion = vq_model.forward_decoder(mids, m_length.clone())

        text_emb, gt_emb = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        text_emb_pred, pred_emb = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)

        text_np_batch = text_emb_pred.cpu().numpy()
        pred_np_batch = pred_emb.cpu().numpy()
        r_precision_sum += calculate_R_precision(text_np_batch, pred_np_batch, top_k=3, sum_all=True)
        matching_sum += euclidean_distance_matrix(text_np_batch, pred_np_batch).trace()

        text_emb_list.append(text_emb_pred)
        gt_emb_list.append(gt_emb)
        pred_emb_list.append(pred_emb)
        nb_sample += bs

    if nb_sample == 0:
        print("No generation samples were found in eval_loader.")
        return None

    gt_np = torch.cat(gt_emb_list, dim=0).cpu().numpy()
    pred_np = torch.cat(pred_emb_list, dim=0).cpu().numpy()

    r_precision = r_precision_sum / nb_sample
    matching = matching_sum / nb_sample

    gt_mu, gt_cov = calculate_activation_statistics(gt_np)
    mu, cov = calculate_activation_statistics(pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity_times = 300 if nb_sample > 300 else min(100, nb_sample - 1)
    diversity = calculate_diversity(pred_np, diversity_times) if nb_sample > 2 else 0.0

    return {
        "r1": r_precision[0], "r2": r_precision[1], "r3": r_precision[2],
        "matching": matching, "fid": fid, "diversity": diversity, "num_samples": nb_sample,
    }


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    return np.std(values) * 1.96 / np.sqrt(len(values))


def summarize_metric(metrics_list, key):
    values = np.array([m[key] for m in metrics_list], dtype=np.float64)
    return np.mean(values), ci95(values)


def format_generation(metrics):
    return (
        f"\tGEN Samples: {metrics['num_samples']}\n"
        f"\tGEN R@1/2/3: {metrics['r1']:.4f} / {metrics['r2']:.4f} / {metrics['r3']:.4f}\n"
        f"\tGEN Matching: {metrics['matching']:.4f}, FID: {metrics['fid']:.4f}, Diversity: {metrics['diversity']:.4f}\n"
    )


def format_editing(metrics):
    return (
        f"\tEDIT Samples: {metrics['num_samples']}\n"
        f"\tEDIT G2T R@1/2/3: {metrics['g2t_r1']:.4f} / {metrics['g2t_r2']:.4f} / {metrics['g2t_r3']:.4f}, AvgR: {metrics['g2t_avgr']:.2f}\n"
        f"\tEDIT G2S R@1/2/3: {metrics['g2s_r1']:.4f} / {metrics['g2s_r2']:.4f} / {metrics['g2s_r3']:.4f}, AvgR: {metrics['g2s_avgr']:.2f}\n"
        f"\tEDIT TMR-FID: {metrics['fid']:.4f}, TMR-Diversity: {metrics['diversity']:.4f}\n"
    )


def write_summary(title, metrics_list, keys, file_obj):
    if len(metrics_list) == 0:
        return
    print(title)
    print(title, file=file_obj, flush=True)
    for label, key, precision in keys:
        mean, ci = summarize_metric(metrics_list, key)
        line = f"\t{label}: {mean:.{precision}f} +/- {ci:.{precision}f}"
        print(line)
        print(line, file=file_obj, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_cfg', type=str, default='./configs/train_vamotion_hml.yaml',
                        help='VAMotion motion editing training config')
    parser.add_argument('--eval_cfg', type=str, default='./configs/evaluation_motion_editing_hml.yaml',
                        help='Eval config: time_steps, cond_scales, repeat_time, sampling params')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override checkpoint filename, default follows eval_cfg.which_ckpt')
    parser.add_argument('--split', type=str, default=None,
                        help='Override eval_cfg.split, e.g. val or test')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override eval batch size, default uses training.batch_size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override eval_cfg.num_workers')
    parser.add_argument('--motionfix_start_id', type=int, default=None,
                        help='Override eval_cfg.motionfix_start_id')
    parser.add_argument('--eval_generation', action='store_true',
                        help='Evaluate generation samples with HumanML3D text-motion metrics')
    parser.add_argument('--eval_editing', action='store_true',
                        help='Evaluate editing samples with G2T/G2S motion retrieval metrics')
    args = parser.parse_args()

    trans_cfg = load_config(args.trans_cfg)
    eval_cfg = load_config(args.eval_cfg)

    # yaml controls default task selection; CLI task flags are explicit overrides
    if args.eval_generation or args.eval_editing:
        eval_generation, eval_editing = args.eval_generation, args.eval_editing
    else:
        eval_generation = getattr(eval_cfg, 'eval_generation', True)
        eval_editing = getattr(eval_cfg, 'eval_editing', True)

    device = torch.device(eval_cfg.device)
    if eval_cfg.device != 'cpu':
        torch.cuda.set_device(eval_cfg.device)

    vq_model, vq_cfg = load_vq_model(trans_cfg, device)
    ckpt_name = args.ckpt or eval_cfg.which_ckpt
    stage_dir = getattr(eval_cfg, 'stage_dir', 'VA_motion')
    trans = load_trans_model(trans_cfg, vq_cfg, ckpt_name, device, stage_dir=stage_dir)

    dataset_opt_path = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, 'Comp_v6_KLD005', 'opt.txt')
    hml3d_opt = prepare_hml_opt(get_opt(dataset_opt_path, device), trans_cfg)
    mean, std = load_mean_std(trans_cfg)
    eval_wrapper = EvaluatorModelWrapper(hml3d_opt)
    edit_eval_wrapper = eval_wrapper
    edit_wrapper_name = getattr(eval_cfg, 'edit_eval_wrapper', getattr(trans_cfg.inference, 'edit_eval_wrapper', 'tmr'))
    if edit_wrapper_name == 'tmr':
        edit_eval_wrapper = TMREvaluatorWrapper(
            tmr_root=trans_cfg.tmr.root,
            run_dir=trans_cfg.tmr.run_dir,
            ckpt_name=getattr(trans_cfg.tmr, 'ckpt_name', 'last'),
            device=device,
            mean=mean,
            std=std,
            input_is_normalized=getattr(trans_cfg.tmr, 'input_is_normalized', True)
        )
    split = args.split or getattr(eval_cfg, 'split', 'test')
    batch_size = args.batch_size or getattr(eval_cfg, 'batch_size', trans_cfg.training.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else getattr(eval_cfg, 'num_workers', 0)
    motionfix_start_id = args.motionfix_start_id if args.motionfix_start_id is not None else getattr(eval_cfg, 'motionfix_start_id', 400000)

    gen_loader = build_generation_loader(trans_cfg, hml3d_opt, mean, std, split, batch_size, num_workers, motionfix_start_id) if eval_generation else None
    edit_loader = build_editing_loader(trans_cfg, hml3d_opt, mean, std, split, batch_size, num_workers, motionfix_start_id) if eval_editing else None

    log_dir = pjoin(trans_cfg.exp.root_ckpt_dir, trans_cfg.data.name, 'VA_motion', 'eval')
    os.makedirs(log_dir, exist_ok=True)
    safe_ckpt_name = ckpt_name.replace(".tar", "").replace("/", "_").replace("\\", "_")
    log_name = f'{eval_cfg.ext}_gen_edit_{split}_{safe_ckpt_name}.log'
    out_path = pjoin(log_dir, log_name)
    f = open(out_path, 'w', encoding='utf-8')
    print(f'Logging to {out_path}')

    topkr = getattr(eval_cfg, 'topkr', 0.9)
    gsample = getattr(eval_cfg, 'gsample', True)
    temp = getattr(eval_cfg, 'temperature', 1)
    source_cond_scale = getattr(eval_cfg, 'source_cond_scale', 1.0)
    text_delta_scale = 1.0   # legacy model arg
    source_hint_ratio = getattr(eval_cfg, 'source_hint_ratio', 0.0)
    base_seed = getattr(trans_cfg.exp, 'seed', 3407)

    for cs in eval_cfg.cond_scales:
        for ts in eval_cfg.time_steps:
            gen_metrics_list, edit_metrics_list = [], []

            for i in range(eval_cfg.repeat_time):
                fixseed(base_seed + i)
                header = f'Guidance scale: {cs}, source_hint_ratio: {source_hint_ratio}, time step: {ts}, repeat: {i}'
                print(header)
                print(header, file=f, flush=True)

                if gen_loader is not None:
                    gen_metrics = evaluation_generation_hml(
                        gen_loader, trans, vq_model, eval_wrapper, device,
                        time_steps=ts, cond_scale=cs, unit_length=trans_cfg.data.unit_length,
                        source_cond_scale=source_cond_scale, text_delta_scale=text_delta_scale, temperature=temp, topk_filter_thres=topkr, gsample=gsample
                    )
                    if gen_metrics is not None:
                        gen_metrics_list.append(gen_metrics)
                        print(format_generation(gen_metrics))
                        print(format_generation(gen_metrics), file=f, flush=True)

                if edit_loader is not None:
                    edit_metrics = evaluation_motion_editing_hml(
                        edit_loader, trans, vq_model, writer=None, ep=i, eval_wrapper=edit_eval_wrapper, device=device,
                        time_steps=ts, cond_scale=cs, unit_length=trans_cfg.data.unit_length,
                        source_cond_scale=source_cond_scale, text_delta_scale=text_delta_scale,
                        source_hint_ratio=source_hint_ratio,
                        temperature=temp, topk_filter_thres=topkr, gsample=gsample, draw=False
                    )
                    if edit_metrics is not None:
                        edit_metrics_list.append(edit_metrics)
                        print(format_editing(edit_metrics))
                        print(format_editing(edit_metrics), file=f, flush=True)

            summary_head = f'=== Final Result  cs={cs}  source_scale={source_cond_scale}  source_hint_ratio={source_hint_ratio}  ts={ts}  split={split}  ckpt={ckpt_name} ==='
            print(summary_head)
            print(summary_head, file=f, flush=True)

            write_summary('--- Generation ---', gen_metrics_list, [
                ('R@1', 'r1', 4), ('R@2', 'r2', 4), ('R@3', 'r3', 4),
                ('Matching', 'matching', 4), ('FID', 'fid', 4), ('Diversity', 'diversity', 4),
            ], f)

            write_summary('--- Editing ---', edit_metrics_list, [
                ('G2T R@1', 'g2t_r1', 4), ('G2T R@2', 'g2t_r2', 4), ('G2T R@3', 'g2t_r3', 4), ('G2T AvgR', 'g2t_avgr', 2),
                ('G2S R@1', 'g2s_r1', 4), ('G2S R@2', 'g2s_r2', 4), ('G2S R@3', 'g2s_r3', 4), ('G2S AvgR', 'g2s_avgr', 2),
                ('TMR-FID', 'fid', 4), ('TMR-Diversity', 'diversity', 4),
            ], f)

    f.close()
    print('Done.')
