import os, argparse, shutil
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
from os.path import join as pjoin
from configs.load_config import load_config
from SnapMogen_model.vq.rvq_model import HRVQVAE
from utils.fixseeds import *
from model import VAMotion
from VA_trainer import VA_motion_trainer
from utils.get_opt import get_opt

DATASET_HML3D     = 'hml3d'
DATASET_SNAPMOGEN = 'snapmogen'


class TaskRatioSampler(Sampler):
    def __init__(self, is_edit, num_samples, edit_ratio=0.5, replacement=True):
        self.is_edit = np.asarray(is_edit, dtype=np.int64)
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.n_edit = int(self.is_edit.sum())
        self.n_gen = int(len(self.is_edit) - self.n_edit)
        self.set_edit_ratio(edit_ratio)

    def set_edit_ratio(self, edit_ratio):
        self.edit_ratio = float(min(max(edit_ratio, 0.0), 1.0))

    def __iter__(self):
        if self.n_edit > 0 and self.n_gen > 0:
            weights = np.where(
                self.is_edit == 1,
                self.edit_ratio / self.n_edit,
                (1.0 - self.edit_ratio) / self.n_gen,
            )
        else:
            weights = np.ones_like(self.is_edit, dtype=np.float64) / max(len(self.is_edit), 1)
        indices = torch.multinomial(
            torch.from_numpy(weights).double(),
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def copy_runtime_training_overrides(saved_cfg, runtime_cfg):
    if hasattr(runtime_cfg.training, "curriculum"):
        saved_cfg.training.curriculum = runtime_cfg.training.curriculum
    for key in ("max_epoch", "edit_sample_ratio", "m_drop", "v_drop", "t_drop", "s_drop", "mask_lo", "mask_hi", "mask_schedule"):
        if key in runtime_cfg.training:
            saved_cfg.training[key] = runtime_cfg.training[key]
    for key in (
        "weight_delta", "weight_latent", "weight_rank", "weight_null_gain", "weight_same_src",
        "null_gain_margin", "null_gain_topk", "null_gain_temp", "eval_null_gain_metric",
    ):
        if key in runtime_cfg.loss:
            saved_cfg.loss[key] = runtime_cfg.loss[key]
    if "delta_beta" in runtime_cfg.model:
        saved_cfg.model.delta_beta = runtime_cfg.model.delta_beta
    return saved_cfg


def build_task_split(split_file, out_file, task_type, motionfix_start_id=400000):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    kept_ids = []
    for name in ids:
        raw = name[1:] if name.startswith("M") else name
        is_edit = raw.isdigit() and int(raw) >= motionfix_start_id
        if (task_type == "editing" and is_edit) or (task_type == "generation" and not is_edit):
            kept_ids.append(name)

    with open(out_file, "w") as f:
        f.write("\n".join(kept_ids))
    print(f"{task_type} split: {len(kept_ids)} / {len(ids)} -> {out_file}")
    return out_file


def load_vq_model(cfg, vq_path, device):
    vq_cfg = load_config(vq_path)
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

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model', cfg.vq_ckpt),
                      map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    for p in vq_model.parameters():
        p.requires_grad_(False)
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_cfg


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
    raise FileNotFoundError("Cannot find mean/std normalization files for motion data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default="./configs/train_vamotion_hml.yaml",
                        help='config file for VA motion training')
    parser.add_argument('--OnGoing_model',
                        type=str,
                        default=None,
                        help='the path for continually training model, keep latest if is None')
    parser.add_argument('--dataset',
                        type=str,
                        default=DATASET_HML3D,
                        choices=[DATASET_SNAPMOGEN, DATASET_HML3D],
                        help='which dataset to use for training (snapmogen | hml3d)')
    parser.add_argument('--gen_only',
                        action='store_true',
                        help='only use HML3D generation ids, skip MotionFix/edit ids')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=None,
                        help='override max_epoch for quick diagnosis')
    parser.add_argument('--stage',
                        type=str,
                        default='source',
                        choices=['source', 'text'],
                        help='kept for old CLI; only source-stage training is active')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_epoch is not None:
        cfg.training.max_epoch = args.max_epoch
    if args.stage != 'source':
        print("stage=text is disabled; keep source-branch stage only")
        args.stage = 'source'
    stage_dirname = 'VA_motion'
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, stage_dirname)

    print(f"dataset: {args.dataset}")
    print(f"gen_only: {args.gen_only}")
    print(f"max_epoch: {cfg.training.max_epoch}")

    if cfg.exp.is_continue:
        n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, os.path.basename(args.config)))
        n_cfg = copy_runtime_training_overrides(n_cfg, cfg)
        vq_cfg_path = pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name)
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
        cfg = n_cfg
    else:
        vq_cfg_path = pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name)
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy(args.config, cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)
    if cfg.debug.open:
        torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name)
    os.makedirs(cfg.exp.model_dir, exist_ok=True)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)
    os.makedirs(cfg.exp.log_dir, exist_ok=True)

    mean, std = load_mean_std(cfg)
    vq_model, vq_cfg = load_vq_model(cfg, vq_cfg_path, device=device)
    cfg.vq = vq_cfg.quantizer
    cfg.vq.nb_code = vq_cfg.quantizer.nb_code

    if args.dataset == DATASET_HML3D:
        from SnapMogen_model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper

        dataset_opt_path = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "Comp_v6_KLD005", "opt.txt")
        hml3d_opt = get_opt(dataset_opt_path, device)
        motionfix_start_id = getattr(cfg.data, "motionfix_start_id", 400000)
        train_split = pjoin(cfg.data.root_dir, "HumanML3D", 'train.txt')
        val_split   = pjoin(cfg.data.root_dir, "HumanML3D", 'val.txt')
        gen_eval_split = build_task_split(val_split, pjoin(cfg.exp.eval_dir, "val_generation_eval.txt"), "generation", motionfix_start_id)
        edit_eval_split = build_task_split(val_split, pjoin(cfg.exp.eval_dir, "val_editing_eval.txt"), "editing", motionfix_start_id)
        if args.gen_only:
            train_split = build_task_split(train_split, pjoin(cfg.exp.eval_dir, "train_generation_only.txt"), "generation", motionfix_start_id)
            val_split = build_task_split(val_split, pjoin(cfg.exp.eval_dir, "val_generation_only.txt"), "generation", motionfix_start_id)
            gen_eval_split = val_split
            edit_eval_split = val_split

        eval_wrapper = EvaluatorModelWrapper(hml3d_opt)
        edit_eval_wrapper = eval_wrapper
        if getattr(cfg.inference, "edit_eval_wrapper", "tmr") == "tmr":
            from evaluator.tmr_eval_wrapper import TMREvaluatorWrapper
            edit_eval_wrapper = TMREvaluatorWrapper(
                tmr_root=cfg.tmr.root,
                run_dir=cfg.tmr.run_dir,
                ckpt_name=getattr(cfg.tmr, "ckpt_name", "last"),
                device=device,
                mean=mean,
                std=std,
                input_is_normalized=getattr(cfg.tmr, "input_is_normalized", True)
            )
    else:
        raise NotImplementedError("The current motion editing model no longer accepts the old SnapMogen video dataset batch.")

    TV2m_transformer = VAMotion(
        cfg=cfg,
        device=device,
        full_length=cfg.data.max_motion_length // cfg.data.unit_length
    )
    if hasattr(TV2m_transformer, "set_vq_quantizer") and hasattr(vq_model, "quantizer"):
        TV2m_transformer.set_vq_quantizer(vq_model.quantizer)
    elif hasattr(TV2m_transformer, "set_vq_codebook") and hasattr(vq_model, "quantizer") and hasattr(vq_model.quantizer, "codebook"):
        TV2m_transformer.set_vq_codebook(vq_model.quantizer.codebook)
    pc_vq = sum(param.numel() for param in TV2m_transformer.parameters())
    print(f"num of transformer parameter: {pc_vq / 1000} k")

    n_trainable = sum(p.numel() for p in TV2m_transformer.parameters() if p.requires_grad)
    print(f"stage: {args.stage}, trainable parameter: {n_trainable / 1000} k")

    trainer = VA_motion_trainer(cfg, vq_model=vq_model, va_transformer=TV2m_transformer, eval_wrapper=eval_wrapper,
                                edit_eval_wrapper=edit_eval_wrapper, device=device, resume_ckpt=args.OnGoing_model)

    if args.dataset == DATASET_HML3D:
        from dataset.HumanML3D_dataset import HML3DMotionEditDataset, Text2MotionDatasetEval, collate_fn
        from utils.word_vectorizer import WordVectorizer
        hml3d_opt.root_dir = cfg.data.root_dir
        hml3d_opt.max_motion_length = cfg.data.max_motion_length
        hml3d_opt.unit_length = cfg.data.unit_length
        hml3d_opt.motion_dir = pjoin(cfg.data.root_dir, "HumanML3D", "new_joint_vecs")
        hml3d_opt.joint_dir = pjoin(cfg.data.root_dir, "HumanML3D", "new_joints")
        hml3d_opt.text_dir = pjoin(cfg.data.root_dir, "HumanML3D", "texts")
        hml3d_opt.motionfix_start_id = getattr(cfg.data, "motionfix_start_id", 400000)
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        train_dataset = HML3DMotionEditDataset(hml3d_opt, mean, std, train_split)
        val_dataset   = HML3DMotionEditDataset(hml3d_opt, mean, std, val_split, w_vectorizer=w_vectorizer)
        edit_eval_dataset = HML3DMotionEditDataset(hml3d_opt, mean, std, edit_eval_split, w_vectorizer=w_vectorizer)
        gen_eval_dataset = Text2MotionDatasetEval(hml3d_opt, mean, std, gen_eval_split, w_vectorizer)

    loader_collate_fn = collate_fn if args.dataset == DATASET_HML3D else None

    # task mix, keep gen dominant for regularization
    is_edit = train_dataset.get_task_flags()
    n_edit, n_gen = int(is_edit.sum()), int(len(is_edit) - is_edit.sum())
    print(f"train samples: gen {n_gen}, edit {n_edit}")
    if n_edit > 0 and n_gen > 0:
        edit_ratio = float(getattr(cfg.training, "edit_sample_ratio", 0.5))
        edit_ratio = min(max(edit_ratio, 0.0), 1.0)
        print(f"task sample ratio: gen {1.0 - edit_ratio:.2f}, edit {edit_ratio:.2f}")
        train_sampler = TaskRatioSampler(is_edit, num_samples=len(is_edit), edit_ratio=edit_ratio, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8, sampler=train_sampler, pin_memory=True, collate_fn=loader_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8, shuffle=True, pin_memory=True, collate_fn=loader_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8, shuffle=True, pin_memory=True, collate_fn=loader_collate_fn)
    # batch 32 = MotionFix batch-protocol pool for edit G2T/G2S retrieval
    eval_loader = DataLoader(edit_eval_dataset, batch_size=32, drop_last=False, num_workers=8, shuffle=False, pin_memory=True, collate_fn=loader_collate_fn)
    gen_eval_loader = DataLoader(gen_eval_dataset, batch_size=cfg.inference.matching_pool_size, drop_last=True, num_workers=8, shuffle=True, pin_memory=True, collate_fn=loader_collate_fn)

    trainer.train(train_loader=train_loader, val_loader=val_loader, eval_loader=eval_loader, gen_eval_loader=gen_eval_loader)
