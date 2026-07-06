import argparse
import os
import random
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F

from configs.load_config import load_config
from model import VAMotion
from utils.fixseeds import fixseed
from utils.visualize_motion_editing_hml import load_mean_std, load_vq_model, build_dataset, select_indices


def to_wsl_path(path):
    if len(path) > 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path


def load_trans(cfg, vq_cfg, ckpt_path, device):
    cfg.vq = vq_cfg.quantizer
    cfg.vq.nb_code = vq_cfg.quantizer.nb_code

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["training_model"] if "training_model" in ckpt else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("text_emb.")}

    # old ckpt switch
    if "task_embed.weight" not in state:
        cfg.model.use_task_token = False

    trans = VAMotion(cfg=cfg, device=device, full_length=cfg.data.max_motion_length // cfg.data.unit_length)
    missing, unexpected = trans.load_state_dict(state, strict=False)
    old_prefix = ("text_delta_encoder.", "condition_encoder.", "edit_map_head.",
                  "part_gate_mlp.", "part_cond_mlp.", "part_text_mlp.", "part_source_mlp.",
                  "edit_loc_head.", "task_embed.", "text_delta_scale", "source_delta_scale")
    unexpected = [k for k in unexpected if not k.startswith(old_prefix)]
    print(f"loaded: {ckpt_path}")
    print(f"epoch: {ckpt.get('epoch', '?')}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if unexpected:
        print("unexpected sample:", unexpected[:5])
    return trans.to(device).eval()


@torch.no_grad()
def encode_text(trans, captions, has_source):
    text_tokens, text_mask = trans.text_emb.get_text_embeddings(captions)
    text_tokens = trans.text_adaptor(text_tokens)
    text_tokens = torch.where(text_mask.unsqueeze(-1).bool(), text_tokens, 0.0)
    text_tokens, text_mask = trans.add_task_token(text_tokens, text_mask, has_source)
    return text_tokens, text_mask


@torch.no_grad()
def single_branch_logits(trans, motion_ids, source_ids, text_tokens, toa_pe, source_pe,
                         source_mask, text_mask, padding_mask, has_source, use_text=True):
    # one branch only, avoids 3x CFG batch memory
    if use_text:
        input_text_embs = text_tokens
    else:
        input_text_embs = trans.null_text_embed.expand_as(text_tokens)
    logits, _, _ = trans.run_condition_branch(
        motion_ids, source_ids, input_text_embs, toa_pe, source_pe,
        source_mask, text_mask, padding_mask, has_source
    )
    return logits


def avg_logp(logits, target_ids, valid_mask, weight=None):
    logp = F.log_softmax(logits.permute(0, 2, 1), dim=-1)
    safe_ids = target_ids.masked_fill(~valid_mask.bool(), 0)
    safe_ids = safe_ids.clamp(min=0, max=logp.shape[-1] - 1)
    token_logp = logp.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
    if weight is None:
        weight = valid_mask.float()
    else:
        weight = weight.float() * valid_mask.float()
    return (token_logp * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


def make_batch(dataset, indices, device):
    samples = [dataset[i] for i in indices]
    batch = {
        "captions": [s[0] for s in samples],
        "src_motion": torch.from_numpy(np.stack([s[1] for s in samples])).float().to(device),
        "tgt_motion": torch.from_numpy(np.stack([s[2] for s in samples])).float().to(device),
        "m_length": torch.tensor([s[3] for s in samples]).long().to(device),
        "src_joints": torch.from_numpy(np.stack([s[4] for s in samples])).float().to(device),
        "tgt_joints": torch.from_numpy(np.stack([s[5] for s in samples])).float().to(device),
        "has_source": torch.tensor([s[6] for s in samples]).long().to(device),
        "src_m_length": torch.tensor([s[12] for s in samples]).long().to(device),
        "names": [s[8] for s in samples],
    }
    return batch


@torch.inference_mode()
def diagnose_batch(cfg, trans, vq_model, batch, device):
    captions = batch["captions"]
    shuffled = captions[1:] + captions[:1]

    target_code_idx, _ = vq_model.encode(batch["tgt_motion"][..., :cfg.data.dim_pose], batch["m_length"])
    source_code_idx, _ = vq_model.encode(batch["src_motion"][..., :cfg.data.dim_pose], batch["src_m_length"])

    m_lens = batch["m_length"] // cfg.data.unit_length
    src_m_lens = batch["src_m_length"] // cfg.data.unit_length
    target_ids, target_mask, toa_pe = trans.prepare_motion_ids(target_code_idx, m_lens)
    source_ids, source_mask, source_pe = trans.prepare_motion_ids(source_code_idx, src_m_lens)
    padding_mask = ~target_mask

    # full-mask target, pure condition test
    motion_ids = torch.where(target_mask, torch.full_like(target_ids, trans.mask_id), torch.full_like(target_ids, trans.pad_id))
    has_source = batch["has_source"].view(-1, 1).bool()
    text_tokens, text_mask = encode_text(trans, captions, has_source)
    shuf_tokens, shuf_mask = encode_text(trans, shuffled, has_source)

    correct_logits = single_branch_logits(trans, motion_ids, source_ids, text_tokens, toa_pe, source_pe,
                                          source_mask, text_mask, padding_mask, has_source, use_text=True)
    null_logits = single_branch_logits(trans, motion_ids, source_ids, text_tokens, toa_pe, source_pe,
                                       source_mask, text_mask, padding_mask, has_source, use_text=False)
    shuf_logits = single_branch_logits(trans, motion_ids, source_ids, shuf_tokens, toa_pe, source_pe,
                                       source_mask, shuf_mask, padding_mask, has_source, use_text=True)

    edit_prob = trans.build_token_edit_prob(batch["src_joints"], batch["tgt_joints"],
                                            m_lens, src_m_lens, batch["has_source"], target_mask)
    edit_weight = (0.1 * target_mask.float() + edit_prob).clamp(max=1.0)

    return {
        "all_correct": avg_logp(correct_logits, target_ids, target_mask).cpu(),
        "all_null": avg_logp(null_logits, target_ids, target_mask).cpu(),
        "all_shuffle": avg_logp(shuf_logits, target_ids, target_mask).cpu(),
        "edit_correct": avg_logp(correct_logits, target_ids, target_mask, edit_weight).cpu(),
        "edit_null": avg_logp(null_logits, target_ids, target_mask, edit_weight).cpu(),
        "edit_shuffle": avg_logp(shuf_logits, target_ids, target_mask, edit_weight).cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_vamotion_hml.yaml")
    parser.add_argument("--ckpt", type=str, default="/mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/checkpoint_dir/humanml3d/VA_motion/model/test/best.tar")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10306)
    parser.add_argument("--motionfix_start_id", type=int, default=400000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = to_wsl_path(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    device = torch.device(cfg.exp.device)
    fixseed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    mean, std = load_mean_std(cfg)
    vq_cfg = load_config(pjoin(cfg.vq_cfg_dir, "configs", cfg.vq_name))
    vq_model = load_vq_model(cfg, vq_cfg, device)
    trans = load_trans(cfg, vq_cfg, ckpt_path, device)
    dataset = build_dataset(cfg, mean, std, args.split, args.motionfix_start_id)
    indices = select_indices(dataset, "edit", 0, args.num_samples, random_select=True, seed=args.seed)
    random.shuffle(indices)
    print(f"samples selected: {len(indices)}, batch_size: {args.batch_size}")

    records = []
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
        if len(batch_indices) < 2:
            continue
        batch = make_batch(dataset, batch_indices, device)
        records.append(diagnose_batch(cfg, trans, vq_model, batch, device))
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not records:
        raise RuntimeError("No edit samples were diagnosed.")

    def cat(key):
        return torch.cat([r[key] for r in records], dim=0)

    all_correct, all_null, all_shuffle = cat("all_correct"), cat("all_null"), cat("all_shuffle")
    edit_correct, edit_null, edit_shuffle = cat("edit_correct"), cat("edit_null"), cat("edit_shuffle")

    print("\nAll-token target logp")
    print(f"  correct/null/shuffle: {all_correct.mean():.4f} / {all_null.mean():.4f} / {all_shuffle.mean():.4f}")
    print(f"  correct-null: {(all_correct - all_null).mean():.4f}")
    print(f"  correct-shuffle: {(all_correct - all_shuffle).mean():.4f}")

    print("\nEdit-weighted target logp")
    print(f"  correct/null/shuffle: {edit_correct.mean():.4f} / {edit_null.mean():.4f} / {edit_shuffle.mean():.4f}")
    print(f"  correct-null: {(edit_correct - edit_null).mean():.4f}")
    print(f"  correct-shuffle: {(edit_correct - edit_shuffle).mean():.4f}")


if __name__ == "__main__":
    main()
