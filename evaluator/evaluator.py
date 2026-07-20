import os
import time
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import *

'''
Code is borrowed from the framework of SnapMogen, including VQ-model, traing process, and most relatted configuratioon
'''

def sync_cuda_if_needed(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

def calculate_retrieval_metrics(query_emb, gallery_emb, top_k=3):
    dist_mat = euclidean_distance_matrix(query_emb, gallery_emb)
    ranking = np.argsort(dist_mat, axis=1)
    gt_index = np.arange(query_emb.shape[0])
    top_k_acc = [(ranking[:, :k] == gt_index[:, None]).any(axis=1).mean() for k in range(1, top_k + 1)]
    avg_rank = np.where(ranking == gt_index[:, None])[1].mean() + 1
    return np.array(top_k_acc), avg_rank

def get_motion_embeddings_aligned(eval_wrapper, motions, m_lens):
    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
    inv_idx = np.argsort(align_idx)
    emb = eval_wrapper.get_motion_embeddings(motions, m_lens)
    return emb[inv_idx]

@torch.no_grad()
def evaluation_generation_hml_mixed(eval_loader, trans, vq_model, writer, ep, eval_wrapper, device, time_steps=10, cond_scale=4, source_cond_scale=1.0, unit_length=4, temperature=1,
                                    topk_filter_thres=0.95, gsample=True, draw=True):
    if draw:
        print("Mixed generation eval is disabled for HML3DMotionEditDataset; use Text2MotionDatasetEval instead.")
    return None

@torch.no_grad()
def evaluation_generation_hml(eval_loader, trans, vq_model, writer, ep, eval_wrapper, device, time_steps=10, cond_scale=4, source_cond_scale=1.0, unit_length=4, temperature=1,
                              topk_filter_thres=0.95, gsample=True, draw=True):
    trans.eval()
    vq_model.eval()

    text_emb_list, gt_emb_list, pred_emb_list = [], [], []
    r_precision_sum = np.zeros(3)
    matching_sum = 0.0
    nb_sample = 0
    inference_seconds = 0.0

    for batch in tqdm(eval_loader):
        word_embeddings, pos_one_hots, caption, sent_len, tgt_motion, m_length, _, _ = batch
        tgt_motion = tgt_motion.to(device).float()
        m_length = m_length.to(device).long()
        gen_has_source = torch.zeros_like(m_length, device=device)

        sync_cuda_if_needed(device)
        start_time = time.perf_counter()
        mids = trans.generate(
            None, caption, m_length // unit_length, gen_has_source, t_drop=0,
            timesteps=time_steps, cond_scale=cond_scale, source_cond_scale=source_cond_scale, temperature=temperature,
            topk_filter_thres=topk_filter_thres, gsample=gsample
        )
        pred_motion = vq_model.forward_decoder(mids, m_length.clone())
        sync_cuda_if_needed(device)
        inference_seconds += time.perf_counter() - start_time

        text_emb, gt_emb = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, tgt_motion, m_length)
        text_emb_pred, pred_emb = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)

        text_np_batch = text_emb_pred.cpu().numpy()
        pred_np_batch = pred_emb.cpu().numpy()
        r_precision_sum += calculate_R_precision(text_np_batch, pred_np_batch, top_k=3, sum_all=True)
        matching_sum += euclidean_distance_matrix(text_np_batch, pred_np_batch).trace()

        text_emb_list.append(text_emb_pred)
        gt_emb_list.append(gt_emb)
        pred_emb_list.append(pred_emb)
        nb_sample += len(pred_emb)

    if nb_sample == 0:
        print("No generation samples were found in eval_loader; skip generation evaluator.")
        return None

    gt_np = torch.cat(gt_emb_list, dim=0).cpu().numpy()
    pred_np = torch.cat(pred_emb_list, dim=0).cpu().numpy()

    r_precision = r_precision_sum / nb_sample
    matching = matching_sum / nb_sample
    gt_mu, gt_cov = calculate_activation_statistics(gt_np)
    mu, cov = calculate_activation_statistics(pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(pred_np, 300 if nb_sample > 300 else min(100, nb_sample - 1)) if nb_sample > 2 else 0.0

    metrics = {
        "r1": r_precision[0], "r2": r_precision[1], "r3": r_precision[2],
        "matching": matching, "fid": fid, "diversity": diversity, "num_samples": nb_sample,
        "aits": inference_seconds / nb_sample,
    }

    if draw:
        print(
            f"--> Gen Eval Ep {ep}: "
            f"R@1/2/3 {r_precision[0]:.4f}/{r_precision[1]:.4f}/{r_precision[2]:.4f}; "
            f"Matching {matching:.4f}; FID {fid:.4f}; Diversity {diversity:.4f}; AITS {metrics['aits']:.4f}s"
        )
        writer.add_scalar('GenEval/R1', metrics["r1"], ep)
        writer.add_scalar('GenEval/R2', metrics["r2"], ep)
        writer.add_scalar('GenEval/R3', metrics["r3"], ep)
        writer.add_scalar('GenEval/Matching', metrics["matching"], ep)
        writer.add_scalar('GenEval/FID', metrics["fid"], ep)
        writer.add_scalar('GenEval/Diversity', metrics["diversity"], ep)
        writer.add_scalar('GenEval/AITS', metrics["aits"], ep)

    return metrics

@torch.no_grad()
def evaluation_motion_editing_hml(eval_loader, trans, vq_model, writer, ep, eval_wrapper, device, time_steps=10, cond_scale=4, source_cond_scale=2.0, unit_length=4, temperature=1,
                                  topk_filter_thres=0.95, gsample=True, draw=True):
    trans.eval()
    vq_model.eval()

    source_emb_list = []
    target_emb_list = []
    pred_emb_list = []
    batch_g2t_r = []
    batch_g2t_rank = []
    batch_g2s_r = []
    batch_g2s_rank = []
    batch_nb_sample = 0
    nb_sample = 0
    inference_seconds = 0.0

    for batch in tqdm(eval_loader):
        caption, src_motion, tgt_motion, m_length, has_source, src_m_length = batch[:6]
        has_source_mask = has_source.bool()
        if not has_source_mask.any():
            continue

        caption = [cap for cap, keep in zip(caption, has_source_mask.tolist()) if keep]
        src_motion = src_motion[has_source_mask].to(device).float()
        tgt_motion = tgt_motion[has_source_mask].to(device).float()
        m_length = m_length[has_source_mask].to(device).long()
        src_m_length = src_m_length[has_source_mask].to(device).long()
        edit_has_source = torch.ones_like(m_length, device=device)

        # source encode, edit generate, decode
        sync_cuda_if_needed(device)
        start_time = time.perf_counter()
        source_code_idx, _ = vq_model.encode(src_motion[..., :trans.cfg.data.dim_pose], src_m_length)
        mids = trans.generate(
            source_code_idx, caption, m_length // unit_length, edit_has_source, t_drop=0,
            timesteps=time_steps, cond_scale=cond_scale, source_cond_scale=source_cond_scale,
            source_m_lens=src_m_length // unit_length, temperature=temperature,
            topk_filter_thres=topk_filter_thres, gsample=gsample
        )
        pred_motion = vq_model.forward_decoder(mids, m_length.clone())
        sync_cuda_if_needed(device)
        inference_seconds += time.perf_counter() - start_time

        # MotionFix-style retrieval is motion-to-motion: generated -> target/source
        source_emb = get_motion_embeddings_aligned(eval_wrapper, src_motion, src_m_length)
        target_emb = get_motion_embeddings_aligned(eval_wrapper, tgt_motion, m_length)
        pred_emb = get_motion_embeddings_aligned(eval_wrapper, pred_motion, m_length)

        source_np = source_emb.cpu().numpy()
        target_np = target_emb.cpu().numpy()
        pred_np = pred_emb.cpu().numpy()

        # batch-32 retrieval protocol; keep partial tail only for global monitor
        if len(pred_np) == 32:
            g2t_r, g2t_rank = calculate_retrieval_metrics(pred_np, target_np)
            g2s_r, g2s_rank = calculate_retrieval_metrics(pred_np, source_np)
            batch_g2t_r.append(g2t_r * len(pred_np))
            batch_g2t_rank.append(g2t_rank * len(pred_np))
            batch_g2s_r.append(g2s_r * len(pred_np))
            batch_g2s_rank.append(g2s_rank * len(pred_np))
            batch_nb_sample += len(pred_np)

        source_emb_list.append(source_emb)
        target_emb_list.append(target_emb)
        pred_emb_list.append(pred_emb)
        nb_sample += len(pred_np)

    if nb_sample == 0:
        print("No editing samples were found in eval_loader; skip motion editing evaluator.")
        return None

    source_np = torch.cat(source_emb_list, dim=0).cpu().numpy()
    target_np = torch.cat(target_emb_list, dim=0).cpu().numpy()
    pred_np = torch.cat(pred_emb_list, dim=0).cpu().numpy()

    global_g2t_r, global_g2t_rank = calculate_retrieval_metrics(pred_np, target_np)
    global_g2s_r, global_g2s_rank = calculate_retrieval_metrics(pred_np, source_np)

    if batch_nb_sample > 0:
        batch_g2t_r = np.stack(batch_g2t_r).sum(axis=0) / batch_nb_sample
        batch_g2s_r = np.stack(batch_g2s_r).sum(axis=0) / batch_nb_sample
        batch_g2t_rank = np.sum(batch_g2t_rank) / batch_nb_sample
        batch_g2s_rank = np.sum(batch_g2s_rank) / batch_nb_sample
    else:
        batch_g2t_r = global_g2t_r
        batch_g2s_r = global_g2s_r
        batch_g2t_rank = global_g2t_rank
        batch_g2s_rank = global_g2s_rank

    gt_mu, gt_cov = calculate_activation_statistics(target_np)
    mu, cov = calculate_activation_statistics(pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(pred_np, 300 if nb_sample > 300 else min(100, nb_sample - 1)) if nb_sample > 2 else 0.0

    metrics = {
        "g2t_r1": batch_g2t_r[0], "g2t_r2": batch_g2t_r[1], "g2t_r3": batch_g2t_r[2], "g2t_avgr": batch_g2t_rank,
        "g2s_r1": batch_g2s_r[0], "g2s_r2": batch_g2s_r[1], "g2s_r3": batch_g2s_r[2], "g2s_avgr": batch_g2s_rank,
        "global_g2t_r1": global_g2t_r[0], "global_g2t_r2": global_g2t_r[1], "global_g2t_r3": global_g2t_r[2], "global_g2t_avgr": global_g2t_rank,
        "global_g2s_r1": global_g2s_r[0], "global_g2s_r2": global_g2s_r[1], "global_g2s_r3": global_g2s_r[2], "global_g2s_avgr": global_g2s_rank,
        "fid": fid, "diversity": diversity, "num_samples": nb_sample, "batch_num_samples": batch_nb_sample,
        "aits": inference_seconds / nb_sample,
    }

    if draw:
        print(
            f"--> MotionEdit Eval Ep {ep}: "
            f"G2T R@1/2/3 {batch_g2t_r[0]:.4f}/{batch_g2t_r[1]:.4f}/{batch_g2t_r[2]:.4f}, AvgR {batch_g2t_rank:.2f}; "
            f"G2S R@1/2/3 {batch_g2s_r[0]:.4f}/{batch_g2s_r[1]:.4f}/{batch_g2s_r[2]:.4f}, AvgR {batch_g2s_rank:.2f}; "
            f"TMR-FID {fid:.4f}, TMR-Diversity {diversity:.4f}; AITS {metrics['aits']:.4f}s"
        )
        print(
            f"    Global monitor: "
            f"G2T R@1/2/3 {global_g2t_r[0]:.4f}/{global_g2t_r[1]:.4f}/{global_g2t_r[2]:.4f}, AvgR {global_g2t_rank:.2f}; "
            f"G2S R@1/2/3 {global_g2s_r[0]:.4f}/{global_g2s_r[1]:.4f}/{global_g2s_r[2]:.4f}, AvgR {global_g2s_rank:.2f}"
        )
        writer.add_scalar('EditEval/G2T_R1', metrics["g2t_r1"], ep)
        writer.add_scalar('EditEval/G2T_R2', metrics["g2t_r2"], ep)
        writer.add_scalar('EditEval/G2T_R3', metrics["g2t_r3"], ep)
        writer.add_scalar('EditEval/G2T_AvgR', metrics["g2t_avgr"], ep)
        writer.add_scalar('EditEval/G2S_R1', metrics["g2s_r1"], ep)
        writer.add_scalar('EditEval/G2S_R2', metrics["g2s_r2"], ep)
        writer.add_scalar('EditEval/G2S_R3', metrics["g2s_r3"], ep)
        writer.add_scalar('EditEval/G2S_AvgR', metrics["g2s_avgr"], ep)
        writer.add_scalar('EditEval_Global/G2T_R1', metrics["global_g2t_r1"], ep)
        writer.add_scalar('EditEval_Global/G2S_R1', metrics["global_g2s_r1"], ep)
        writer.add_scalar('EditEval/TMR_FID', metrics["fid"], ep)
        writer.add_scalar('EditEval/TMR_Diversity', metrics["diversity"], ep)
        writer.add_scalar('EditEval/AITS', metrics["aits"], ep)

    return metrics
