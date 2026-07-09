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

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, plot_func, time_steps = 10,
                           cond_scale = 4, video_emphasis = 1.0, unit_length = 4, save_ckpt=False,  draw=True):
    trans.eval()
    vq_model.eval()
    # Initialization statistics
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # in valid dataset, we have both video and text input, default t_drop & v_drop = 0
    t_drop = 0
    v_drop = 0

    num_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        caption, motion, m_length, mean_gaussian, stddev_gaussian, v_length, joints, j_len, dropout = batch
        mean_gaussian = mean_gaussian.to(device).float()
        stddev_gaussian = stddev_gaussian.to(device).float()
        joints = joints.to(device).float()
        j_len = j_len.to(device).float()
        video_input = (mean_gaussian, stddev_gaussian) #according to our model input format

        v_length = v_length.to(device).float()
        B = motion.shape[0]
        motion = motion.to(device).float()
        m_length = m_length.to(device).long()

        #calculate fid encode: GT
        et, _ = eval_wrapper.encode_text(caption, sample_mean=True)
        # use skeleton feature only
        fid_em, em, _ = eval_wrapper.encode_motion(motion[..., :148], m_length, sample_mean=True)  # 148 ??

        #calculate fid encode: Prediction
        mids = trans.generate(video_input, caption, m_length//unit_length, v_length, joints, j_len, t_drop, v_drop, time_steps, cond_scale, video_emphasis, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_length, sample_mean=True)

        if dropout.any(): 
            mask = ~dropout.to(device).bool()
            et = et[mask]
            em = em[mask]
            em_pred = em_pred[mask]
            fid_em = fid_em[mask]
            fid_em_pred = fid_em_pred[mask]

        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        num_sample += B

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if num_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if num_sample > 300 else 100)

    R_precision_real = R_precision_real / num_sample
    R_precision = R_precision / num_sample

    matching_score_real = matching_score_real / num_sample
    matching_score_pred = matching_score_pred / num_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    
    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    if draw: print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw:print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            torch.save({"training_model":trans.state_dict(), "ep":ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw:print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw:print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw:print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw:print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw:print(msg)
        best_top3 = R_precision[2]

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching

@torch.no_grad()
def evaluation_momask(val_loader, vq_model, trans, repeat_id, eval_wrapper, 
                      time_steps, cond_scale, video_emphasis, temperature, topkr, t_drop, v_drop, unit_length = 4, gsample=True, cal_mm=True):  
    trans.eval()
    vq_model.eval()

    device = trans.device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 1
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        caption, motions, m_lengths, mean_gaussian, stddev_gaussian, v_length, joints, j_len, dropout = batch
        mean_gaussian = mean_gaussian.to(device).float()
        stddev_gaussian = stddev_gaussian.to(device).float()
        joints = joints.to(device).float()
        j_len = j_len.to(device).float()
        video_input = (mean_gaussian, stddev_gaussian) #according to our model input format

        # motions = motions[..., :148]
        v_length = v_length.to(device).float()
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(caption, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs = motions.shape[0]

        if i < num_mm_batch:  # only first batch for mm
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):

                mids = trans.generate(video_input, caption, m_lengths//unit_length, v_length, joints, j_len, t_drop, 
                                      v_drop, time_steps, cond_scale, video_emphasis, temperature=temperature, topk_filter_thres = topkr, gsample = gsample)

                pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(video_input, caption, m_lengths//unit_length, v_length, joints, j_len, t_drop, 
                                  v_drop, time_steps, cond_scale, video_emphasis, temperature=temperature, topk_filter_thres = topkr, gsample = gsample)

            pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

            fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # fid_em_pred, em_pred = fid_em, em
        # pose = pose.cuda().float()
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality

@torch.no_grad()
def evaluation_mask_transformer_hml(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, plot_func, time_steps = 10,
                           cond_scale = 4, video_emphasis = 1.0, unit_length = 4, save_ckpt=False,  draw=True):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        text_emb_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('text_emb.')]
        for e in text_emb_weights:
            del t2m_trans_state_dict[e]
        state = {
            'training_model': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_recon_list = []
    R_precision_real = 0
    R_precision = 0
    R_precision_recon = 0
    matching_score_real = 0
    matching_score_pred = 0
    matching_score_recon = 0

    t_drop = 0
    v_drop = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, mean_gaussian, stddev_gaussian, v_length, joints, j_len, dropout  = batch

        # m_length = m_length.cuda()
        # motions = motions.to(device).float().detach()

        mean_gaussian = mean_gaussian.to(device).float()
        stddev_gaussian = stddev_gaussian.to(device).float()
        joints = joints.to(device).float()
        j_len = j_len.to(device).float()
        video_input = (mean_gaussian, stddev_gaussian) #according to our model input format

        v_length = v_length.to(device).float()
        pose = pose.to(device).float().detach()
        m_length = m_length.to(device).long()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        # mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        mids = trans.generate(video_input, clip_text, m_length//unit_length, v_length, joints, j_len, t_drop, v_drop, time_steps, cond_scale, video_emphasis, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())

        # VQ-VAE reconstruction upper bound (matches SnapMogen official eval_vqvae_hml path)
        _, all_codes = vq_model.encode(pose[..., :263], m_length.clone())
        recon_motions = vq_model.decode(all_codes, m_length.clone())

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)
        et_recon, em_recon = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, recon_motions.clone(), m_length)

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        motion_recon_list.append(em_recon)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        temp_R = calculate_R_precision(et_recon.cpu().numpy(), em_recon.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_recon.cpu().numpy(), em_recon.cpu().numpy()).trace()
        R_precision_recon += temp_R
        matching_score_recon += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    motion_recon_np = torch.cat(motion_recon_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    mu_recon, cov_recon = calculate_activation_statistics(motion_recon_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    diversity_recon = calculate_diversity(motion_recon_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    R_precision_recon = R_precision_recon / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    matching_score_recon = matching_score_recon / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    fid_recon = calculate_frechet_distance(gt_mu, gt_cov, mu_recon, cov_recon)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)
    print(f"--> \t [VQ-VAE Recon Upper Bound] FID={fid_recon:.4f}, Diversity={diversity_recon:.4f}, R_precision={R_precision_recon}, matching={matching_score_recon:.4f}")

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir,  'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching: 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_vamotion_hml(eval_loader, vq_model, trans, repeat_id, eval_wrapper,
                             time_steps, cond_scale, video_emphasis, t_drop, v_drop,
                             unit_length=4, temperature=1, topkr=0.9, gsample=True, cal_mm=False):
    """
    Standalone evaluation for VAMotion on HumanML3D test set.
    Uses Euclidean-distance based R-precision (matching official HumanML3D eval protocol).
    Also reports VQ-VAE reconstruction upper bound alongside generation metrics.
    Returns: (fid, diversity, R_precision[3], matching_score, multimodality)
    """
    trans.eval()
    vq_model.eval()
    device = next(trans.parameters()).device

    motion_annotation_list = []
    motion_pred_list = []
    motion_recon_list = []
    motion_multimodality = []

    R_precision_real = 0
    R_precision = 0
    R_precision_recon = 0
    matching_score_real = 0
    matching_score_pred = 0
    matching_score_recon = 0
    multimodality = 0
    nb_sample = 0

    num_mm_batch = 1 if cal_mm else 0

    for i, batch in enumerate(tqdm(eval_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, _, \
            mean_gaussian, stddev_gaussian, v_length, joints, j_len, _ = batch

        mean_gaussian = mean_gaussian.to(device).float()
        stddev_gaussian = stddev_gaussian.to(device).float()
        joints = joints.to(device).float()
        j_len = j_len.to(device).float()
        video_input = (mean_gaussian, stddev_gaussian)

        v_length = v_length.to(device).float()
        pose = pose.to(device).float().detach()
        m_length = m_length.to(device).long()
        bs = pose.shape[0]

        # GT embeddings
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)

        # VQ-VAE reconstruction upper bound
        _, all_codes = vq_model.encode(pose[..., :263], m_length.clone())
        recon_motions = vq_model.decode(all_codes, m_length.clone())
        et_recon, em_recon = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, recon_motions.clone(), m_length)

        # Generation (with optional multimodality computation on first batch)
        if i < num_mm_batch:
            mm_batch = []
            for _ in range(30):
                mids = trans.generate(video_input, clip_text, m_length // unit_length,
                                      v_length, joints, j_len, t_drop, v_drop,
                                      time_steps, cond_scale, video_emphasis, temperature=temperature,
                                      topk_filter_thres=topkr, gsample=gsample)
                pred_mm = vq_model.forward_decoder(mids, m_length.clone())
                _, em_mm = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pred_mm.clone(), m_length)
                mm_batch.append(em_mm.unsqueeze(1))
            motion_multimodality.append(torch.cat(mm_batch, dim=1))
            # One extra generate for the main metrics
            mids = trans.generate(video_input, clip_text, m_length // unit_length,
                                  v_length, joints, j_len, t_drop, v_drop,
                                  time_steps, cond_scale, video_emphasis, temperature=temperature,
                                  topk_filter_thres=topkr, gsample=gsample)
        else:
            mids = trans.generate(video_input, clip_text, m_length // unit_length,
                                  v_length, joints, j_len, t_drop, v_drop,
                                  time_steps, cond_scale, video_emphasis, temperature=temperature,
                                  topk_filter_thres=topkr, gsample=gsample)

        pred_motions = vq_model.forward_decoder(mids, m_length.clone())
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        motion_recon_list.append(em_recon)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        temp_R = calculate_R_precision(et_recon.cpu().numpy(), em_recon.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_recon.cpu().numpy(), em_recon.cpu().numpy()).trace()
        R_precision_recon += temp_R
        matching_score_recon += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    motion_recon_np = torch.cat(motion_recon_list, dim=0).cpu().numpy()

    if cal_mm and motion_multimodality:
        mm_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(mm_np, 10)

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    mu_recon, cov_recon = calculate_activation_statistics(motion_recon_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    diversity_recon = calculate_diversity(motion_recon_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    R_precision_recon = R_precision_recon / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    matching_score_recon = matching_score_recon / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    fid_recon = calculate_frechet_distance(gt_mu, gt_cov, mu_recon, cov_recon)

    print(
        f"--> Repeat {repeat_id}: FID={fid:.4f}  Div_real={diversity_real:.4f}  Div={diversity:.4f}  "
        f"R@1={R_precision[0]:.4f}  R@2={R_precision[1]:.4f}  R@3={R_precision[2]:.4f}  "
        f"Match={matching_score_pred:.4f}  MM={multimodality:.4f}"
    )
    print(
        f"    [VQ Recon UB]  FID={fid_recon:.4f}  Div={diversity_recon:.4f}  "
        f"R@3={R_precision_recon[2]:.4f}  Match={matching_score_recon:.4f}"
    )

    return fid, diversity, R_precision, matching_score_pred, multimodality
