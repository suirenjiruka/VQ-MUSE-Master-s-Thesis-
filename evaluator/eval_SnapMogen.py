import torch 
import os, argparse, math
from torch.utils.data import DataLoader
import numpy as np
from os.path import join as pjoin


from SnapMogen_model.evaluator.evaluator_wrapper import EvaluatorWrapper
from configs.load_config import load_config
from SnapMogen_model.vq.rvq_model import HRVQVAE
from model import VAMotion
from dataset.dataset import Text_2D_MotionDataset
from dataset.PR_VIPE_end import pr_vipe_infer
from evaluator.evaluator import evaluation_momask

if __name__ == "__main__": 
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default="./configs/evaluation.yaml",
                        help='config file for evaluation setting')
    parser.add_argument('--vqcfg',
                        type=str,
                        default="./configs/residual_vqvae.yaml",
                        help='config file for VQ model')
    parser.add_argument('--trans_cfg',
                        type=str,
                        default="./checkpoint_dir/snapmogen/VA_motion/train_vamotion.yaml",
                        help='config file for VA mottion training')
    
    args = parser.parse_args()
    config = load_config(args.config)
    vq_cfg = load_config(args.vqcfg)
    tran_cfg = load_config(args.trans_cfg)
    tran_cfg.vq = vq_cfg.quantizer

    device = config.device

    # generate predivtion motion
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
    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model', tran_cfg.vq_ckpt),
                            map_location=device, weights_only=True) # To future Ina， 正式training時要記得調整
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key]) # type: ignore

    trans = VAMotion(
       cfg = tran_cfg,
       device = device,
       full_length= tran_cfg.data.max_motion_length // tran_cfg.data.unit_length,
       max_video_len = (tran_cfg.data.max_motion_length // 7) + 2,  # The rermainder parts occupy one extra sequence frame at the head and tail respectively, hence +2. ex: (320 // 7) + 2 = 47
       video_joint_num= None
    )
    checkpoint = torch.load(pjoin(tran_cfg.exp.model_dir, 'snapmogen', 'VA_motion', "model", 'net_best_fid.tar'), map_location=device, weights_only=True)
    trans.load_state_dict(checkpoint["training_model"])

    # tran all data into GPU
    vq_model.to(device)
    trans.to(device)

    config.data_dir = pjoin(config.data.root_dir, "renamed_feats")
    meta_dir = pjoin(config.data.root_dir, 'meta_data')
    Mean = np.load(pjoin(meta_dir, 'mean.npy'))
    Std = np.load(pjoin(meta_dir, 'std.npy'))
    # load data 
    data_split_dir = pjoin(tran_cfg.data.root_dir, 'data_split_info')
    test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')
    # load caprion
    all_caption_path = pjoin(tran_cfg.data.root_dir, 'all_caption_clean.json')

    eval_dataset = Text_2D_MotionDataset(tran_cfg, Mean, Std, test_mid_split_file,test_cid_split_file, all_caption_path, pr_vipe_infer)
    eval_cfg = load_config(tran_cfg.exp.evaluator_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_cfg.matching_pool_size, drop_last=True, num_workers=2,
                              shuffle=True, pin_memory=True)
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    out_path = pjoin(tran_cfg.exp.model_dir, 'snapmogen', 'VA_motion', 'evaluation.log')
    f = open(pjoin(out_path), 'w')

    t_drop = 0
    v_drop = 0

    vs = config.video_emphasis

    for cs in config.cond_scales:
            for ts in config.time_steps:
                fid = []
                div = []
                top1 = []
                top2 = []
                top3 = []
                matching = []
                mm = []
                for i in range(config.repeat_time):
                    
                    print(f'Guidance scale: {cs}, time step: {ts}')
                    print(f'Guidance scale: {cs}, time step: {ts}', file=f, flush=True)

                    with torch.no_grad():
                        best_fid, best_div, Rprecision, best_matching, best_mm = (
                            evaluation_momask(
                                eval_loader,
                                vq_model,
                                trans,
                                i,
                                eval_wrapper=eval_wrapper,
                                time_steps=ts,
                                cond_scale=cs,
                                video_emphasis = vs,
                                t_drop = t_drop,   #eval 時是否接受text input
                                v_drop = v_drop,   #eval 時是否接受video input
                                temperature=config.temperature,
                                gsample=config.gsample,
                                topkr=config.topkr,
                                cal_mm=config.cal_mm,
                            )
                        )
                    fid.append(best_fid)
                    div.append(best_div)
                    top1.append(Rprecision[0])
                    top2.append(Rprecision[1])
                    top3.append(Rprecision[2])
                    matching.append(best_matching)
                    mm.append(best_mm)

                fid = np.array(fid)
                div = np.array(div)
                top1 = np.array(top1)
                top2 = np.array(top2)
                top3 = np.array(top3)
                matching = np.array(matching)
                mm = np.array(mm)

                print(f'final result (Guidance scale: {cs}, time step: {ts}):')
                print(f'final result Guidance scale: {cs}, time step: {ts}():', file=f, flush=True)

                msg_final = (
                    f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(config.repeat_time):.3f}\n"
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(config.repeat_time):.3f}\n"
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(config.repeat_time):.3f}, "
                    f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(config.repeat_time):.3f}, "
                    f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(config.repeat_time):.3f}\n"
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(config.repeat_time):.3f}\n"
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(config.repeat_time):.3f}\n\n"
                )
                # logger.info(msg_final)
                print(msg_final)
                print(msg_final, file=f, flush=True)

    f.close()
