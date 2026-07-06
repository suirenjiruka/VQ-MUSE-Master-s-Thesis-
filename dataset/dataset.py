import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
from typing import Callable
import numpy as np
from torch.utils import data
from os.path import join as pjoin
import random
import json
from dataset.PR_VIPE_end import read_input, pr_vipe_infer
from model import  Reparameter_sampler
from dataset.SnapMogen_dataset import TextMotionDataset
from configs.load_config import load_config

class Text_joint_MotionDataset(TextMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path, all_caption_path)
        drop = {}
        data_2D = {}
        for i, mid in enumerate(self.mid_list):
            try:
                feat2D_path = pjoin(cfg.data.feat2D_dir, "%s.json" % mid)  #please ensure feat2D_dir in cfg.data
                data = np.array(read_input(feat2D_path))
                data_2D[mid] = data.reshape(-1, 30)
                drop[mid] = 0
            except:
               data = np.zeros([1,7,15,2], dtype=float)
               data_2D[mid] = data.reshape(-1, 30)
               drop[mid] = 1   #這邊dropout是整個video input當成沒有
        self.data_2D = data_2D
        self.drop = drop
        del data_2D

    def __getitem__(self, item):
        caption, motion, m_length, idx = super().__getitem__(item)

        cid = self.cid_list[item]
        mid, start, end = cid.split("#")

        video_joints = self.data_2D[mid][int(start) + idx: int(start) + idx + m_length]
        v_len = len(video_joints)
        if v_len < self.cfg.data.max_motion_length:
           video_joints = np.concatenate(
                [
                    video_joints,
                    np.zeros(
                        (self.cfg.data.max_motion_length - v_len, video_joints.shape[1])
                    ),
                ],
                axis=0,
            )

        dropout = self.drop[mid]  #2d features missing, viewed as dropout (less than 5%)

        #print(motion.shape)
        #print(mid, video_gaussian[0].shape, dropout, v_len)

        #ensure all the input is tenor.float(), except dropout(int)
        return caption, motion, m_length, video_joints, v_len, dropout


class Text_2D_MotionDataset(TextMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path, data_2D_handler: Callable):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path, all_caption_path)
        drop = {}
        data_2D = {}
        joint = {}
        for i, mid in enumerate(self.mid_list):
            try:
                feat2D_path = pjoin(cfg.data.feat2D_dir, "%s.json" % mid)  #please ensure feat2D_dir in cfg.data
                data = np.array(read_input(feat2D_path))
                data_2D[mid] = data
                joint[mid] = data.reshape(-1, 30)
                drop[mid] = 0
            except:
               data = np.zeros([1,7,15,2], dtype=float)
               data_2D[mid] = data
               joint[mid] = data.reshape(-1, 30)
               drop[mid] = 1   #這邊dropout是整個video input當成沒有
        Gaussian_data_2D = data_2D_handler(cfg.vipe_checkpt, data_2D)  #handler should follow this input format (str, dict) -> dict
        self.data_2D = Gaussian_data_2D
        self.joint = joint
        self.drop = drop
        del data_2D, joint

    def __getitem__(self, item):
        caption, motion, m_length, idx = super().__getitem__(item)

        cid = self.cid_list[item]
        mid, start, end = cid.split("#")

        video_gaussian = (self.data_2D[mid][0][(int(start) + idx) // 7: (((int(start) + idx) + m_length) // 7) + 1], # 7 is the temporal length in Pr-vipe 
                          self.data_2D[mid][1][(int(start) + idx) // 7: (((int(start) + idx)+m_length) // 7) + 1])   # tuple ([m_length / 7, 19], [m_length / 7, 19])
        v_len = len(video_gaussian[0])
        # (self.cfg.data.max_motion_length // 7) + 2， 除不盡的部分各在頭尾多佔1個sequence frame的video block，所以+2
        if(v_len < (self.cfg.data.max_motion_length // 7) + 2):
            video_gaussian = (np.concatenate([video_gaussian[0], np.zeros(((self.cfg.data.max_motion_length // 7) + 2 - v_len, video_gaussian[0].shape[1]), dtype=float)],axis=0).astype(np.float32),
                            np.concatenate([video_gaussian[1], np.zeros(((self.cfg.data.max_motion_length // 7) + 2 - v_len, video_gaussian[1].shape[1]), dtype=float)],axis=0).astype(np.float32))
        #raw joint data concatenate to len
        video_joints = self.joint[mid][int(start) + idx: int(start) + idx + m_length]
        joint_len = len(video_joints)
        if joint_len < self.cfg.data.max_motion_length:
           video_joints = np.concatenate(
                [
                    video_joints,
                    np.zeros(
                        (self.cfg.data.max_motion_length - joint_len, video_joints.shape[1])
                    ),
                ],
                axis=0,
            )

        dropout = self.drop[mid]  #2d features missing, viewed as dropout (less than 5%)

        #print(motion.shape)
        #print(mid, video_gaussian[0].shape, dropout, v_len)

        #ensure all the input is tenor.float(), except dropout(int)
        return caption, motion, m_length, video_gaussian[0], video_gaussian[1], v_len, video_joints, joint_len, dropout


class Text_SMPL_VIPE_MotionDataset(TextMotionDataset):
    # LEGACY_2DH36M13 order: Head, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, RKnee, LFoot, RFoot.
    # Assumes SMPL order: Pelvis, LHip, RHip, Spine1, LKnee, RKnee, Spine2, LAnkle, RAnkle, Spine3, LFoot, RFoot, Neck, LCollar, RCollar, Head, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist.
    SMPL_TO_H36M13 = [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 10, 11]
    GLOBAL_DESCRIPTOR_INDEX = 22
    FRAME_LENGTH = 7
    RAW_JOINT_DIM = 23 * 3

    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path, data_2D_handler: Callable):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path, all_caption_path)
        drop = {}
        data_2D = {}
        joint = {}
        for mid in tqdm(self.mid_list):
            try:
                feat2D_path = pjoin(cfg.data.feat2D_dir, "%s.json" % mid)
                smpl_data = np.array(read_input(feat2D_path), dtype=np.float32)
                self._validate_smpl_data(smpl_data, mid)
                data_2D[mid] = self._to_pr_vipe_15x2(smpl_data)
                joint[mid] = smpl_data.reshape(-1, self.RAW_JOINT_DIM)
                drop[mid] = 0
            except Exception as e:
               print(f"[SMPL dataset drop] {mid}: {e}")
               smpl_data = np.zeros([1, self.FRAME_LENGTH, 23, 3], dtype=np.float32)
               data_2D[mid] = self._to_pr_vipe_15x2(smpl_data)
               joint[mid] = smpl_data.reshape(-1, self.RAW_JOINT_DIM)
               drop[mid] = 1
        print(f"SMPL video drops: {sum(drop.values())}/{len(drop)}")
        Gaussian_data_2D = data_2D_handler(cfg.vipe_checkpt, data_2D)
        self.data_2D = Gaussian_data_2D
        self.joint = joint
        
        self.drop = drop
        del data_2D, joint

    @classmethod
    def _validate_smpl_data(cls, smpl_data, mid):
        if smpl_data.ndim != 4 or smpl_data.shape[1:] != (cls.FRAME_LENGTH, 23, 3):
            raise ValueError(f"{mid}: expected shape (blocks, 7, 23, 3), got {smpl_data.shape}")

    @classmethod
    def _to_pr_vipe_15x2(cls, smpl_data):
        vipe_joints = smpl_data[:, :, cls.SMPL_TO_H36M13, :2]
        global_xyz = smpl_data[:, :, cls.GLOBAL_DESCRIPTOR_INDEX, :3]
        global_pseudo = np.zeros([smpl_data.shape[0], cls.FRAME_LENGTH, 2, 2], dtype=smpl_data.dtype)
        global_pseudo[:, :, 0, 0] = global_xyz[:, :, 0]
        global_pseudo[:, :, 0, 1] = global_xyz[:, :, 1]
        global_pseudo[:, :, 1, 0] = global_xyz[:, :, 2]
        return np.concatenate([vipe_joints, global_pseudo], axis=2).astype(np.float32)

    def __getitem__(self, item):
        caption, motion, m_length, idx = super().__getitem__(item)

        cid = self.cid_list[item]
        mid, start, end = cid.split("#")
        start = int(start)

        video_gaussian = (self.data_2D[mid][0][(start + idx) // self.FRAME_LENGTH: (((start + idx) + m_length) // self.FRAME_LENGTH) + 1],
                          self.data_2D[mid][1][(start + idx) // self.FRAME_LENGTH: (((start + idx) + m_length) // self.FRAME_LENGTH) + 1])
        v_len = len(video_gaussian[0])
        max_video_len = (self.cfg.data.max_motion_length // self.FRAME_LENGTH) + 2
        if(v_len < max_video_len):
            video_gaussian = (np.concatenate([video_gaussian[0], np.zeros((max_video_len - v_len, video_gaussian[0].shape[1]), dtype=float)], axis=0).astype(np.float32),
                              np.concatenate([video_gaussian[1], np.zeros((max_video_len - v_len, video_gaussian[1].shape[1]), dtype=float)], axis=0).astype(np.float32))

        video_joints = self.joint[mid][start + idx: start + idx + m_length]
        joint_len = len(video_joints)
        if joint_len < self.cfg.data.max_motion_length:
           video_joints = np.concatenate([video_joints, np.zeros((self.cfg.data.max_motion_length - joint_len, video_joints.shape[1]))], axis=0)

        dropout = self.drop[mid]
        return caption, motion, m_length, video_gaussian[0], video_gaussian[1], v_len, video_joints, joint_len, dropout

#datasett & TCN testing   
if __name__ == "__main__":   
    cfg = load_config('/mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/configs/train_vamotion.yaml')

    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info')
    train_mid_split_file = pjoin(data_split_dir, 'train_fnames.txt')
    train_cid_split_file = pjoin(data_split_dir, 'train_ids.txt')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    dataset_cls = Text_SMPL_VIPE_MotionDataset if cfg.data.get('video_joint_format', '2d15') == 'smpl23_3d' else Text_2D_MotionDataset
    train_dataset = dataset_cls(cfg, mean, std, train_mid_split_file, train_cid_split_file, all_caption_path, data_2D_handler= pr_vipe_infer)#
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    
    video_adaptor = Reparameter_sampler()

    i = 0
    for (caption, motion, m_length, mean_gaussian, stddev_gaussian, v_len, joint, j_len, dropout) in train_loader:  #
        print(f"motion shape: {motion.shape}")
        print(type(caption), type(motion), type(m_length), type(mean_gaussian), type(stddev_gaussian), type(v_len), type(dropout))
        print(video_adaptor.foward(mu=mean_gaussian, stddev=stddev_gaussian, length= v_len))
        i += 1
        if(i >= 1):
            break


