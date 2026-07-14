from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import hashlib
from collections import defaultdict
from .PR_VIPE_end import pr_vipe_infer

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            raw = name[1:] if name.startswith("M") else name
            motionfix_start_id = getattr(opt, "motionfix_start_id", None)
            if motionfix_start_id is not None and raw.isdigit() and int(raw) >= motionfix_start_id:
                continue
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                       'mid':    name,
                                                       'offset': int(f_tag * 20)}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                        'mid':    name,
                                        'offset': 0}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), idx
    
class Text_2D_MotionDatasetEval(Text2MotionDatasetEval):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, feat2D_dir, vipe_checkpt):
        super().__init__(opt, mean, std, split_file, w_vectorizer)
        from dataset.PR_VIPE_end import read_input, pr_vipe_infer

        seen, data_2D_raw, drop = set(), {}, {}  # inspect all exisitng mid without repeat
        for name in self.name_list:
            mid = self.data_dict[name]['mid']
            if mid in seen:
                continue
            seen.add(mid)
            try:
                raw = np.array(read_input(pjoin(feat2D_dir, f"{mid}.json")))
                # raw: [num_windows, 7, 15, 2]
                data_2D_raw[mid] = raw
                drop[mid] = 0
            except Exception:
                data_2D_raw[mid] = np.zeros((1, 7, 15, 2), dtype=float)
                drop[mid] = 1

        gaussian_data = pr_vipe_infer(vipe_checkpt, data_2D_raw)

        # reshape to frame feature
        self.data_joint   = {mid: arr.reshape(-1, 30)
                             for mid, arr in data_2D_raw.items()}
        self.data_gaussian = gaussian_data   # mid to gaussian
        self.drop          = drop

    def __getitem__(self, item):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, idx  = super().__getitem__(item)
        list_idx = self.pointer + item
        name     = self.name_list[list_idx]
        entry    = self.data_dict[name]
        mid    = entry["mid"]
        offset = entry['offset']   # frame offset within mid's full sequence

        video_startt = offset + idx

        # Raw joints [max_motion_length, 30]
        joints = self.data_joint[mid]
        video_joints = joints[video_startt : video_startt + m_length]
        joint_len    = len(video_joints)
        if joint_len < self.max_motion_length:
            video_joints = np.concatenate([video_joints,np.zeros((self.max_motion_length - joint_len, 30))],axis=0)

        # PR-VIPE Gaussian (window stride = 7 frames)
        max_vipe_len = self.max_motion_length // 7 + 2  #includ the windows of head and tails
        g_mean, g_std = self.data_gaussian[mid]
        gaussian_start = video_startt // 7
        gaussian_end = (video_startt + m_length) // 7 + 1
        g_mean = g_mean[gaussian_start: gaussian_end].astype(np.float32)
        g_std  = g_std [gaussian_start: gaussian_end].astype(np.float32)
        v_len  = len(g_mean)
        if v_len < max_vipe_len:
            pad    = max_vipe_len - v_len
            g_mean = np.concatenate([g_mean, np.zeros((pad, g_mean.shape[1]), dtype=np.float32)], axis=0)
            g_std  = np.concatenate([g_std,  np.zeros((pad, g_std.shape[1]),  dtype=np.float32)], axis=0)

        dropout = self.drop[mid]

        return (word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, g_mean, g_std, v_len,
                video_joints, joint_len, dropout)



class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]  #Hml3D preprocess is 20 fps
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                       'mid':    name,
                                                       'offset': int(f_tag * 20)}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                        'mid':    name,
                                        'offset': 0}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        print("Total number of motions {}, snippets {}".format(len(name_list), len(length_list)))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length, idx

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

class Text_2D_MotionDataset(Text2MotionDataset):
    def __init__(self, opt, mean, std, split_file, feat2D_dir, vipe_checkpt):
        super().__init__(opt, mean, std, split_file)
        from dataset.PR_VIPE_end import read_input, pr_vipe_infer

        seen, data_2D_raw, drop = set(), {}, {}  # inspect all exisitng mid without repeat
        for name in self.name_list:
            mid = self.data_dict[name]['mid']
            if mid in seen:
                continue
            seen.add(mid)
            try:
                raw = np.array(read_input(pjoin(feat2D_dir, f"{mid}.json")))
                # raw: [num_windows, 7, 15, 2]
                data_2D_raw[mid] = raw
                drop[mid] = 0
            except Exception:
                data_2D_raw[mid] = np.zeros((1, 7, 15, 2), dtype=float)
                drop[mid] = 1

        gaussian_data = pr_vipe_infer(vipe_checkpt, data_2D_raw)

        # reshape to frame feature
        self.data_joint   = {mid: arr.reshape(-1, 30)
                             for mid, arr in data_2D_raw.items()}
        self.data_gaussian = gaussian_data   # mid to gaussian
        self.drop          = drop

    def __getitem__(self, item):
        caption, motion, m_length, idx = super().__getitem__(item)
        list_idx = self.pointer + item
        name     = self.name_list[list_idx]
        entry    = self.data_dict[name]
        mid    = entry['mid']
        offset = entry['offset']   # frame offset within mid's full sequence

        video_startt = offset + idx

        # Raw joints [max_motion_length, 30]
        joints = self.data_joint[mid]
        video_joints = joints[video_startt : video_startt + m_length]
        joint_len  = len(video_joints)
        if joint_len < self.max_motion_length:
            video_joints = np.concatenate([video_joints,np.zeros((self.max_motion_length - joint_len, 30))],axis=0)

        # PR-VIPE Gaussian (window stride = 7 frames)
        max_vipe_len = self.max_motion_length // 7 + 2  #includ the windows of head and tails
        g_mean, g_std = self.data_gaussian[mid]
        gaussian_start = video_startt // 7
        gaussian_end = (video_startt + m_length) // 7 + 1
        g_mean = g_mean[gaussian_start: gaussian_end].astype(np.float32)
        g_std  = g_std [gaussian_start: gaussian_end].astype(np.float32)
        v_len  = len(g_mean)
        if v_len < max_vipe_len:
            pad    = max_vipe_len - v_len
            g_mean = np.concatenate([g_mean, np.zeros((pad, g_mean.shape[1]), dtype=np.float32)], axis=0)
            g_std  = np.concatenate([g_std,  np.zeros((pad, g_std.shape[1]),  dtype=np.float32)], axis=0)

        dropout = self.drop[mid]

        return (caption, motion, m_length, g_mean, g_std, v_len,
                video_joints, joint_len, dropout)
    
class HML3DMotionEditDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, motionfix_start_id=400000, w_vectorizer=None):
        self.opt, self.mean, self.std = opt, mean, std
        self.max_motion_length = opt.max_motion_length
        self.unit_length = opt.unit_length
        self.motionfix_start_id = motionfix_start_id
        self.motion_dir = getattr(opt, "motion_dir", pjoin(opt.root_dir, "HumanML3D", "new_joint_vecs"))
        self.text_dir = getattr(opt, "text_dir", pjoin(opt.root_dir, "HumanML3D", "texts"))
        self.data_dict, self.name_list = {}, []
        self.edit_meta = {}
        self.same_src_neg = {}

        id_list = [line.strip() for line in open(split_file) if line.strip()]
        for name in tqdm(id_list):
            raw = name[1:] if name.startswith("M") else name
            task_type = "editing" if raw.isdigit() and int(raw) >= self.motionfix_start_id else "generation"

            try:
                motion = np.load(pjoin(self.motion_dir, name + ".npy"), allow_pickle=True)
            except Exception as e:
                print(e)
                continue

            if task_type == "editing":
                motion = motion.item()
                src_motion, tgt_motion = motion["source"], motion["target"]
                if len(tgt_motion) < 40 or len(tgt_motion) > self.max_motion_length:
                    continue
                # source keeps own len
                if len(src_motion) < 40 or len(src_motion) > self.max_motion_length:
                    continue
                text_data = self._read_edit_text_data(name)
                if len(text_data) == 0:
                    continue
                text = text_data[0]["caption"]
                self.data_dict[name] = (text_data, src_motion, tgt_motion, len(tgt_motion), 1)
                self.edit_meta[name] = {
                    "src": self._array_hash(src_motion),
                    "tgt": self._array_hash(tgt_motion),
                    "text": text,
                }
                self.name_list.append(name)
                continue

            try:
                text_data, motion = [], np.asarray(motion)
                if len(motion) < 40 or len(motion) >= 200:
                    continue
                with open(pjoin(self.text_dir, name + ".txt"), encoding="utf-8") as f:
                    for line_id, line in enumerate(f.readlines()):
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag, to_tag = float(line_split[2]), float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        if f_tag == 0.0 and to_tag == 0.0:
                            text_data.append({"caption": caption, "tokens": tokens})
                        else:
                            s, e = int(f_tag * 20), int(to_tag * 20)
                            seg_motion = motion[s:e]
                            if len(seg_motion) < 40 or len(seg_motion) >= 200:
                                continue
                            new_name = f"{name}_{line_id}"
                            self.data_dict[new_name] = ([{"caption": caption, "tokens": tokens}], None, seg_motion, len(seg_motion), 0)
                            self.name_list.append(new_name)

                if len(text_data) > 0:
                    self.data_dict[name] = (text_data, None, motion, len(motion), 0)
                    self.name_list.append(name)
            except Exception as e:
                print(e)

        self._build_same_source_neg()
        print(f"Total samples: {len(self.name_list)}")
        if len(self.edit_meta) > 0:
            n_hard = len(self.same_src_neg)
            print(f"Same-source hard negatives: {n_hard}/{len(self.edit_meta)} edit samples")

    def _read_edit_text_data(self, name):
        text_data = []
        seen = set()
        with open(pjoin(self.text_dir, name + ".txt"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                caption = line.split("#")[0].strip()
                key = caption.lower()
                if not caption or key in seen:
                    continue
                seen.add(key)
                text_data.append({"caption": caption, "tokens": None})
        return text_data
    def _array_hash(self, arr, decimals=3):
        # stable source/target content key
        arr = np.ascontiguousarray(np.round(np.asarray(arr, dtype=np.float64), decimals))
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _build_same_source_neg(self):
        # same source, different target/text -> true hard negative pair
        by_source = defaultdict(list)
        for name, meta in self.edit_meta.items():
            by_source[meta["src"]].append(name)

        for names in by_source.values():
            if len(names) < 2:
                continue
            for name in names:
                meta = self.edit_meta[name]
                neg_texts = []
                for other in names:
                    if other == name:
                        continue
                    other_meta = self.edit_meta[other]
                    if other_meta["tgt"] == meta["tgt"]:
                        continue
                    if other_meta["text"].lower() == meta["text"].lower():
                        continue
                    neg_texts.append(other)
                if len(neg_texts) > 0:
                    self.same_src_neg[name] = neg_texts

    def __len__(self):
        return len(self.name_list)

    def get_task_flags(self):
        """Return one edit flag per name without sampling dataset items.

        data_dict stores (text, source, target, target_length, has_source).
        Keep this schema knowledge inside the dataset instead of duplicating a
        fragile tuple index in the training entrypoint.
        """
        return np.asarray(
            [int(bool(self.data_dict[name][4])) for name in self.name_list],
            dtype=np.int64,
        )

    def inv_transform(self, data):
        return data * self.std + self.mean

    def _crop_length(self, m_length):
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            return (m_length // self.unit_length - 1) * self.unit_length
        return (m_length // self.unit_length) * self.unit_length

    def _pad_motion(self, motion):
        if len(motion) < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - len(motion), motion.shape[1]))], axis=0)
        return motion

    def __getitem__(self, item):
        name = self.name_list[item]
        text_list, src_motion, tgt_motion, m_length, has_source = self.data_dict[name]
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        same_src_text = ""
        same_src_flag = 0
        same_tgt_motion, same_m_length = None, m_length
        if name in self.same_src_neg:
            neg_name = random.choice(self.same_src_neg[name])
            same_src_text = self.edit_meta[neg_name]["text"]
            same_src_flag = 1

        if has_source:
            # edit pair, keep full len
            m_length = (m_length // self.unit_length) * self.unit_length
            src_m_length = (len(src_motion) // self.unit_length) * self.unit_length
            tgt_motion = tgt_motion[:m_length]
            src_motion = src_motion[:src_m_length]
            if same_src_flag:
                neg_tgt_motion = self.data_dict[neg_name][2]
                same_m_length = (len(neg_tgt_motion) // self.unit_length) * self.unit_length
                same_tgt_motion = neg_tgt_motion[:same_m_length]
        else:
            # gen crop, null source later
            m_length = self._crop_length(m_length)
            idx = random.randint(0, len(tgt_motion) - m_length)
            tgt_motion = tgt_motion[idx:idx + m_length]
            src_motion = np.zeros_like(tgt_motion)
            src_m_length = m_length
            same_m_length = m_length

        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std
        if same_src_flag:
            same_tgt_motion = (same_tgt_motion - self.mean) / self.std
        else:
            same_tgt_motion = np.zeros_like(tgt_motion)

        src_motion, tgt_motion = self._pad_motion(src_motion), self._pad_motion(tgt_motion)
        same_tgt_motion = self._pad_motion(same_tgt_motion)

        return caption, src_motion, tgt_motion, m_length, has_source, src_m_length, same_src_text, same_src_flag, same_tgt_motion, same_m_length
