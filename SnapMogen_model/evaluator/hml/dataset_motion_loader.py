from dataset.HumanML3D_dataset import Text_2D_MotionDatasetEval
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from utils.get_opt import get_opt

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def get_dataset_motion_loader(opt_path, batch_size, fname, feat2D_dir, vipe_checkpt, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    # if opt.dataset_name == 'humanml3d' or opt.dataset_name == 'kit':
    print('Loading dataset %s ...' % opt.dataset_name)

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    split_file = pjoin(opt.data_root, '%s.txt'%fname)
    dataset = Text_2D_MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, feat2D_dir, vipe_checkpt)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                            collate_fn=collate_fn, shuffle=True)
    # else:
    #     raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset