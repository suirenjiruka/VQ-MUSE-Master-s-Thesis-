import os
import sys
import torch
import torch.nn.functional as F
import numpy as np


class TMREvaluatorWrapper(object):
    def __init__(self, tmr_root, run_dir, device, mean, std, ckpt_name="last", input_is_normalized=True):
        self.device = device
        self.mean = torch.as_tensor(mean, device=device).float()
        self.std = torch.as_tensor(std, device=device).float()
        self.input_is_normalized = input_is_normalized

        if not tmr_root or not os.path.isdir(tmr_root):
            raise FileNotFoundError(f"TMR repo not found: {tmr_root}")
        if not run_dir or not os.path.isdir(run_dir):
            raise FileNotFoundError(f"TMR run/model dir not found: {run_dir}")

        if tmr_root not in sys.path:
            sys.path.insert(0, tmr_root)

        try:
            import src.prepare  # noqa: F401
            from src.config import read_config
            from src.load import load_model_from_cfg
            from src.data.collate import collate_x_dict
            from hydra.utils import instantiate
        except Exception as e:
            raise ImportError(
                f"Cannot import official TMR code: {e}. Please install TMR requirements and set tmr.root to the TMR repo."
            ) from e

        self.collate_x_dict = collate_x_dict
        self.cfg = read_config(run_dir)
        norm_cfg = self.cfg.data.motion_loader.normalizer
        if not os.path.isabs(norm_cfg.base_dir):
            norm_cfg.base_dir = os.path.join(tmr_root, norm_cfg.base_dir)
        self.model = load_model_from_cfg(self.cfg, ckpt_name, eval_mode=True, device=device)
        self.normalizer = instantiate(self.cfg.data.motion_loader.normalizer)
        self.normalizer.mean = self.normalizer.mean.to(device).float()
        self.normalizer.std = self.normalizer.std.to(device).float()
        self.model.to(device).eval()
        print(f"Loading TMR motion encoder completed: {run_dir} ({ckpt_name})")

    def _prepare_motion(self, motion, length):
        motion = motion[:int(length)].to(self.device).float()
        if self.input_is_normalized:
            motion = motion * self.std + self.mean
        return self.normalizer(motion)

    # same ordering behavior as EvaluatorModelWrapper: output follows length-sorted order
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device).long()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            batch = []
            for motion, length in zip(motions, m_lens):
                batch.append({"x": self._prepare_motion(motion, length), "length": int(length.item())})

            motion_x_dict = self.collate_x_dict(batch)
            motion_x_dict = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in motion_x_dict.items()
            }
            latent = self.model.encode(motion_x_dict, modality="motion", sample_mean=True)
            latent = F.normalize(latent, dim=-1)
        return latent
