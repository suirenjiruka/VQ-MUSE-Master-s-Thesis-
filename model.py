import torch
import torch.nn as nn
import math
import random
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from SnapMogen_model.encode_text import T5TextEncoder
from SnapMogen_model.transformer.tools import *
from Adaln_encoder import AdaLN_Encoder, AdaLN_ControlBranch

'''
Most of codes in our VAModel were borrowed from SnapMogen's Momask transformer, viewing details in https://github.com/snap-research/SnapMoGen/blob/main/model/transformer/transformer.py#L141
'''

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=500): # use [B, seq, dim] PE
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
      
    
class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim = 512, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # residual block
        self.residual_block = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4), 
            nn.LayerNorm(output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 4, output_dim)
        )

        # 3. Zero Init
        # zero init
        nn.init.zeros_(self.residual_block[-1].weight)
        nn.init.zeros_(self.residual_block[-1].bias)

    def forward(self, x):
        x = self.input_proj(x)
        return x + self.residual_block(x)
    
class Cross_attention(nn.Module):
    def __init__(self, kv_dim, query_dim = 512, output_dim=512, dropout_rate = 0.1, head_num = 8, ff_rate = 4):
        super().__init__()
        self.key_value = nn.Linear(kv_dim, output_dim * 2)
        self.query = nn.Linear(query_dim, output_dim)
        assert output_dim % head_num == 0

        self.kv_dim = kv_dim
        self.query_dim = query_dim
        self.head_size = output_dim // head_num
        self.head_num = head_num

        assert self.query_dim == self.head_num * self.head_size

        self.Qnorm = nn.LayerNorm(query_dim)  #To future Ina, You have to check whether KV needs to be normalized before enter into cross-attention or not
        self.KVnorm = nn.LayerNorm(kv_dim)
        self.norm = nn.LayerNorm(output_dim)

        self.ff = nn.Sequential(
            nn.Linear(output_dim, ff_rate * output_dim),   # ff hidden dim
            nn.GELU(),
            nn.Linear(ff_rate * output_dim, output_dim),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, q_input, kv_input, key_padding_mask=None):
        B, seq_len, _ = q_input.shape
        residual = q_input
        q_norm = self.Qnorm(q_input)
        kv_norm = self.KVnorm(kv_input)
        q = self.query(q_norm).view(B, -1, self.head_num, self.head_size).transpose(1, 2)#According to the heade, divide the tensor into shape [B, head_num, seq, head_size]
        kv = self.key_value(kv_norm) # calculate toogether would be faster
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, -1, self.head_num, self.head_size).transpose(1, 2) 
        v = v.view(B, -1, self.head_num, self.head_size).transpose(1, 2)
        
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.bool().view(B, 1, 1, -1) # mask shape is [B, header_num, query_len, kv_len]

        output = F.scaled_dot_product_attention(q,k,v, attn_mask).transpose(1, 2).contiguous()  #multi attention in pytorch (here) is 1 for keep, 0 for drop
        output = residual + output.view(B, -1, self.head_num * self.head_size)   #[B, seq_len, output_dim] & residual

        residual = output                     
        ff_output = self.ff(self.norm(output))   #normal & ff connection
        output = residual + ff_output            #residual again
        return output    #[B(32 or 64), seq_len (query length), output_dim]

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states) 
        output = output.transpose(1, 2)  # [B, L, code_idx] -> [B, code_idx, L]
        return output
    
class VAMotion(nn.Module):
    def __init__(self, cfg, device = None, full_length=80):  # max token len
        super().__init__()
        self.scales = cfg.vq.scales  # multi-scale setup
        self.cfg = cfg
        if device is None:
            device = cfg.exp.device
        # dimension setting
        self.text_dim = cfg.text_embedder.dim_embed
        self.motion_dim = cfg.vq.code_dim
        self.latent_dim = cfg.model.latent_dim
        #basic configuration
        self.device = device
        self.patch_sizes = [int(full_length // scale) for scale in self.scales]
        self.full_length = full_length
        self.motion_drop_prob = getattr(cfg.training, "m_drop", getattr(cfg.training, "target_full_mask_prob", 0.1)) # target fully masked CFG branch
        self.text_drop_prob = getattr(cfg.training, "t_drop", getattr(cfg.training, "text_cfg_drop_prob", 0.1))   # text dropout for CFG
        self.source_drop_prob = getattr(cfg.training, "v_drop", getattr(cfg.training, "source_cfg_drop_prob", 0.0)) # legacy, no source CFG drop now
        self.source_token_drop_prob = getattr(cfg.training, "s_drop", getattr(cfg.training, "source_token_drop_prob", 0.0))
        self.use_abs_pe = getattr(cfg.model, "use_abs_pe", False)
        init_std = math.sqrt(1 / self.latent_dim / 3) # init std
        self.noise_schedule = cosine_schedule # for cosine reduction, using on generation masking

        # code book setup
        self.lvl_embed = nn.Embedding(len(self.patch_sizes), self.latent_dim) # scale embedding
     
        _num_tokens = cfg.vq.nb_code + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = cfg.vq.nb_code
        self.pad_id = cfg.vq.nb_code + 1
        # Cross Attention layer
        #self.null_motion_embed = nn.Parameter(torch.randn(1, 1, self.latent_dim))  # null tokens for condition masking
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
        self.null_source_embed = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
        self.use_task_token = getattr(cfg.model, "use_task_token", True)
        self.task_embed = nn.Embedding(2, self.latent_dim) # 0: gen, 1: edit
        self.use_cond_delta = getattr(cfg.model, "use_cond_delta", True)
        self.text_delta_scale = nn.Parameter(torch.tensor(float(getattr(cfg.model, "text_delta_init", 1.0))))
        self.source_delta_scale = nn.Parameter(torch.tensor(float(getattr(cfg.model, "source_delta_init", 1.0))))
        self.use_vq_delta = getattr(cfg.model, "use_vq_delta", False)
        self.delta_alpha = float(getattr(cfg.model, "delta_alpha", 0.0))
        self.delta_beta = float(getattr(cfg.model, "delta_beta", 1.0))
        self.delta_temp = float(getattr(cfg.model, "delta_temp", 0.1))
        self.text_cross = Cross_attention(kv_dim=self.latent_dim, query_dim=self.latent_dim, output_dim=self.latent_dim, head_num=cfg.model.n_heads)
        self.source_cross = Cross_attention(kv_dim=self.latent_dim, query_dim=self.latent_dim, output_dim=self.latent_dim, head_num=cfg.model.n_heads)
        # Info Nce loss mlp
        self.motion_mlp = nn.Linear(self.latent_dim, 1)
        self.text_mlp = nn.Linear(self.latent_dim, 1)
        # global text for AdaLN
        self.text_global_mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self.latent_dim * 4, self.latent_dim),
        )
        # mask-ratio embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self.latent_dim * 4, self.latent_dim),
        )

        # Feature adaptor
        self.token_emb = nn.Embedding(_num_tokens, self.motion_dim)
        self.motion_adpator = nn.Linear(self.motion_dim, self.latent_dim)
        self.motion_position_enc = PositionalEncoding(self.latent_dim, dropout=cfg.model.dropout, max_len=sum(self.patch_sizes) + 8)
        self.text_adaptor = ResidualAdapter(input_dim=self.text_dim, output_dim = self.latent_dim, dropout=0.05)
        
        # Transformer setup
        """self.trans_input_norm = nn.LayerNorm(self.latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=cfg.model.n_heads,   # num of attenttion heads in transformer layer
                                                              dim_feedforward=cfg.model.ff_size, # feed foward layer size
                                                              dropout=cfg.model.dropout,
                                                              activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=cfg.model.n_layers)"""
        self.adaln_encoder = AdaLN_Encoder(model_dim=self.latent_dim, n_heads=cfg.model.n_heads, cond_dim=self.latent_dim,
                            ff_size=cfg.model.ff_size, n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
        # ControlNet branch S (preservation): trainable copy of the AdaLN stack, injects per-layer residuals for edit samples
        self.control_encoder = AdaLN_ControlBranch(model_dim=self.latent_dim, n_heads=cfg.model.n_heads, cond_dim=self.latent_dim,
                            ff_size=cfg.model.ff_size, n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
        # Text-guided VQ delta branch: predicts source->target codebook displacement and injects edit residuals
        self.delta_code_proj = nn.Linear(self.motion_dim, self.latent_dim)
        self.delta_control_proj = nn.Linear(self.latent_dim * 4, self.latent_dim)
        self.delta_encoder = AdaLN_ControlBranch(model_dim=self.latent_dim, n_heads=cfg.model.n_heads, cond_dim=self.latent_dim,
                            ff_size=cfg.model.ff_size, n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.motion_dim)
        )
        self.output_process = OutputProcess_Bert(out_feats=cfg.vq.nb_code, latent_dim=self.latent_dim)
        self.register_buffer("vq_codebook", torch.empty(0), persistent=False)

        #parameter initialization
        self.apply(self.__init_weights)

        #Adaln zero init 
        for block in self.adaln_encoder.layers:
            nn.init.zeros_(block.adaln_proj[-1].weight)
            nn.init.zeros_(block.adaln_proj[-1].bias)
        nn.init.zeros_(self.adaln_encoder.final_adaln_proj[-1].weight)
        nn.init.zeros_(self.adaln_encoder.final_adaln_proj[-1].bias)
        nn.init.zeros_(self.text_adaptor.residual_block[-1].weight)
        nn.init.zeros_(self.text_adaptor.residual_block[-1].bias)
        # control branches: interior blocks keep normal init (ControlNet's copy is a working feature extractor from step 0);
        # only the boundary zero connections are re-zeroed here (apply() overwrote them)
        self.control_encoder.init_zero_layers()
        self.delta_encoder.init_zero_layers()
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        d = torch.cat([torch.full((ps,), i) for i, ps in enumerate(self.patch_sizes)]) #[1 * 10..., 2 * 20..., 3 * 40..., 4 * 80...]
        self.register_buffer('lvl_1L', d.contiguous())   # scale ids
        
        self.text_emb = T5TextEncoder(         #  use_text_preprocessing,
            device, 
            local_files_only=False, 
            from_pretrained=cfg.text_embedder.version, 
            model_max_length=cfg.data.max_text_length
        )


    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            # conv init
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if  module.bias is not None:
                    module.bias.data.zero_()
                if  module.weight is not None:
                    module.weight.data.fill_(1.0)

    def sinusoidal_encoding(self, t):
        """
        Compute sinusoidal positional encoding for a batch of timesteps t.
        Args:
            t (Tensor): Shape (B, L), representing the timestep indices.
            d_model (int): Embedding dimension. (self.latent dim)

        Returns:
            Tensor of shape (B, L, D).
        """
        div_term = torch.exp(torch.arange(0, self.latent_dim, 2, dtype=torch.float32, device=t.device) * (-math.log(10000.0) / self.latent_dim))
        
        pe = torch.zeros(*t.shape, self.latent_dim, device=t.device)  # (B, L, D)
        pe[..., 0::2] = torch.sin(t.unsqueeze(-1) * div_term)  # Apply sin to even indices
        pe[..., 1::2] = torch.cos(t.unsqueeze(-1) * div_term)  # Apply cos to odd indices
        
        return pe
    
    def get_pe_from_mlens(self, mlens, max_len):
        B = len(mlens)
        t = torch.arange(max_len, device=mlens.device).unsqueeze(0).expand(B, max_len) # [0, 1, 2, 3,..., max_len - 1]
        T = mlens.unsqueeze(1).expand(B, max_len) # [max_len, max_len, max_len, ..., max_len]
        t_progress = (t / (T - 1 + 1e-4)) * self.full_length
        torch.clamp_min_(t_progress, 0.)
        return self.sinusoidal_encoding(t_progress)

    def timestep_embed(self, t):
        # t: mask ratio
        te = self.sinusoidal_encoding((t.float() * 1000.0).unsqueeze(1)).squeeze(1)  # (B, D)
        return self.time_mlp(te)

    def motion_process(self, motion_ids, relative_pe):
        motion_tokens = self.token_emb(motion_ids)
        motion_tokens = self.motion_adpator(motion_tokens)
        if self.use_abs_pe:
            motion_tokens = self.motion_position_enc(motion_tokens)
        if self.cfg.model.use_toa_pe:
            motion_tokens = motion_tokens + relative_pe  # add relative positional embedding 
        if self.cfg.model.use_lvl_pe:
            motion_tokens = motion_tokens + self.lvl_embed(self.lvl_1L).unsqueeze(0)   # add scale embedding
        return motion_tokens

    def prepare_motion_ids(self, motion_input, m_lens):
        # motion input ids
        motion_mask = []
        motion_ids = []
        time_to_arrival_pe = []

        # multi-scale motion ids
        for scale, ele in zip(self.scales, motion_input):
            # check scale length
            assert ele.shape[1] == int(self.full_length // scale), \
                f"scale {scale}: token length {ele.shape[1]} != full_length//scale {int(self.full_length // scale)}"
            ds_mlens = (m_lens // scale).long()
            ds_non_pad_mask = lengths_to_mask(ds_mlens, ele.shape[1])
            motion_mask.append(ds_non_pad_mask)
            motion_ids.append(ele)
            time_to_arrival_pe.append(self.get_pe_from_mlens(ds_mlens, ele.shape[1]))

        motion_ids = torch.cat(motion_ids, dim=1)
        motion_mask = torch.cat(motion_mask, dim=1)
        time_to_arrival_pe = torch.cat(time_to_arrival_pe, dim=1)
        # Motion masking
        motion_ids = torch.where(motion_mask, motion_ids, self.pad_id)
        return motion_ids, motion_mask, time_to_arrival_pe

    def masked_mean_pool(self, tokens, mask):
        mask = mask.float().unsqueeze(-1)
        return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def build_token_edit_prob(self, source_joints, target_joints, m_lens, source_m_lens, has_source, target_mask):
        # clean joint change, then downsample to token levels
        if source_joints is None or target_joints is None:
            return target_mask.float()
        B, F = target_joints.shape[:2]
        target_frame_lens = (m_lens * self.cfg.data.unit_length).long().clamp(min=1, max=F)
        source_frame_lens = (source_m_lens * self.cfg.data.unit_length).long().clamp(min=1, max=source_joints.shape[1])

        t = torch.arange(F, device=target_joints.device).float().unsqueeze(0)
        idx = torch.round(t * (source_frame_lens - 1).float().unsqueeze(1) / (target_frame_lens - 1).clamp(min=1).float().unsqueeze(1)).long()
        idx = torch.minimum(idx, (source_frame_lens - 1).unsqueeze(1))
        idx = idx.view(B, F, 1, 1).expand(-1, -1, source_joints.shape[2], source_joints.shape[3])
        aligned_source_joints = torch.gather(source_joints, 1, idx)

        pos_dist = torch.norm(target_joints - aligned_source_joints, dim=-1).mean(dim=-1)
        vel_dist = torch.zeros_like(pos_dist)
        vel_dist[:, 1:] = torch.norm((target_joints[:, 1:] - target_joints[:, :-1]) - (aligned_source_joints[:, 1:] - aligned_source_joints[:, :-1]), dim=-1).mean(dim=-1)
        change = pos_dist + 0.5 * vel_dist

        frame_mask = lengths_to_mask(target_frame_lens, F).bool()
        c_min = change.masked_fill(~frame_mask, 1e6).amin(dim=1, keepdim=True)
        c_max = change.masked_fill(~frame_mask, 0.0).amax(dim=1, keepdim=True)
        score = (change - c_min) / (c_max - c_min).clamp(min=1e-6)
        score = score.masked_fill(~frame_mask, 0.0)

        token_scores = []
        for scale, ps in zip(self.scales, self.patch_sizes):
            token_len = (m_lens // scale).long().clamp(min=1)
            t = torch.arange(ps, device=target_joints.device).float().unsqueeze(0)
            idx = torch.round(t * (target_frame_lens - 1).float().unsqueeze(1) / (token_len - 1).clamp(min=1).float().unsqueeze(1)).long()
            idx = torch.minimum(idx, (target_frame_lens - 1).unsqueeze(1))
            token_scores.append(torch.gather(score, 1, idx))

        edit_prob = torch.cat(token_scores, dim=1) * target_mask.float()
        edit_prob = torch.where(has_source.view(-1, 1).bool(), edit_prob, torch.zeros_like(edit_prob))
        return edit_prob

    def fuse_condition_delta(self, motion_tokens, text_cross, source_cross):
        # keep the motion stream once, then add clean condition deltas
        if not self.use_cond_delta:
            return motion_tokens + source_cross
        text_delta = text_cross - motion_tokens
        source_delta = source_cross - text_cross
        return motion_tokens + self.text_delta_scale * text_delta + self.source_delta_scale * source_delta

    def add_task_token(self, text_tokens, text_mask, has_source):
        if not self.use_task_token:
            return text_tokens, text_mask
        task_id = has_source.view(-1).long().clamp(min=0, max=1)
        task_token = self.task_embed(task_id).unsqueeze(1)
        task_mask = torch.ones(text_mask.shape[0], 1, device=text_mask.device, dtype=text_mask.dtype)
        text_tokens = torch.cat([task_token, text_tokens], dim=1)
        text_mask = torch.cat([task_mask, text_mask], dim=1)
        return text_tokens, text_mask

    def make_null_text(self, text_tokens):
        # keep task token if it exists
        null_text = self.null_text_embed.expand_as(text_tokens).clone()
        if self.use_task_token:
            null_text[:, :1] = text_tokens[:, :1]
        return null_text

    def source_token_dropout(self, source_ids, source_mask, has_source):
        if (not self.training) or self.source_token_drop_prob <= 0:
            return source_ids
        drop_mask = torch.rand_like(source_mask, dtype=torch.float, device=source_ids.device) < self.source_token_drop_prob
        drop_mask = drop_mask & source_mask.bool() & has_source.view(-1, 1).bool()
        return torch.where(drop_mask, self.mask_id, source_ids)

    def set_vq_codebook(self, codebook):
        # frozen VQ latent table for delta supervision/logits
        self.vq_codebook = codebook.detach().clone().float()

    def vq_latents(self, ids, valid_mask=None):
        if self.vq_codebook.numel() == 0:
            raise RuntimeError("VQ codebook is not set. Call set_vq_codebook() before using VQ delta.")
        safe_ids = ids.clamp(min=0, max=self.cfg.vq.nb_code - 1)
        z = F.embedding(safe_ids, self.vq_codebook.to(device=ids.device, dtype=torch.float32)).to(dtype=self.delta_head[-1].weight.dtype)
        if valid_mask is None:
            valid_mask = (ids >= 0) & (ids < self.cfg.vq.nb_code)
        else:
            valid_mask = valid_mask.bool() & (ids >= 0) & (ids < self.cfg.vq.nb_code)
        return z * valid_mask.unsqueeze(-1).float()

    def codebook_similarity_logits(self, z):
        # cosine codebook classifier, shape: [B, code, L]
        codebook = self.vq_codebook.to(device=z.device, dtype=z.dtype)
        z = F.normalize(z, dim=-1)
        codebook = F.normalize(codebook, dim=-1)
        logits = torch.matmul(z, codebook.t()) / max(self.delta_temp, 1e-6)
        return logits.transpose(1, 2)

    def run_delta_branch(self, transformer_input, text_cross, source_cross, aligned_src_ids,
                         target_mask, padding_mask, source_present):
        if (not self.use_vq_delta) or self.vq_codebook.numel() == 0:
            return None, None, None, None

        source_gate = source_present.float()
        z_src = self.vq_latents(aligned_src_ids, target_mask & source_present.squeeze(-1))
        z_src = z_src * source_gate
        z_src_proj = self.delta_code_proj(z_src)

        # text-guided source->target delta branch
        delta_control = self.delta_control_proj(torch.cat([transformer_input, text_cross, source_cross, z_src_proj], dim=-1))
        delta_residuals, delta_hiddens = self.delta_encoder(
            transformer_input, delta_control, cond=text_cross, padding_mask=padding_mask, return_hiddens=True
        )
        delta_pred = self.delta_head(delta_hiddens[-1]) * source_gate
        # residual codebook logits: no source-copy bias when delta_pred starts at zero
        latent_logits = self.codebook_similarity_logits(z_src + delta_pred) - self.codebook_similarity_logits(z_src).detach()
        return delta_residuals, latent_logits, delta_pred, z_src

    def apply_latent_logits(self, token_logits, latent_logits, source_present):
        if latent_logits is None or self.delta_alpha == 0.0:
            return token_logits
        return token_logits + self.delta_alpha * source_present.float() * latent_logits

    def vq_delta_losses(self, delta_pred, z_src, latent_logits, target_ids, target_mask, delta_active, predict_mask):
        zero = target_ids.new_tensor(0.0, dtype=torch.float)
        if delta_pred is None or latent_logits is None:
            return zero, zero

        loss_mask = predict_mask.bool() & target_mask.bool() & delta_active.squeeze(-1).bool()
        if not loss_mask.any():
            return zero, zero

        z_tgt = self.vq_latents(target_ids, target_mask)
        delta_gt = z_tgt - z_src
        delta_loss = F.smooth_l1_loss(delta_pred[loss_mask], delta_gt[loss_mask])

        latent_labels = torch.where(loss_mask, target_ids, self.mask_id)
        latent_ce, _, _ = cal_performance(latent_logits, latent_labels, ignore_index=self.mask_id)
        return delta_loss, latent_ce

    def InfoNCE_text(self, motion, text_input, m_mask, t_mask, temperature=0.15, detach_motion=False):
        if detach_motion:
            motion = motion.detach()
            with torch.no_grad():
                m_weight = self.motion_mlp(motion) # text-only contrastive target
        else:
            m_weight = self.motion_mlp(motion) # [B, L, D] -> [B, L, 1]
        t_weight = self.text_mlp(text_input)

        # valid / weighted token mask
        m_pool_weight = m_mask.float().unsqueeze(-1)
        m_weight = m_weight.masked_fill(m_pool_weight <= 0, -1e4)
        t_weight = t_weight.masked_fill(t_mask.unsqueeze(-1) == 0, -1e4)
        m_weight = torch.softmax(m_weight, dim=1)   # token weight
        t_weight = torch.softmax(t_weight, dim=1)
        m_weight = m_weight * m_pool_weight
        m_weight = m_weight / m_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # attention pooling
        motion_squeezed = F.normalize((m_weight * motion).sum(dim=1), dim=-1)
        text_squeezed = F.normalize((t_weight * text_input).sum(dim=1), dim=-1)

        # cosine logits
        cos_mt = (motion_squeezed @ text_squeezed.T) / temperature

        labels = torch.arange(len(motion), device=motion.device)
        info_loss_mt = (F.cross_entropy(cos_mt, labels, reduction='none') + F.cross_entropy(cos_mt.T, labels, reduction='none')) / 2
        return info_loss_mt
    
    def CFG_mask(self, cond, prob):
        bs = cond.shape[0]
        if isinstance(prob, torch.Tensor):
            # per-sample binary mask (inference): [B] or [B, 1]
            mask = prob.view(bs, *([1] * (cond.ndim - 1))).bool()
        elif prob > 0.0:
            # scalar dropout probability (training)
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * prob)
            mask = mask.view(bs, *([1] * (cond.ndim - 1))).bool()
        else:
            return torch.zeros(bs, *([1] * (cond.ndim - 1)), device=cond.device, dtype=torch.bool)
        return mask 

    def align_source_tokens(self, source_tensor, target_mask, source_mask):
        # align source timeline to target token timeline, per VQ scale
        aligned = []
        start = 0
        for ps in self.patch_sizes:
            tgt_len = target_mask[:, start:start + ps].sum(dim=1).clamp(min=1)
            src_len = source_mask[:, start:start + ps].sum(dim=1).clamp(min=1)
            t = torch.arange(ps, device=source_tensor.device).float().unsqueeze(0)
            idx = torch.round(t * (src_len - 1).float().unsqueeze(1) / (tgt_len - 1).clamp(min=1).float().unsqueeze(1)).long()
            idx = torch.minimum(idx, (src_len - 1).unsqueeze(1))
            if source_tensor.ndim == 3:
                idx = idx.unsqueeze(-1).expand(-1, -1, source_tensor.shape[-1])
            aligned.append(torch.gather(source_tensor[:, start:start + ps], 1, idx))
            start += ps
        return torch.cat(aligned, dim=1)

    def run_condition_branch(self, motion_ids, source_ids, text_tokens, time_to_arrival_pe, source_pe,
                             source_mask, text_mask, motion_padding_mask, has_source, delta_active=None):
        # one condition branch: used by train, CFG, and diagnostics
        target_mask = ~motion_padding_mask
        input_motion_embs = self.motion_process(motion_ids, time_to_arrival_pe)
        source_tokens = self.motion_process(source_ids, source_pe)
        source_present = has_source.view(-1, 1, 1).bool()
        if delta_active is None:
            delta_active = source_present
        else:
            delta_active = delta_active.view(-1, 1, 1).bool()
        source_tokens = torch.where(source_present, source_tokens, self.null_source_embed.expand_as(source_tokens))

        aligned_src = self.align_source_tokens(source_tokens, target_mask, source_mask)
        aligned_src_ids = self.align_source_tokens(source_ids, target_mask, source_mask)
        aligned_src = torch.where(source_present, aligned_src, torch.zeros_like(aligned_src))

        text_cross = self.text_cross(input_motion_embs, text_tokens, text_mask)
        source_cross = self.source_cross(text_cross, source_tokens, source_mask)
        source_cross = torch.where(source_present, source_cross, text_cross)
        transformer_input = self.fuse_condition_delta(input_motion_embs, text_cross, source_cross)

        control_residuals = self.control_encoder(transformer_input, aligned_src, cond=text_cross, padding_mask=motion_padding_mask)
        delta_residuals, latent_logits, _, _ = self.run_delta_branch(
            transformer_input, text_cross, source_cross, aligned_src_ids,
            target_mask, motion_padding_mask, delta_active
        )
        output = self.adaln_encoder(
            x=transformer_input,
            cond=text_cross,
            padding_mask=motion_padding_mask,
            control_residuals=control_residuals,
            control_gate=source_present.float(),
            delta_residuals=delta_residuals,
            delta_gate=delta_active.float() * self.delta_beta
        )
        logits = self.output_process(output)
        logits = self.apply_latent_logits(logits, latent_logits, delta_active)
        return logits, output, aligned_src

    def forward(self, target_input, source_input, text_input, m_lens, has_source, source_m_lens=None,
                source_joints=None, target_joints=None):
        # target/source input are both multi-scale VQ ids
        target_ids, target_mask, time_to_arrival_pe = self.prepare_motion_ids(target_input, m_lens)
        if source_m_lens is None:
            source_m_lens = m_lens   # fallback len
        source_ids, source_mask, source_pe = self.prepare_motion_ids(source_input, source_m_lens)  # source own len

        # text embedding
        with torch.no_grad():
            text_tokens, text_mask = self.text_emb.get_text_embeddings(text_input)
        text_tokens = self.text_adaptor(text_tokens)
        text_tokens = torch.where(text_mask.unsqueeze(-1).bool(), text_tokens, 0.0)

        # task context
        task_is_edit = has_source.to(target_ids.device).view(-1, 1, 1).bool()
        text_tokens, text_mask = self.add_task_token(text_tokens, text_mask, has_source.to(target_ids.device))
        text_tokens_for_loss = text_tokens

        # CFG drop for text only
        text_cfg_mask = self.CFG_mask(text_tokens, self.text_drop_prob)
        delta_active = task_is_edit & (~text_cfg_mask)
        text_tokens = torch.where(text_cfg_mask, self.make_null_text(text_tokens), text_tokens)

        # source motion condition
        source_ids_clean = source_ids
        source_ids = self.source_token_dropout(source_ids, source_mask, task_is_edit.squeeze(-1).squeeze(-1))
        source_tokens_clean = self.motion_process(source_ids, source_pe)
        source_present = task_is_edit
        source_tokens = torch.where(source_present, source_tokens_clean, self.null_source_embed.expand_as(source_tokens_clean))
        aligned_src = self.align_source_tokens(source_tokens, target_mask, source_mask)
        aligned_src_ids = self.align_source_tokens(source_ids_clean, target_mask, source_mask)
        aligned_src = torch.where(source_present, aligned_src, torch.zeros_like(aligned_src))

        # ---- target masking: MoMask/BERT-style, per-sample uniform ratio ----
        bs = target_ids.shape[0]
        rand_time = uniform((bs,), device=target_ids.device)
        rand_mask_probs = self.noise_schedule(rand_time) # sample mask ratio
        # token num to predict
        n_valid = target_mask.sum(dim=1)
        num_pred = (n_valid.float() * rand_mask_probs).round().clamp(min=1).long()
        rand_score = torch.rand_like(target_mask, dtype=torch.float).masked_fill(~target_mask.bool(), 1.0)
        ranks = rand_score.argsort(dim=1).argsort(dim=1)
        predict_mask = ranks < num_pred.unsqueeze(-1)                          # loss mask
        # full-mask CFG branch
        motion_cfg_mask = self.CFG_mask(target_ids, self.motion_drop_prob)
        full_drop = motion_cfg_mask & target_mask.bool()                       # full mask sample
        predict_mask = predict_mask | full_drop
        # full mask uses timestep 1
        rand_mask_probs = torch.where(motion_cfg_mask.squeeze(-1), torch.ones_like(rand_mask_probs), rand_mask_probs)
        # CE only on masked token
        labels = torch.where(predict_mask, target_ids, self.mask_id)

        # BERT-style corruption
        masked_motion_ids = target_ids.clone()
        rand_id_mask = (torch.rand_like(target_mask, dtype=torch.float) < 0.10) & predict_mask
        masked_motion_ids = torch.where(rand_id_mask, torch.randint_like(target_ids, high=self.mask_id), masked_motion_ids)
        to_mask = (torch.rand_like(target_mask, dtype=torch.float) < 0.88) & predict_mask & ~rand_id_mask
        masked_motion_ids = torch.where(to_mask, self.mask_id, masked_motion_ids)
        # source inpainting corruption: a fraction of masked slots show the aligned source token instead of [MASK];
        # labels stay = target ids, so the model learns the per-position copy-vs-edit decision (matches source-init inference)
        src_fill_prob = getattr(self.cfg.training, "src_fill_prob", 0.0)
        if self.training and src_fill_prob > 0.:
            fill_mask = (torch.rand_like(target_mask, dtype=torch.float) < src_fill_prob) & to_mask & task_is_edit.squeeze(-1)
            masked_motion_ids = torch.where(fill_mask, aligned_src_ids, masked_motion_ids)
        # perturb visible token
        if self.training and getattr(self.cfg.training, "pert_prob", 0.0) > 0.:
            pert_mask = (torch.rand_like(target_mask, dtype=torch.float) < self.cfg.training.pert_prob) & (~predict_mask) & target_mask.bool()
            masked_motion_ids = torch.where(pert_mask, torch.randint_like(target_ids, high=self.mask_id), masked_motion_ids)
        # force full mask sample
        masked_motion_ids = torch.where(full_drop, self.mask_id, masked_motion_ids)

        # CA stack
        masked_motion_tokens = self.motion_process(masked_motion_ids, time_to_arrival_pe)
        text_cross = self.text_cross(masked_motion_tokens, text_tokens, text_mask)
        source_cross = self.source_cross(text_cross, source_tokens, source_mask)
        source_cross = torch.where(source_present, source_cross, text_cross)
        transformer_input = self.fuse_condition_delta(masked_motion_tokens, text_cross, source_cross)

        # source branch controls edit samples
        control_residuals = self.control_encoder(transformer_input, aligned_src, cond=text_cross, padding_mask=~target_mask)
        delta_residuals, latent_logits, delta_pred, z_src = self.run_delta_branch(
            transformer_input, text_cross, source_cross, aligned_src_ids,
            target_mask, ~target_mask, delta_active
        )
        output = self.adaln_encoder(
            x=transformer_input,
            cond=text_cross,
            padding_mask=~target_mask,
            control_residuals=control_residuals,
            control_gate=source_present.float(),
            delta_residuals=delta_residuals,
            delta_gate=delta_active.float() * self.delta_beta
        )
        token_logits = self.output_process(output)
        logits = self.apply_latent_logits(token_logits, latent_logits, delta_active)

        # masked token CE
        ce_loss, _, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
        delta_loss, latent_ce = self.vq_delta_losses(delta_pred, z_src, latent_logits, target_ids, target_mask,
                                                     delta_active, predict_mask)

        # InfoNCE text: clean target motion only
        target_tokens_clean = self.motion_process(target_ids, time_to_arrival_pe)
        loss_mt = self.InfoNCE_text(target_tokens_clean, text_tokens_for_loss, target_mask.float(), text_mask,
                                    temperature=0.15, detach_motion=True)

        return ce_loss, loss_mt.mean(), delta_loss, latent_ce, acc
    
    def forward_with_cond_scale(self, motion_ids, source_ids, text_embs, time_to_arrival_pe, source_pe,
                                source_mask, text_mask, motion_padding_mask, has_source, mask_ratio,
                                cond_scale=3, source_scale=None, text_delta_scale=1.0):    # text-only CFG
        # Two branches:
        #   B(base): given source + null text
        #   C(full): given source + text
        # scaled = B + cond_scale * (C - B)
        input_motion_ids = torch.cat([motion_ids, motion_ids], dim=0)
        input_source_ids = torch.cat([source_ids, source_ids], dim=0)
        input_source_mask = torch.cat([source_mask, source_mask], dim=0)
        input_text_mask = torch.cat([text_mask, text_mask], dim=0)
        input_motion_padding_mask = torch.cat([motion_padding_mask, motion_padding_mask], dim=0)
        input_toa_pe = torch.cat([time_to_arrival_pe, time_to_arrival_pe], dim=0)
        input_source_pe = torch.cat([source_pe, source_pe], dim=0)

        # text: base branch uses null text, full branch uses real text
        null_text = self.make_null_text(text_embs)
        input_text_embs = torch.cat([null_text, text_embs], dim=0)
        task_is_edit = has_source.view(-1, 1, 1).bool()
        input_has_source = torch.cat([task_is_edit, task_is_edit], dim=0)
        input_delta_active = torch.cat([torch.zeros_like(task_is_edit), task_is_edit], dim=0)

        # same branch forward as training
        output_logits, _, _ = self.run_condition_branch(
            input_motion_ids, input_source_ids, input_text_embs, input_toa_pe, input_source_pe,
            input_source_mask, input_text_mask, input_motion_padding_mask, input_has_source,
            delta_active=input_delta_active
        )
        base_logits, full_logits = output_logits.chunk(2, dim=0)

        if torch.is_tensor(cond_scale):
            cond_scale = cond_scale.to(full_logits.device, dtype=full_logits.dtype).view(-1, 1, 1)

        # source_scale is kept as a legacy arg; CFG now guides text only
        scaled_logits = base_logits + cond_scale * (full_logits - base_logits)
        return scaled_logits #[B, dim, L]  

    @torch.no_grad()
    @eval_decorator
    def generate(self, source_input, text_input, m_lens, has_source, t_drop,
                 timesteps: int,
                 cond_scale: int,
                 source_cond_scale=None,   # legacy arg, no-op in text-only CFG
                 source_m_lens=None,       # source len
                 text_delta_scale=1.0,     # legacy eval knob, no-op here
                 source_hint_ratio=0.0,    # first-step source hints inside the masked input, then fully replaced by generated ids
                 temperature=1,
                 topk_filter_thres=0.95,
                 gsample=False):   #borrow from SnapMogen
        device = self.device
        B = len(m_lens)

        # task mode
        if not torch.is_tensor(has_source):
            has_source = torch.tensor(has_source, device=device).expand(B)
        has_source = has_source.to(device).view(-1, 1).bool()

        # encode text once
        text_tokens, text_mask = self.text_emb.get_text_embeddings(text_input)
        text_tokens = self.text_adaptor(text_tokens)
        text_mask = text_mask.bool()
        text_tokens, text_mask = self.add_task_token(text_tokens, text_mask, has_source)
        text_cfg_mask = self.CFG_mask(text_tokens, t_drop)
        text_tokens = torch.where(text_cfg_mask, self.make_null_text(text_tokens), text_tokens)

        # optional source ids
        if source_input is None:
            source_input = [torch.zeros(B, int(self.full_length//scale), dtype=torch.long, device=device) for scale in self.scales]
        if source_m_lens is None:
            source_m_lens = m_lens   # fallback len
        source_ids, source_mask, source_pe = self.prepare_motion_ids(source_input, source_m_lens)  # source own len

        # valid masks by final len
        non_padding_mask = []
        lengths_div = []
        new_mlens = torch.zeros_like(m_lens)
        time_to_arrival_pe = []
        for scale in self.scales:
            non_padding_mask.append(
                lengths_to_mask((m_lens//scale).long(), int(self.full_length//scale))
            )
            lengths_div.append(int(self.full_length//scale))
            new_mlens += m_lens // scale
            time_to_arrival_pe.append(self.get_pe_from_mlens((m_lens//scale).long(), int(self.full_length//scale)))

        non_padding_mask = torch.cat(non_padding_mask, dim=1)
        padding_mask = ~non_padding_mask
        time_to_arrival_pe = torch.cat(time_to_arrival_pe, dim=1)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.0)
        starting_temperature = temperature

        aligned_src_ids = None
        if source_hint_ratio > 0:
            aligned_src_ids = self.align_source_tokens(source_ids, non_padding_mask, source_mask)

        for step_id, timestep in enumerate(torch.linspace(0, 1, timesteps, device=device)):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * new_mlens).clamp(min=1)  # (b, )

            # remask low-confidence tokens for the next refinement step
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)
            if step_id == 0 and aligned_src_ids is not None:
                source_hint_mask = (torch.rand_like(scores) < source_hint_ratio) & is_mask & non_padding_mask & has_source
                ids = torch.where(source_hint_mask, aligned_src_ids, ids)

            '''
            Preparing input
            '''
            # predict masked target ids with source motion and edit text
            logits = self.forward_with_cond_scale(ids, 
                                                  source_ids,
                                                  text_tokens, 
                                                  time_to_arrival_pe=time_to_arrival_pe,
                                                  source_pe=source_pe,
                                                  source_mask=source_mask, # 1 for keep, 0 for drop
                                                  text_mask=text_mask,      # 1 for keep, 0 for drop
                                                  motion_padding_mask=padding_mask,     # 1 for drop, 0 for keep
                                                  has_source=has_source,
                                                  mask_ratio=rand_mask_prob,
                                                  cond_scale=cond_scale,
                                                  source_scale=source_cond_scale,
                                                  text_delta_scale=text_delta_scale)
            

            logits = logits.permute(0, 2, 1)  # (b, ntoken, seqlen) -> (b, seqlen, ntoken)
            # print(logits.shape, self.cfg.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)


            ids = torch.where(is_mask, pred_ids, ids)
            ids = torch.where(padding_mask, self.pad_id, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens.
            scores = scores.masked_fill(~is_mask, 1e5)

        # split concatenated multi-scale ids back to VQ decoder format
        ids = torch.where(non_padding_mask & ((ids < 0) | (ids >= self.cfg.vq.nb_code)), torch.zeros_like(ids), ids)
        ids = torch.where(padding_mask, -1, ids)
        return_list = []
        start = 0
        for length in lengths_div:
            return_list.append(ids[..., start:start+length])
            start += length
        # print("Final", ids.max(), ids.min())
        return return_list



