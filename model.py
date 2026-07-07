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
        self.source_drop_prob = getattr(cfg.training, "v_drop", getattr(cfg.training, "source_cfg_drop_prob", 0.0)) # source dropout for CFG
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
        # source control branch: source canvas residuals for edit samples
        self.control_encoder = AdaLN_ControlBranch(model_dim=self.latent_dim, n_heads=cfg.model.n_heads, cond_dim=self.latent_dim,
                            ff_size=cfg.model.ff_size, n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)

        # VQ delta control branch: source code latent + text/source context -> target code delta
        self.use_vq_delta = getattr(cfg.model, "use_vq_delta", False)
        self.delta_alpha = float(getattr(cfg.model, "delta_alpha", 0.0))
        self.delta_beta = float(getattr(cfg.model, "delta_beta", 0.3))
        self.delta_temp = float(getattr(cfg.model, "delta_temp", 0.15))
        self.use_latent_classifier = getattr(cfg.model, "use_latent_classifier", False)
        self.latent_w_min = float(getattr(cfg.model, "latent_w_min", 0.5))
        self.latent_w_max = float(getattr(cfg.model, "latent_w_max", 0.9))
        self.raw_w_sum = float(getattr(cfg.model, "raw_w_sum", 1.4))
        self.delta_code_proj = nn.Linear(self.motion_dim, self.latent_dim)
        self.delta_control_proj = nn.Linear(self.latent_dim * 4, self.latent_dim)
        self.delta_encoder = AdaLN_ControlBranch(model_dim=self.latent_dim, n_heads=cfg.model.n_heads, cond_dim=self.latent_dim,
                            ff_size=cfg.model.ff_size, n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.motion_dim),
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
        # source branch keeps normal inner init; zero only the output boundary
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

    def source_token_dropout(self, source_ids, source_mask, has_source):
        if (not self.training) or self.source_token_drop_prob <= 0:
            return source_ids
        drop_mask = torch.rand_like(source_mask, dtype=torch.float, device=source_ids.device) < self.source_token_drop_prob
        drop_mask = drop_mask & source_mask.bool() & has_source.view(-1, 1).bool()
        return torch.where(drop_mask, self.mask_id, source_ids)

    def set_vq_codebook(self, codebook):
        # frozen VQ table for latent-token logits
        self.vq_codebook = codebook.detach().clone().float()

    def get_vq_codebook(self, device, dtype):
        if self.vq_codebook.numel() > 0:
            return self.vq_codebook.to(device=device, dtype=dtype)
        return self.token_emb.weight[:self.cfg.vq.nb_code].detach().to(device=device, dtype=dtype)

    def vq_latents(self, ids, valid_mask=None):
        safe_ids = ids.clamp(min=0, max=self.cfg.vq.nb_code - 1)
        codebook = self.get_vq_codebook(ids.device, self.token_emb.weight.dtype)
        latents = F.embedding(safe_ids, codebook)
        valid = (ids >= 0) & (ids < self.cfg.vq.nb_code)
        if valid_mask is not None:
            valid = valid & valid_mask.bool()
        return latents.masked_fill(~valid.unsqueeze(-1), 0.0)

    def codebook_similarity_logits(self, z):
        codebook = self.get_vq_codebook(z.device, z.dtype)
        z_norm = F.normalize(z, dim=-1)
        code_norm = F.normalize(codebook, dim=-1)
        logits = torch.einsum("bld,kd->blk", z_norm, code_norm) / self.delta_temp
        return logits.transpose(1, 2)   # [B, code, L]

    def run_delta_branch(self, transformer_input, text_cross, source_cross, aligned_src_ids,
                         target_mask, padding_mask, delta_active):
        # extra edit path, silent for gen/null-text samples
        if (not self.use_vq_delta) or (self.delta_alpha == 0.0 and self.delta_beta == 0.0 and not self.use_latent_classifier):
            return None, None, None, None

        active = delta_active.view(-1, 1, 1).bool()
        if active.sum() == 0:
            return None, None, None, None

        valid = target_mask.bool() & active.squeeze(-1)
        z_src = self.vq_latents(aligned_src_ids, valid)
        z_src_proj = self.delta_code_proj(z_src)
        delta_control = torch.cat([transformer_input, text_cross, source_cross, z_src_proj], dim=-1)
        delta_control = self.delta_control_proj(delta_control)

        delta_residuals, delta_hiddens = self.delta_encoder(
            transformer_input, delta_control, cond=text_cross, padding_mask=padding_mask, return_hiddens=True
        )
        delta_pred = self.delta_head(delta_hiddens[-1]) * active.float()
        pred_z = z_src + delta_pred

        latent_logits = self.codebook_similarity_logits(pred_z)
        src_logits = self.codebook_similarity_logits(z_src.detach()).detach()
        latent_res_logits = (latent_logits - src_logits) * active.float().transpose(1, 2)
        return delta_residuals, latent_res_logits, latent_logits, delta_pred

    def apply_vq_delta_logits(self, token_logits, latent_res_logits):
        if latent_res_logits is None or self.delta_alpha == 0.0:
            return token_logits
        return token_logits + self.delta_alpha * latent_res_logits

    def latent_classifier_logits(self, raw_logits, latent_logits, active_edit, mask_ratio):
        # edit uses latent classifier as main path; gen keeps raw logits
        if (not self.use_latent_classifier) or latent_logits is None:
            return raw_logits

        t = mask_ratio.to(raw_logits.device, dtype=raw_logits.dtype).view(-1)
        if t.numel() == 1:
            t = t.expand(raw_logits.shape[0])
        elif t.numel() != raw_logits.shape[0]:
            repeat = raw_logits.shape[0] // t.numel()
            if repeat * t.numel() != raw_logits.shape[0]:
                t = t.mean().expand(raw_logits.shape[0])
            else:
                t = t.repeat(repeat)
        t = t.clamp(0.0, 1.0).view(-1, 1, 1)

        latent_w = self.latent_w_min + (self.latent_w_max - self.latent_w_min) * t
        raw_w = self.raw_w_sum - latent_w
        mixed_logits = latent_w * latent_logits + raw_w * raw_logits
        return torch.where(active_edit.bool(), mixed_logits, raw_logits)

    def vq_delta_losses(self, latent_logits, delta_pred, target_ids, source_ids,
                        target_mask, predict_mask, active_edit):
        zero = self.token_emb.weight.new_tensor(0.0)
        if latent_logits is None or delta_pred is None:
            return zero, zero, zero

        active = active_edit.view(-1, 1).bool()
        valid = target_mask.bool() & predict_mask.bool() & active
        valid = valid & (target_ids >= 0) & (target_ids < self.cfg.vq.nb_code)
        valid = valid & (source_ids >= 0) & (source_ids < self.cfg.vq.nb_code)
        if valid.sum() == 0:
            return zero, zero, zero

        labels = torch.where(valid, target_ids, self.mask_id)
        latent_loss = F.cross_entropy(latent_logits, labels, ignore_index=self.mask_id)

        z_src = self.vq_latents(source_ids, valid)
        z_tgt = self.vq_latents(target_ids, valid)
        gt_delta = (z_tgt - z_src).detach()

        weight = valid.float()
        delta_dist = F.smooth_l1_loss(delta_pred, gt_delta, reduction="none").mean(dim=-1)
        delta_loss = (delta_dist * weight).sum() / weight.sum().clamp(min=1.0)

        pred_z = z_src + delta_pred
        pred_n = F.normalize(pred_z, dim=-1)
        tgt_n = F.normalize(z_tgt.detach(), dim=-1)
        src_n = F.normalize(z_src.detach(), dim=-1)
        dist_tgt = 1.0 - (pred_n * tgt_n).sum(dim=-1)
        dist_src = 1.0 - (pred_n * src_n).sum(dim=-1)
        margin = float(getattr(self.cfg.loss, "delta_rank_margin", 0.2))
        rank_dist = F.relu(margin + dist_tgt - dist_src)
        rank_loss = (rank_dist * weight).sum() / weight.sum().clamp(min=1.0)
        return delta_loss, latent_loss, rank_loss

    def InfoNCE_text(self, motion, text_input, m_mask, t_mask, temperature=0.15):
        m_weight = self.motion_mlp(motion) # [B, L, D] -> [B, L, 1]
        t_weight = self.text_mlp(text_input)

        # valid token mask
        m_weight = m_weight.masked_fill(m_mask.unsqueeze(-1) == 0, -1e4)
        t_weight = t_weight.masked_fill(t_mask.unsqueeze(-1) == 0, -1e4)
        m_weight = torch.softmax(m_weight, dim=1)   # token weight
        t_weight = torch.softmax(t_weight, dim=1)

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

    def forward(self, target_input, source_input, text_input, m_lens, has_source, source_m_lens=None,
                source_joints=None, target_joints=None):
        # target/source input are both multi-scale VQ ids
        target_ids, target_mask, time_to_arrival_pe = self.prepare_motion_ids(target_input, m_lens)
        if source_m_lens is None:
            source_m_lens = m_lens   # fallback len
        source_ids, source_mask, source_pe = self.prepare_motion_ids(source_input, source_m_lens)  # source own len
        source_ids_clean = source_ids
        #Motion vqvae embedding
        target_tokens = self.motion_process(target_ids, time_to_arrival_pe)

        # text embedding
        with torch.no_grad():
            text_tokens, text_mask = self.text_emb.get_text_embeddings(text_input)
        text_tokens = self.text_adaptor(text_tokens)
        text_tokens = torch.where(text_mask.unsqueeze(-1).bool(), text_tokens, 0.0)
        text_tokens_for_loss = text_tokens

        # task context: 0=gen, 1=edit
        task_is_edit = has_source.to(target_ids.device).view(-1, 1, 1).bool()

        # source CFG drop trains the editing null-source/null-text branch
        source_cfg_mask = self.CFG_mask(source_ids, self.source_drop_prob).view(-1, 1, 1) & task_is_edit
        text_cfg_mask = self.CFG_mask(text_tokens, self.text_drop_prob)
        text_cfg_mask = text_cfg_mask | source_cfg_mask
        text_tokens = torch.where(text_cfg_mask, self.null_text_embed, text_tokens)

        # source motion is the editing condition; generation samples use null source
        source_ids = self.source_token_dropout(source_ids, source_mask, task_is_edit.squeeze(-1).squeeze(-1))
        source_tokens = self.motion_process(source_ids, source_pe)
        source_present = task_is_edit & ~source_cfg_mask
        source_tokens = torch.where(source_present, source_tokens, self.null_source_embed.expand_as(source_tokens))

        # ControlNet canvas: align source tokens to target timeline
        aligned_src = []
        aligned_src_ids = []
        start = 0
        for scale, ps in zip(self.scales, self.patch_sizes):
            tgt_len = (m_lens // scale).long().clamp(min=1)                                # [B]
            src_len = (source_m_lens // scale).long().clamp(min=1)                         # [B]
            t = torch.arange(ps, device=target_ids.device).float().unsqueeze(0)            # [1, ps]
            idx = torch.round(t * (src_len - 1).float().unsqueeze(1) / (tgt_len - 1).clamp(min=1).float().unsqueeze(1)).long()
            idx = torch.minimum(idx, (src_len - 1).unsqueeze(1))                           # stay inside source valid range
            aligned_src.append(torch.gather(source_tokens[:, start:start + ps], 1, idx.unsqueeze(-1).expand(-1, -1, source_tokens.shape[-1])))
            aligned_src_ids.append(torch.gather(source_ids_clean[:, start:start + ps], 1, idx))
            start += ps
        aligned_src = torch.cat(aligned_src, dim=1)
        aligned_src_ids = torch.cat(aligned_src_ids, dim=1)
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
        # perturb visible token
        if self.training and getattr(self.cfg.training, "pert_prob", 0.0) > 0.:
            pert_mask = (torch.rand_like(target_mask, dtype=torch.float) < self.cfg.training.pert_prob) & (~predict_mask) & target_mask.bool()
            masked_motion_ids = torch.where(pert_mask, torch.randint_like(target_ids, high=self.mask_id), masked_motion_ids)
        # force full mask sample
        masked_motion_ids = torch.where(full_drop, self.mask_id, masked_motion_ids)
        masked_motion_tokens = self.motion_process(masked_motion_ids, time_to_arrival_pe)

        # CA Stack
        text_cross = self.text_cross(masked_motion_tokens, text_tokens, text_mask)
        source_cross = self.source_cross(text_cross, source_tokens, source_mask)
        source_cross = torch.where(source_present, source_cross, text_cross)
        transformer_input = masked_motion_tokens + source_cross

        # source control residuals; gated off for generation samples
        control_residuals = self.control_encoder(transformer_input, aligned_src, cond=text_cross, padding_mask=~target_mask)
        # VQ delta path is only for edit samples with real source and real text
        active_edit = source_present & (~text_cfg_mask)
        delta_residuals, latent_res_logits, latent_logits, delta_pred = self.run_delta_branch(
            transformer_input, text_cross, source_cross, aligned_src_ids,
            target_mask, ~target_mask, active_edit
        )

        # AdaLN uses motion-aligned text condition
        output = self.adaln_encoder(
            x=transformer_input,
            cond=text_cross,
            padding_mask=~target_mask,
            control_residuals=control_residuals,
            control_gate=source_present.float(),
            delta_residuals=delta_residuals,
            delta_gate=active_edit.float() * self.delta_beta
        )
        raw_logits = self.output_process(output)
        if self.use_latent_classifier:
            logits = self.latent_classifier_logits(raw_logits, latent_logits, active_edit, rand_mask_probs)
        else:
            logits = self.apply_vq_delta_logits(raw_logits, latent_res_logits)

        # masked token CE
        ce_loss, _, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        delta_loss, latent_loss, rank_loss = self.vq_delta_losses(
            latent_logits, delta_pred, target_ids, aligned_src_ids,
            target_mask, predict_mask, active_edit
        )

        #Info_NCE loss computation
        loss_mt = self.InfoNCE_text(target_tokens, text_tokens_for_loss, target_mask, text_mask, temperature=0.15)

        return ce_loss, loss_mt.mean(), delta_loss, latent_loss, rank_loss, acc
    
    def forward_with_cond_scale(self, motion_ids, source_ids, text_embs, time_to_arrival_pe, source_pe,
                                source_mask, text_mask, motion_padding_mask, has_source, mask_ratio,
                                cond_scale=3, source_scale=None):    # dual CFG: source-preservation + text-edit
        # Three branches for composable guidance:
        #   A(uncond): null source + null text
        #   B(src)   : source      + null text
        #   C(full)  : source      + text
        # scaled = A + source_scale * (B - A) + cond_scale * (C - B)
        if source_scale is None:
            source_scale = 1.0   # keep source at normal strength; cond_scale only boosts edit text

        input_motion_ids = torch.cat([motion_ids, motion_ids, motion_ids], dim=0)
        input_source_ids = torch.cat([source_ids, source_ids, source_ids], dim=0)
        input_source_mask = torch.cat([source_mask, source_mask, source_mask], dim=0)
        input_text_mask = torch.cat([text_mask, text_mask, text_mask], dim=0)
        input_motion_padding_mask = torch.cat([motion_padding_mask, motion_padding_mask, motion_padding_mask], dim=0)
        input_toa_pe = torch.cat([time_to_arrival_pe, time_to_arrival_pe, time_to_arrival_pe], dim=0)
        input_source_pe = torch.cat([source_pe, source_pe, source_pe], dim=0)

        # text: branch A/B use null text, branch C uses the real edit text
        null_text = self.null_text_embed.expand(text_embs.shape)
        input_text_embs = torch.cat([null_text, null_text, text_embs], dim=0)
        task_is_edit = has_source.view(-1, 1, 1).bool()
        input_has_source = torch.cat([torch.zeros_like(task_is_edit), task_is_edit, task_is_edit], dim=0)
        input_has_text = torch.cat([torch.zeros_like(task_is_edit), torch.zeros_like(task_is_edit), torch.ones_like(task_is_edit)], dim=0)

        # source condition (null for branch A, or for has_source=0 samples)
        input_motion_embs = self.motion_process(input_motion_ids, input_toa_pe)
        source_tokens = self.motion_process(input_source_ids, input_source_pe)
        source_tokens = torch.where(input_has_source, source_tokens, self.null_source_embed.expand_as(source_tokens))

        # ControlNet canvas: same aligned gather as training, per-scale lens derived from the masks
        non_pad = ~input_motion_padding_mask
        aligned_src = []
        aligned_src_ids = []
        start = 0
        for ps in self.patch_sizes:
            tgt_len = non_pad[:, start:start + ps].sum(dim=1).clamp(min=1)                 # [3B]
            src_len = input_source_mask[:, start:start + ps].sum(dim=1).clamp(min=1)       # [3B]
            t = torch.arange(ps, device=motion_ids.device).float().unsqueeze(0)            # [1, ps]
            idx = torch.round(t * (src_len - 1).float().unsqueeze(1) / (tgt_len - 1).clamp(min=1).float().unsqueeze(1)).long()
            idx = torch.minimum(idx, (src_len - 1).unsqueeze(1))                           # stay inside source valid range
            aligned_src.append(torch.gather(source_tokens[:, start:start + ps], 1, idx.unsqueeze(-1).expand(-1, -1, source_tokens.shape[-1])))
            aligned_src_ids.append(torch.gather(input_source_ids[:, start:start + ps], 1, idx))
            start += ps
        aligned_src = torch.cat(aligned_src, dim=1)
        aligned_src_ids = torch.cat(aligned_src_ids, dim=1)
        aligned_src = torch.where(input_has_source, aligned_src, torch.zeros_like(aligned_src))

        # HML-1 style CA
        text_cross = self.text_cross(input_motion_embs, input_text_embs, input_text_mask)
        source_cross = self.source_cross(text_cross, source_tokens, input_source_mask)
        source_cross = torch.where(input_has_source, source_cross, text_cross)
        transformer_input = input_motion_embs + source_cross

        # source branch: branch A gate 0, branch B/C use aligned source
        control_residuals = self.control_encoder(transformer_input, aligned_src, cond=text_cross, padding_mask=input_motion_padding_mask)
        active_edit = input_has_source & input_has_text
        delta_residuals, latent_res_logits, latent_logits, _ = self.run_delta_branch(
            transformer_input, text_cross, source_cross, aligned_src_ids,
            non_pad, input_motion_padding_mask, active_edit
        )

        # predict token logits from current masked ids
        output = self.adaln_encoder(
            x=transformer_input,
            cond=text_cross,
            padding_mask=input_motion_padding_mask,
            control_residuals=control_residuals,
            control_gate=input_has_source.float(),
            delta_residuals=delta_residuals,
            delta_gate=active_edit.float() * self.delta_beta
        )
        raw_logits = self.output_process(output) # [B, L, dim] -> [B, dim, L]
        if self.use_latent_classifier:
            output_logits = raw_logits
        else:
            output_logits = self.apply_vq_delta_logits(raw_logits, latent_res_logits)
        uncond_logits, source_logits, full_logits = output_logits.chunk(3, dim=0)

        if torch.is_tensor(cond_scale):
            cond_scale = cond_scale.to(full_logits.device, dtype=full_logits.dtype).view(-1, 1, 1)
        if torch.is_tensor(source_scale):
            source_scale = source_scale.to(full_logits.device, dtype=full_logits.dtype).view(-1, 1, 1)

        # source_scale controls preservation strength, cond_scale controls edit-text strength
        scaled_logits = uncond_logits + source_scale * (source_logits - uncond_logits) + cond_scale * (full_logits - source_logits)
        if self.use_latent_classifier and latent_logits is not None:
            _, _, latent_full_logits = latent_logits.chunk(3, dim=0)
            scaled_logits = self.latent_classifier_logits(scaled_logits, latent_full_logits, task_is_edit, mask_ratio)
        return scaled_logits #[B, dim, L]

    @torch.no_grad()
    @eval_decorator
    def generate(self, source_input, text_input, m_lens, has_source, t_drop,
                 timesteps: int,
                 cond_scale: int,
                 source_cond_scale=None,   # dual CFG: preservation strength; None -> 1.0
                 source_m_lens=None,       # source len
                 temperature=1,
                 topk_filter_thres=0.95,
                 gsample=False):   #borrow from SnapMogen
        device = self.device
        B = len(m_lens)

        # encode text once
        text_tokens, text_mask = self.text_emb.get_text_embeddings(text_input)
        text_tokens = self.text_adaptor(text_tokens)
        text_mask = text_mask.bool()
        text_cfg_mask = self.CFG_mask(text_tokens, t_drop)
        text_tokens = torch.where(text_cfg_mask,  self.null_text_embed, text_tokens)

        # optional source ids
        if source_input is None:
            source_input = [torch.zeros(B, int(self.full_length//scale), dtype=torch.long, device=device) for scale in self.scales]
        if not torch.is_tensor(has_source):
            has_source = torch.tensor(has_source, device=device).expand(B)
        has_source = has_source.to(device).view(-1, 1).bool()
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
        scores = torch.where(padding_mask, 1e4, 0.0)
        starting_temperature = temperature

        for timestep in torch.linspace(0, 1, timesteps, device=device):
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
                                                  source_scale=source_cond_scale)
            

            logits = logits.permute(0, 2, 1)  # (b, ntoken, seqlen) -> (b, seqlen, ntoken)
            # print(logits.shape, self.cfg.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature - 0.5 * timestep   # temperature linearly reduce, 1 -> 0.5
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

            # keep tokens that were not sampled in this step
            scores = scores.masked_fill(~is_mask, 1e4)

        # split concatenated multi-scale ids back to VQ decoder format
        ids = torch.where(padding_mask, -1, ids)
        return_list = []
        start = 0
        for length in lengths_div:
            return_list.append(ids[..., start:start+length])
            start += length
        # print("Final", ids.max(), ids.min())
        return return_list



