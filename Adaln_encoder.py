import torch
import torch.nn as nn


class AdaLN_Block(nn.Module):
    def __init__(self, model_dim, n_heads, cond_dim, ff_size, dropout=0.1):
        super().__init__()
        # self-attention
        self.attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, model_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        
        # AdaLN 投影層：將 cond 映射成 4 個參數 (gamma_1, beta_1, gamma_2, beta_2)
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, model_dim * 6)
        )

        nn.init.normal_(self.adaln_proj[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adaln_proj[-1].bias)

    def forward(self, input, cond, padding_mask):
        # input = [B, L, D]
        # genearted nomalization factor: beta & gemma
        scale_shift = self.adaln_proj(cond)
        gamma_1, beta_1, gamma_2, beta_2,alpha_1, alpha_2 = scale_shift.chunk(6, dim=-1)
        
        # Self-Attention
        input_norm = self.norm1(input)
        input_modulated = input_norm * (1 + gamma_1) + beta_1 
        
        attn_out, _ = self.attn(query=input_modulated, 
                                key=input_modulated, 
                                value=input_modulated, 
                                key_padding_mask=padding_mask,
                                need_weights=False)
        input = input + alpha_1 * attn_out
        
        #  Feed Forward
        input_norm2 = self.norm2(input)
        input_modulated2 = input_norm2 * (1 + gamma_2) + beta_2
        
        ffn_out = self.ffn(input_modulated2)
        output = input + alpha_2 * ffn_out
        
        return output


class AdaLN_ControlBranch(nn.Module):
    # ControlNet-style trainable copy: reads (main input + aligned source canvas),
    # returns per-layer residuals, each through its own zero-init linear (zero-conv)
    def __init__(self, model_dim, n_heads, cond_dim, ff_size, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([AdaLN_Block(model_dim, n_heads, cond_dim, ff_size, dropout) for _ in range(n_layers)])
        self.zero_in = nn.Linear(model_dim, model_dim)
        self.zero_projs = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(n_layers)])
        self.init_zero_layers()

    def init_zero_layers(self):
        # zero connections: injection starts silent, main branch untouched at step 0
        nn.init.zeros_(self.zero_in.weight)
        nn.init.zeros_(self.zero_in.bias)
        for proj in self.zero_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x, control, cond, padding_mask=None, return_hiddens=False):
        c = x + self.zero_in(control)
        residuals = []
        hiddens = []
        for layer, zero_proj in zip(self.layers, self.zero_projs):
            c = layer(c, cond, padding_mask)
            residuals.append(zero_proj(c))
            hiddens.append(c)
        if return_hiddens:
            return residuals, hiddens
        return residuals   # [r_1 ... r_L], depth-matched to the main blocks


class AdaLN_Encoder(nn.Module):
    def __init__(self, model_dim, n_heads, cond_dim, ff_size, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([AdaLN_Block(model_dim, n_heads, cond_dim, ff_size, dropout) for _ in range(n_layers)])
        
        # DiT 標準架構
        self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.final_adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim,model_dim * 2) # 最後只需要一組 gamma 和 beta (No FFN)
        )

        nn.init.normal_(self.final_adaln_proj[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.final_adaln_proj[-1].bias)

    def forward(self, x, cond, padding_mask=None, control_residuals=None, control_gate=None,
                delta_residuals=None, delta_gate=None):
        """
        x: [B, L, model_dim] -> 純動作特徵 (masked_motion_tokens)
        cond: [B, L, cond_dim] -> 融合後的 Text+Video 條件特徵 (cond_fused)
        control_residuals: per-layer residuals from AdaLN_ControlBranch (ControlNet-style injection)
        control_gate: [B, 1, 1], 1 for edit samples, 0 keeps gen samples structurally untouched
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, cond, padding_mask)
            if control_residuals is not None:
                r = control_residuals[i]
                if control_gate is not None:
                    r = r * control_gate
                x = x + r
            if delta_residuals is not None:
                r = delta_residuals[i]
                if delta_gate is not None:
                    r = r * delta_gate
                x = x + r
            
        # final normalize - without feed forward
        gamma_f, beta_f = self.final_adaln_proj(cond).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + gamma_f) + beta_f
        
        return x
