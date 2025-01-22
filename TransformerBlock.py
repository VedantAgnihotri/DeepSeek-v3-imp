import torch
import torch.nn as nn
import torch.nn.functional as F
from MLAwRoPE import MultiHeadLatentAttention
from DS_MoE import DeepSeekMoe

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return x/norm * self.scale

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, N_s, N_r, top_k, n_latents):
        super(TransformerBlock, self).__init__()

        self.attn_norm = RMSNorm(d_model)
        self.mla_with_rope = MultiHeadLatentAttention(d_embed=d_model, n_heads=n_heads, n_latents=n_latents)

        self.moe_norm = RMSNorm(d_model)
        self.moe = DeepSeekMoe(d_input=d_model, N_s=N_s, N_r=N_r, top_k=top_k)

    def _attn_block(self, x):
        x_norm = self.attn_norm(x)
        attn_out = self.mla_with_rope(x_norm)
        return x + attn_out
    
    def _moe_block(self, x):
        x_norm = self.moe_norm(x)
        moe_out = self.moe(x_norm)
        return x + moe_out
    
    def forward(self, x):
        attn_out = self._attn_block(x)
        moe_out = self._moe_block(attn_out)
        return moe_out
