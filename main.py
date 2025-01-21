import torch
import torch.nn as nn
import torch.nn.functional as F
from MLAwRoPE import MultiHeadLatentAttention
from DS_MoE import DeepSeekMoe

class MLAwRoPEWithMoE(nn.Module):
    def __init__(self, d_model, n_heads, N_s, N_r, top_k, n_latents):
        super(MLAwRoPEWithMoE, self).__init__()

        self.mla_with_rope = MultiHeadLatentAttention(d_embed=d_model, n_heads=n_heads, n_latents=n_latents)
        self.moe = DeepSeekMoe(d_input=d_model, N_s=N_s, N_r=N_r, top_k=top_k)

    def forward(self, x):
        mla_output = self.mla_with_rope(x) #[batch_size, n_latents, d_embed]

        moe_output = self.moe(mla_output)

        return moe_output
    

if __name__ == "__main__":
    # Example configuration
    batch_size = 32
    seq_len = 128
    d_embed = 512
    n_heads = 8
    n_latents = 64
    N_r = 4  # Number of routed experts
    N_s = 4  # Number of shared experts
    top_k = 2
    d_input = d_embed

    model = MLAwRoPEWithMoE(d_embed, n_heads, N_s, N_r, top_k, n_latents)

    x = torch.randn(batch_size, seq_len, d_embed)

    output = model(x)
    print(f"Output shape: {output.shape}")  