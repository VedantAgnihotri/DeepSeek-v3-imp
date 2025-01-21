import torch
from torch import nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, dim):
        super(RoPE, self).__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) #inverse frequency calc
        self.inv_freq = inv_freq

    def apply_rope(self, tensor, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        sinusoidal_inp = torch.einsum('i,j->ij', t, self.inv_freq)
        sin, cos = sinusoidal_inp.sin(), sinusoidal_inp.cos()

        original_shape = tensor.shape
        tensor_reshaped = tensor.view(*original_shape[:-1], -1, 2)

        sin = sin.unsqueeze(0).unsqueeze(0).unsqueeze(-1) 
        cos = cos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Stack rotated pairs instead of concatenating
        tensor_rotated = torch.stack([-tensor_reshaped[..., 1], tensor_reshaped[..., 0]], dim=-1)

        rotated_tensor = tensor_reshaped * cos + tensor_rotated * sin

        rotated_tensor = rotated_tensor.view(original_shape)
    
        return rotated_tensor

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_embed, n_heads, n_latents):
        super(MultiHeadLatentAttention, self).__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads
        self.n_latents = n_latents

        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"

        self.latents = nn.Parameter(torch.randn(n_latents, d_embed)) # (c_t^k)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.v_proj = nn.Linear(d_embed, d_embed)

        self.out_proj = nn.Linear(d_embed, d_embed)

        self.rope = RoPE(self.d_head)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        latent_q = self.q_proj(self.latents)
        latent_q = latent_q.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, num_latents, embed_dim)

        k = self.k_proj(x)
        v = self.v_proj(x)

        latent_q = latent_q.view(batch_size, self.n_latents, self.n_heads, self.d_head).transpose(1, 2)
        latent_q = self.rope.apply_rope(latent_q, self.n_latents)

        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.rope.apply_rope(k, seq_len)

        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(latent_q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attn_w = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_w, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.n_latents, embed_dim)

        output = self.out_proj(attn_output)
        return output