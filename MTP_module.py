import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerBlock import TransformerBlock

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return x/norm * self.scale
    
class MTPModule(nn.Module):
    def __init__(self, d_model, vocab_size, k_depth, main_emb_layer):
        super().__init__()
        self.depth = k_depth
        self.start_pos = 2
        self.embedding = main_emb_layer
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)

        self.linear_proj = nn.Linear(2*d_model, d_model) # for concat emb
        self.transformer_block = TransformerBlock(
            d_model=d_model,
            n_heads=8,
            N_s=4,
            N_r=64,
            top_k=2,
            n_latents=64
        )
        self.head = nn.Linear(d_model, vocab_size) #shared

    def forward(self, input_ids, targets=None):
        embeddings = self.embedding(input_ids)

        current_emb = embeddings[:, :-self.depth, :]
        future_emb = embeddings[:, self.depth:, :]
        concat =  torch.concat([current_emb, future_emb], dim=-1)

        proj = self.linear_proj(concat)
        x = self.rms_norm1(proj)

        x = self.transformer_block(x)
        x = self.rms_norm2(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            mtp_targets = targets[:, self.depth:-self.depth]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                mtp_targets.contiguous().view(-1),
                ignore_index=-1
            )
        return logits, loss
