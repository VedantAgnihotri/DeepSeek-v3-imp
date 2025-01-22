import torch
import torch.nn as nn
import torch.nn.functional as F
from MLAwRoPE import MultiHeadLatentAttention
from TransformerBlock import TransformerBlock
from DS_MoE import DeepSeekMoe

class DeepSeekModel(nn.Module):
    def __init__(
            self,
            seq_len: int,
            vocab_size,
            d_model: int = 512,
            n_heads: int = 8,
            N_s: int = 4,
            N_r: int = 4,
            top_k: int = 2,
            n_latents: int = 64,
            num_layers: int = 12
    ):
        super().__init__()

        self.seq_len = seq_len  
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                N_s=N_s,
                N_r=N_r,
                top_k=top_k,
                n_latents=n_latents
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def forward(self, x, targets=None):
        x = self.token_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)

        #calc loss if targets provided--
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
