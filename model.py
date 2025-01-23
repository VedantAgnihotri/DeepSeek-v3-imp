import torch
import torch.nn as nn
import torch.nn.functional as F
from MLAwRoPE import MultiHeadLatentAttention
from TransformerBlock import TransformerBlock
from DS_MoE import DeepSeekMoe
from MTP_module import MTPModule

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
            num_layers: int = 12,
            mtp_depths = [2, 4, 8],
            lambda_mtp=0.1
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

        self.mtp_modules = nn.ModuleList([
            MTPModule(d_model=d_model, vocab_size=vocab_size, k_depth=depth, main_emb_layer=self.token_emb)
            for depth in mtp_depths
        ])

        self.main_norm = nn.LayerNorm(d_model)
        self.main_head = nn.Linear(d_model, vocab_size)
        self.lambda_mtp = lambda_mtp

        self._init_weights()

    def forward(self, input_ids, targets=None):
        x = self.token_emb(input_ids)

        for block in self.blocks:
            x = block(x)
        main_logits = self.main_head(self.main_norm(x))
        
        # MTP loss
        mtp_loss = 0
        if targets is not None and self.training:
            for mtp_module in enumerate(self.mtp_modules):
                _, loss = mtp_module(input_ids, targets)
                mtp_loss += loss
            
            mtp_loss = self.lambda_mtp * mtp_loss / len(self.mtp_modules)


        # Combine losses
        total_loss = None
        if targets is not None:
            main_loss = F.cross_entropy(
                main_logits.view(-1, main_logits.size(-1)), # flattens logits to [Batch_size * seq_len -> individual preds]
                targets.view(-1), # flattens targets to [Batch_size * seq_len -> individual targets]
                ignore_index=-1 #for padded tokens
            )
            total_loss = main_loss + mtp_loss

        return main_logits, total_loss
    

@torch.no_grad()
def generate(self, input_ids, max_new_tokens=100, temp=1.0, top_k=50):
    for _ in range(max_new_tokens):
        self.eval()
        logits, _ = self(input_ids[:, -self.seq_len:]) #ensures only the most recent seq_len tokens are used.
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf') # masks all logits below k-th largest to -inf
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples = 1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
