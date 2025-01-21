import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekMoe(nn.Module):
    def __init__(self, d_input, N_s, N_r, top_k):
        super(DeepSeekMoe, self).__init__()
        self.N_r = N_r
        self.N_s = N_s
        self.top_k = top_k

        self.routed_experts = nn.ModuleList(
            [
                nn.Linear(d_input, d_input)
                for _ in range(N_r)
            ]
        )

        self.shared_experts = nn.ModuleList(
            [
                nn.Linear(d_input, d_input)
                for _ in range(N_s)
            ]
        )

        self.centroids = nn.Parameter(torch.randn(N_r, d_input))

    def forward(self, u):
        batch_size, n_latents, input_dim = u.size()

        s = torch.sigmoid(torch.einsum('btd, rd -> btr', u, self.centroids))

        g = s/s.sum(dim=-1, keepdim=True)

        topk_values, topk_indices = torch.topk(s, self.top_k, dim=-1)
        g_topk = torch.zeros_like(g).scatter_(-1, topk_indices, topk_values)

        routed_outputs = torch.zeros_like(u)
        for i, expert in enumerate(self.routed_experts):
            routed_outputs += g_topk[..., i].unsqueeze(-1)*expert(u)

        shared_outputs = torch.zeros_like(u)
        for expert in self.shared_experts:
            shared_outputs += expert(u)

        h = u + routed_outputs + shared_outputs
        return h