import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekMoe(nn.Module):
    def __init__(self, d_input, N_s, N_r, top_k, gamma=0.01):
        super(DeepSeekMoe, self).__init__()
        self.N_r = N_r
        self.N_s = N_s
        self.top_k = top_k
        self.gamma = gamma # learning rate for load balancing (controls adjustment speed)

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
        self.register_buffer('expert_biases', torch.zeros(N_r))

        #Load tracking
        self.register_buffer('running_load', torch.zeros(N_r)) #tracks long-term load stats :. maintains EMA of expert utilization
        self.momentum = 0.1 #EMA momentum (Exponential Moving Average) - prevents oscillations

    def forward(self, u):
        batch_size, n_latents, input_dim = u.size()

        raw_scores = torch.einsum('bsd, ed -> bse', u, self.centroids) #[Batch, Seq_len, d_model], [n_experts, d_model] -> [Batch, Seq_len, n_experts]
        biased_scores = raw_scores + self.expert_biases

        s = torch.softmax(biased_scores)
        g = s/ (s.sum(dim=-1, keepdim=True)+1e-6) #Normalized Gates

        topk_values, topk_indices = torch.topk(s, self.top_k, dim=-1)
        g_topk = torch.zeros_like(g).scatter_(-1, topk_indices, topk_values)

        if self.training: #track load only during training
            current_load = g_topk.sum(dim=(0,1)).detach()
            self.running_load = (self.momentum * current_load + (1-self.momentum) * self.running_load)

        routed_outputs = sum(
            g_topk[..., i].unsqueeze(-1)*expert(u)
            for i, expert in enumerate(self.routed_experts)
        )

        shared_outputs = sum(expert(u) for expert in self.shared_experts)/self.N_s

        h = u + routed_outputs + shared_outputs
        return h
    
    def update_biases(self):
        if not self.training: #skip biases during inference
            return
        
        avg_load = self.running_load.mean()
        delta = self.gamma * (avg_load - self.running_load) #instantaneous bias adjustment
        self.expert_biases += delta #update biases
