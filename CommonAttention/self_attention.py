import torch
import torch.nn as nn
import numpy as np
from multiHead_attention import MultiHeadAttention


class SelfAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_x, d_o) -> None:
        super().__init__()

        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(num_heads=n_head, d_k=d_k, d_v=d_v, d_o=d_o)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)
        
    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, out = self.mha(q, k, v, mask=mask)

        return attn, out