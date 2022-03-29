import torch
import torch.nn as nn
import numpy as np
from scaleDotProduct_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_o, num_heads=8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_model, num_heads * d_k)
        self.fc_k = nn.Linear(d_model, num_heads * d_k)
        self.fc_v = nn.Linear(d_model, num_heads * d_v)

        self.attn = ScaleDotProductAttention(scale = np.power(d_k, 0.5))

        self.fc_o = nn.Linear(num_heads * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        batch, nq, d_q = q.size()
        batch, nk, d_k = k.size()
        batch, nv, d_v = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.view(batch, nq, self.num_heads, d_q).permute(2, 0, 1, 3).contiguous().view(-1, nq, d_q)
        k = k.view(batch, nk, self.num_heads, d_k).permute(2, 0, 1, 3).contiguous().view(-1, nk, d_k)
        v = v.view(batch, nv, self.num_heads, d_v).permute(2, 0, 1, 3).contiguous().view(-1, nv, d_v)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        attn, out = self.attn(q, k, v, mask)
        out = out.view(self.num_heads, batch, nq, d_v).permute(1, 2, 0, 3).contiguous().view(batch, nq, -1)
        out = self.fc_o(out)
        
        return attn, out


