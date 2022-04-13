import torch
import torch.nn as nn
import numpy as np


class ScaleDotProductAttention(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        
        attn = self.softmax(u)
        out = torch.bmm(attn, v)

        return attn, out


