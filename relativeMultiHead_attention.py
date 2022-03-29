import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = int(d_model // num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = np.sqrt(d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.pos = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, pos_embedding, mask=None):
        batch_size = v.size(0)

        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_head)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        pos_embedding = self.pos(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((q + self.u_bias).transpose(1, 2), k.transpose(2, 3))
        pos_score = torch.matmul((q + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_postional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)


    def _compute_relative_postional_encoding(self, pos_score):
        batch_size, num_heads, seq_len1, seq_len2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_len1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_len2 + 1, seq_len1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

