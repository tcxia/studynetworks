from black import main
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, project_dim, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scale = 1 / math.sqrt(self.head_dim)
        self.q_ff = nn.Linear(embed_dim, embed_dim)
        self.k_ff = nn.Linear(embed_dim, embed_dim)
        self.v_ff = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * project_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * project_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weight=False,
        attn_mask=None,
    ):
        tgt_len = query.size(0)
        src_len = key.size(0)
        bs = query.size(1)

        q = (
            self.q_ff(query)
            .view(tgt_len, bs * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            self.k_ff(key)
            .view(src_len, bs * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            self.v_ff(value)
            .view(src_len, bs * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        e = (
            self.k_proj(key)
            .view(src_len, bs * self.num_heads, self.project_dim)
            .permute(1, 2, 0)
        )
        f = (
            self.v_proj(value)
            .view(src_len, bs * self.num_heads, self.project_dim)
            .permute(1, 2, 0)
        )

        # todo @标识符
        attn = self.scale * q @ (e @ k).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.type == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn = attn.view(bs * self.num_heads, tgt_len, self.project_dim)
        attn = F.dropout(
            F.softmax(attn, dim=-1), p=self.dropout, training=self.training
        )
        out = attn @ (f @ v)
        out = self.out(
            out.transpose(0, 1).contiguous().view(tgt_len, bs, self.embed_dim)
        )
        if need_weight:
            attn = (
                attn.view(bs, self.num_heads, tgt_len, self.project_dim).sum(dim=1)
                / self.num_heads
            )
            return out, attn
        else:
            return out, None


def test_func():
    x = torch.tensor([[[1, 2, 3], [4, 5, 6], [6, 7, 8]]])
    y = torch.tensor([[[1, 2, 3], [4, 5, 6], [6, 7, 8]]])
    print("x_shape", x.shape)
    print("y_shape", y.shape)

    c = x @ y
    print("c_shape", c.shape)
    print(c)


if __name__ == "__main__":
    test_func()
