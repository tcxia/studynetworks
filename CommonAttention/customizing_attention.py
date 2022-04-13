import torch
import torch.nn as nn
from scaleDotProduct_attention import ScaleDotProductAttention


class CustomizingAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, conv_out_channel=10) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim // num_heads)
        self.scaled_dot_attn = ScaleDotProductAttention(self.dim)

        self.conv1d = nn.Conv1d(1, conv_out_channel, kernel_size=3, padding=1)
        self.query = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))

    def forward(self, q, v, last_attn):
        batch_size, q_len, v_len = v.size(0), q.size(1), v.size(1)

        if last_attn is None:
            last_attn = v.new_zeros(batch_size * self.num_heads, v_len)

        loc_energy = self.get_loc_energy(last_attn, batch_size, v_len)
        
        q = self.query(q).view(batch_size, q_len, self.num_heads * self.dim)
        v = self.value(v).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        q = q.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        v = v.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        q = q.contiguous().view(-1, q_len, self.dim)
        v = v.contiguous().view(-1, v_len, self.dim)

        out, attn = self.scaled_dot_attn(q, v)
        attn = attn.squeeze()

        out = out.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        out = out.coutiguous().view(batch_size, q_len, -1)

        return out, attn



    def get_loc_energy(self, last_attn, batch_size, v_len):
        conv_feat = self.conv1d(last_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)
        return loc_energy