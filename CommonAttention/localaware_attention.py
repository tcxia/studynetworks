import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    def __init__(self, hidden_dim, smooth=True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)

        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smooth

    def forward(self, query, value, last_attn):
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(
            torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(
                    batch_size, -1, hidden_dim
                )
                + self.value_proj(value.reshape(-1, hidden_dim)).view(
                    batch_size, -1, hidden_dim
                )
                + conv_attn
                + self.bias
            )
        ).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)
        
        out = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)
        return out, attn
