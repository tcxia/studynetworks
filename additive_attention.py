
import torch
import torch.nn as nn

from show_heatmaps import show_heatmaps

class AdditiveAttention(nn.Module):
    def __init__(self, d_q, d_k, nh, dropout=0.1) -> None:
        super().__init__()
        self.w_k = nn.Linear(d_k, nh, bias=False)
        self.w_q = nn.Linear(d_q, nh, bias=False)
        self.w_v = nn.Linear(nh, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, querys, keys, values, valid_lens=None):
        q = self.w_q(querys)
        k = self.w_k(keys)
        features = q.unsqueeze(2) + k.unsqueeze(1)
        features = torch.Tanh(features)

        scores = self.w_v(features)
        scores = scores.squeeze(-1)

        self.attn = self.masked_softmax(scores, valid_lens)
        self.attn = self.dropout(self.attn)
         # 计算两个tensor的矩阵乘法，torch.bmm(a, b)，其中a的shape为[b, h, w]，b的shape为[b, w, h]
         # 这两个tensor的shape必须为3
        out = torch.bmm(self.attn, values)
        return out

    # 用于去除不需要的padding部分。mask部分的attention score可以忽略
    def masked_softmax(self, X, valid_lens):
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)

        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)

            X = self.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)


    def sequence_mask(self, X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X


if __name__ == '__main__':
    querys = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 6))

    attn = AdditiveAttention(d_k=2, d_q=20, nh=8, dropout=0.1)
    attn.eval()
    ret = attn(querys, keys, values)

    show_heatmaps(attn.attn.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')