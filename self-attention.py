import torch
import torch.nn as nn
import numpy as np
from init_weights import init_weights


class ScaleDotProductAtten(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.1) -> None:
        super().__init__()
        '''
        d_model: 模型输出维度
        d_k: query和keys的维度
        d_v: values的维度
        h: 多头的数量
        '''
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        init_weights(self.modules())


    def forward(self, querys, keys, values, attn_mask=None, attn_weight=None):
        '''
        querys: [b_s, nq, d_model]
        keys: [b_s, nk, d_model]
        values: [b_s, nk, d_model]
        attn_mask: mask over attention values (b_s, h, nq, nk), True表示masking
        attn_weights: Multiplicative weights for attention values (b_s, h, nq, nk)
        '''
        b_s, nq = querys.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(querys).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_v(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1) # 直接转置
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attn_weight is not None:
            attn = attn * attn_weight
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)
        
        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out

