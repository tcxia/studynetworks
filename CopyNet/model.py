import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        '''
            x: [bs, seq]
        '''
        embedded = self.embed(x)
        out, h = self.gru(embedded) # out: [bs, seq, hid * 2]
        return out, h


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size + hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.Ws = nn.Linear(hidden_size * 2, hidden_size)
        self.Wo = nn.Linear(hidden_size, vocab_size)
        self.Wc = nn.Linear(hidden_size * 2, hidden_size)

        self.nonlinear = nn.Tanh()

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order):

        bs = encoded.size(0)
        seq = encoded.size(1)
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        if order == 0:
            prev_state = self.Ws(encoded[:, 1])
            weighted = torch.Tensor(bs, 1, hidden_size * 2).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0)

        gru_input = torch.cat([self.embed(input_idx).unsqueeze(1), weighted], 2)
        _, state = self.gru(gru_input, prev_state)
        state = state.squeeze()

        score_g = self.Wo(state)

        score_c = F.tanh(self.Wc(encoded.contiguous().view(-1, hidden_size*2)))
        score_c = score_c.view(bs, -1, hidden_size)
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze()

        encoded_mask = torch.Tensor(np.array(encoded_idx == 0, dtype=float) * (-1000))
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)

        score_c = score_c + encoded_mask
        score_c = F.tanh(score_c)

        score = torch.cat([score_g, score_c], 1)
        probs = F.softmax(score)
        prob_g = probs[:, :vocab_size]
        prob_c = probs[:, vocab_size:]

        prob_c_to_g = self.to_cuda(torch.Tensor(bs, vocab_size).zero_())
        prob_c_to_g = Variable(prob_c_to_g)
        for b_idx in range(bs):
            for s_idx in range(seq):
                prob_c_to_g[b_idx, encoded_idx[b_idx, s_idx]] = prob_c_to_g[b_idx, encoded_idx[b_idx, s_idx]] + prob_c[b_idx, s_idx]

        
        out = prob_g + prob_c_to_g
        out = out.unsqueeze(1)

        idx_from_input = []
        for i, j in enumerate(encoded_idx):
            idx_from_input.append([int(k == input_idx[i].data[0]) for k in j])
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)

        for i in range(bs):
            if idx_from_input[i].sum().data[0] > 1:
                idx_from_input[i] = idx_from_input[i] / idx_from_input[i].sum().data[0]

        attn = prob_c * idx_from_input

        attn = attn.unsqueeze(1)
        weighted = torch.bmm(attn, encoded)
        return out, state, weighted




    def to_cuda(self, tensor):
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor
