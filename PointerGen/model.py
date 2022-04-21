import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.normal_(std=1e-4)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True) # 双向LSTM
        self._init_wt(self.lstm)

        self.w_h = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _  = pad_packed_sequence(output, batch_first=True)
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * self.hidden_size)
        encoder_feature = self.w_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

    def _init_wt(self, lstm):
        for names in lstm._all_weights:
            for name in names:
                if name.startswith('weight_'):
                    wt = getattr(lstm, name)
                    wt.data.uniform_(-0.02, 0.02)
                elif name.startswith('bias_'):
                    bias = getattr(lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data.fill_(0.)
                    bias.data[start:end].fill_(1.)


class ReduceState(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden):
        h, c = hidden
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        