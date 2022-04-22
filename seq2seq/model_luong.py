
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def forward(self, src):
        output, hidden = self.gru(src)
        return output, hidden

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, method) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, enc_output, s):
        bs = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        alpha_hat = torch.zeros(bs, seq_len)

        if self.method == 'concat':
            srclen = enc_output.shape[1]
            s = s.unsqueeze(1).repeat(1, srclen, 1)
            mul = torch.tanh(self.attn(torch.cat((enc_output, s), dim=2)))
            alpha_hat = self.v(mul).squeeze(2)

        elif self.method == 'general':
            enc_output = self.attn(enc_output)
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_output, s).squeeze(2)

        elif self.method == 'dot':
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_output, s).squeeze(2)

        return F.softmax(alpha_hat, dim=1)


# class LocalAttention(nn.Module):
