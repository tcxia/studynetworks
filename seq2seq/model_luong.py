import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random



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


class LocalAttention(nn.Module):
    def __init__(self, hidden_size, method, s_len, windows) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.method = method
        self.s_len = s_len
        self.D = windows

        self.omega_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.omega_v = nn.Linear(self.hidden_size, 1)
        self.v = nn.Linear(self.hidden_size, 1)

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, enc_output, s):
        sigmoid_inp = self.omega_v(F.tanh(self.omega_w(s))).squeeze(1)
        S = self.s_len - 2 * self.D - 1
        ps = S * F.sigmoid(sigmoid_inp)
        pt = ps + self.D

        bs = enc_output.shape[0]
        enc_output_ = torch.zeros(1, 2 * self.D + 1, self.hidden_size)
        for i in range(bs):
            enc_output_i = enc_output[i, int(pt[i]) - self.D :int(pt[i]) + self.D + 1, :]
            enc_output_ = torch.cat((enc_output_, enc_output_i.unsqueeze(0)), dim=0)

        enc_output = enc_output_[1:, :, :]
        enc_w_output = enc_output
        seq_len = enc_w_output.size(1)
        sigma = self.D / 2
        alpha_hat = torch.zeros(bs, seq_len)

        if self.method == 'concat':
            s = s.unsqueeze(1).repeat(1, seq_len, 1)
            mul = torch.tanh(self.attn(torch.cat((enc_w_output, s), dim=2)))
            alpha_hat = self.v(mul).squeeze(2)

        elif self.method == 'general':
            enc_w_output = self.attn(enc_w_output)
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_w_output, s).squeeze(2)
        
        elif self.method == 'dot':
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_w_output, s).squeeze(2)

        gauss = []
        for i in range(bs):
            gauss_score = []
            for j in range(int(pt[i]) - self.D, int(pt[i]) + self.D + 1):
                gauss_score_i = math.exp(-(pow((j - pt[i].item()), 2)) / (2 * (pow(sigma, 2))))
                gauss_score.append(gauss_score_i)
            gauss.append(gauss_score)

        gauss = torch.Tensor(gauss)
        energies = alpha_hat * gauss

        return F.softmax(energies, dim=1), enc_w_output

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, input_size, dropout, attention) -> None:
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, dec_input, lasts, lastc, enc_output):
        dec_input = dec_input.unsqueeze(1)
        lastc = lastc.unsqueeze(1)
        gru_input = torch.cat((dec_input, lastc), dim=2)
        dec_output, hidden = self.gru(gru_input, lasts)

        alpha, enc_w_output = self.attention(enc_output, hidden.squeeze(0))

        attn_weight = torch.bmm(alpha.unsqueeze(1), enc_w_output)

        dec_output = self.out(dec_output)

        return dec_output.squeeze(1), attn_weight.squeeze(1), hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decocer = decoder
        self.device = device
    
    
    def forward(self, src, trg, hidden_size, teaching_force_ratio=0.5):
        bs = src.shape[0]
        trg_len = trg.shape[1]
        dim = src.shape[2]

        outputs = torch.zeros(bs, trg_len, dim).to(self.device)

        enc_output, s = self.encoder(src)

        dec_input = torch.zeros(bs, dim).to(self.device)
        dec_context = torch.zeros(bs, hidden_size).to(self.device)

        for t in range(0, trg_len):
            dec_output, dec_context, s = self.decocer(dec_input, s, dec_context, enc_output)
            outputs[:, t, :] = dec_output
            teacher_force = random.random() < teaching_force_ratio
            dec_input = trg[:, t, :] if teacher_force else dec_output

        return outputs