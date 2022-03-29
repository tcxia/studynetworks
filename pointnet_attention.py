import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, bidirect) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim // 2 if bidirect else hidden_dim
        self.n_layers = n_layers * 2 if bidirect else n_layers
        self.bidirect = bidirect
        self.lstm = nn.LSTM(
            embedding_dim,
            self.hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=bidirect
        )
        self.h0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        batch_size = embedded_inputs.size(0)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch_size, self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        nn.init.uniform(self.V, -1, 1)

    def forward(self, input, context, mask):
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        attn = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)

        if len(attn[mask]) > 0:
            attn[mask] = self.inf[mask]
        alpha = self.softmax(attn)

        hidden_size = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_size, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


# class Decoder(nn.Module):
