import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, bidirect) -> None:
        super().__init__()
        '''
            embedding_dim: embedding的维度
            hidden_dim: lstm的隐层单元数量
            n_layers: lstm的层数
        '''

        self.hidden_dim = hidden_dim // 2 if bidirect else hidden_dim
        self.n_layers = n_layers * 2 if bidirect else n_layers
        self.bidirect = bidirect
        self.lstm = nn.LSTM(
            embedding_dim,
            self.hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=bidirect,
        )
        self.h0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        '''
            embedded_inputs: pointer-net的embeded inputs
            hidden: LSTM的隐层输出
        '''
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        batch_size = embedded_inputs.size(0)
        h0 = (
            self.h0.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.n_layers, batch_size, self.hidden_dim)
        )
        c0 = (
            self.h0.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.n_layers, batch_size, self.hidden_dim)
        )

        return h0, c0


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = nn.Parameter(
            torch.FloatTensor([float("-inf")]), requires_grad=False
        )
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


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = Attention(hidden_dim, hidden_dim)

        self.mask = nn.Parameter(torch.ones(1), requires_grad=False)
        self.runner = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedding_inputs, decoder_input, hidden, context):
        batch_size = embedding_inputs.size(0)
        input_len = embedding_inputs.size(1)

        mask = self.mask.repeat(input_len).unsqueeze(0).repeat(batch_size, 1)
        self.attn.init_inf(mask.size())

        runner = self.runner.repeat(input_len)
        for i in range(input_len):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            h, c = hidden
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            input, forget, cell, out = gates.chunk(4, 1)
            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.sigmoid(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            hidden_t, output = self.attn(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        for _ in range(input_len):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            masked_outs = outs * mask
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (
                runner == indices.unsqueeze(1).expand(-1, outs.size()[1])
            ).float()

            mask = mask * (1 - one_hot_pointers)

            embedding_mask = (
                one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            )
            decoder_input = embedding_inputs[embedding_mask.data].view(
                batch_size, self.embedding_dim
            )

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNet(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, lstm_layers, dropout, bidirect
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.bidirect = bidirect
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(
            embedding_dim, hidden_dim, lstm_layers, dropout, bidirect
        )
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = nn.Parameter(
            torch.FloatTensor(embedding_dim), requires_grad=False
        )

        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_len = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        inputs = inputs.view(batch_size * input_len, -1)
        embedding_inputs = self.embedding(inputs).view(batch_size, input_len, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedding_inputs)
        encoder_outputs, encoder_hidden = self.encoder(
            embedding_inputs, encoder_hidden0
        )

        if self.bidirect:
            decoder_hidden0 = (
                torch.cat(encoder_hidden[0][-2:], dim=-1),
                torch.cat(encoder_hidden[1][-2:], dim=-1),
            )

        else:
            decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])

        (outputs, pointers), decoder_hidden = self.decoder(
            embedding_inputs, decoder_input0, decoder_hidden0, encoder_outputs
        )

        return outputs, pointers
