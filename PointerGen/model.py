import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.normal_(std=1e-4)

        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )  # 双向LSTM
        self._init_wt(self.lstm)

        self.w_h = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * self.hidden_size)
        encoder_feature = self.w_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

    def _init_wt(self, lstm):
        for names in lstm._all_weights:
            for name in names:
                if name.startswith("weight_"):
                    wt = getattr(lstm, name)
                    wt.data.uniform_(-0.02, 0.02)
                elif name.startswith("bias_"):
                    bias = getattr(lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data.fill_(0.0)
                    bias.data[start:end].fill_(1.0)


class ReduceState(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden):
        h, c = hidden

        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden_reduce_h = F.relu(self.reduce_h(h_in))

        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden_reduce_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduce_h.unsqueeze(0), hidden_reduce_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self, hidden_size, is_coverage=False) -> None:
        super().__init__()
        self.is_coverage = is_coverage
        self.hidden_size = hidden_size

        if is_coverage:
            self.W_c = nn.Linear(1, hidden_size * 2, bias=False)

        self.decode_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(
        self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage
    ):
        b, t_k, n = list(encoder_outputs.size())
        dec_feat = self.decode_proj(s_t_hat)
        dec_feat_expanded = dec_feat.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_feat_expanded = dec_feat_expanded.view(-1, n)

        attn_feat = encoder_feature + dec_feat_expanded

        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)
            coverage_feat = self.W_c(coverage_input)
            attn_feat = attn_feat + coverage_feat

        e = F.relu(attn_feat)
        scores = self.v(e)
        scores = scores.view(-1, t_k)

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask
        norm_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / norm_factor

        attn_dist = attn_dist.unsqueeze(1)
        c_t = torch.bmm(attn_dist, encoder_outputs)
        c_t = c_t.view(-1, self.hidden_size * 2)

        attn_dist = attn_dist.view(-1, t_k)

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pointer_gen=False) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.pointer_gen = pointer_gen

        self.attn_net = Attention()

        self.embedding = nn.Embedding()

        self.x_content = nn.Linear(hidden_size * 2 + embed_size, embed_size)

        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False
        )

        if pointer_gen:
            self.p_gen = nn.Linear(hidden_size * 4 + embed_size, 1)

        self.out1 = nn.Linear(hidden_size * 3, hidden_size)
        self.out2 = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        y_t_1,
        s_t_1,
        encoder_outputs,
        encoder_feature,
        enc_padding_mask,
        c_t_1,
        extra_zeros,
        enc_batch_extend_vocab,
        coverage,
        step,
    ):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat(
                (
                    h_decoder.view(-1, self.hidden_size),
                    c_decoder.view(-1, self.hidden_size),
                ),
                1,
            )
            c_t, _, coverage_next = self.attn_net(
                s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage
            )
            coverage = coverage_next

        y_t_1_embed = self.embedding(y_t_1)
        x = self.x_content(torch.cat((c_t_1, y_t_1_embed), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat(
            (
                h_decoder.view(-1, self.hidden_size),
                c_decoder.view(-1, self.hidden_size),
            ),
            1,
        )
        c_t, attn_dist, coverage_next = self.attn_net(
            s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage
        )

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hidden_size), c_t), 1)
        output = self.out1(output)

        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(object):
    def __init__(self, model_file_path=None, is_eval=None, use_cuda=False) -> None:
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        decoder.embedding.weight = encoder.embedding.weight  # 权重共享

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(
                model_file_path, map_location=lambda storage, location: storage
            )
            self.encoder.load_state_dict(state["encoder_state_dict"])
            self.decoder.load_state_dict(state["decoder_state_dict"], strict=False)
            self.reduce_state.load_state_dict(state["reduce_state_dict"])
