import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from dev_misc import cache
from nd.dataset.charset import SOW_ID, get_charset

from .lstm_state import LSTMState


class UniversalCharEmbedding(nn.Module):

    def __init__(self, langs, char_emb_dim, universal_charset_size, mapping_temperature=0.0):
        super(UniversalCharEmbedding, self).__init__()
        self.langs = langs
        self.charsets = {l: get_charset(l) for l in langs}
        self.char_emb_dim = char_emb_dim
        self.universal_charset_size = universal_charset_size
        self.mapping_temperature = mapping_temperature
        self.char_emb = nn.Embedding(self.universal_charset_size, self.char_emb_dim)
        self.char_weights = nn.ModuleDict({
            l: nn.Embedding(len(self.charsets[l]), self.universal_charset_size)
            for l in self.langs})

    def forward(self, char_seq, lang):
        char_emb = self.get_char_weight(lang)
        return char_emb[char_seq]

    def project(self, input_, lang):
        char_emb = self.get_char_weight(lang)
        return input_.matmul(char_emb.t())

    @cache(full=True)
    def get_char_weight(self, lang):
        mapping = self.mapping(lang)
        char_emb = mapping.matmul(self.char_emb.weight)
        return char_emb

    @cache(full=True)
    def mapping(self, lang):
        weight = self.char_weights[lang].weight
        if self.mapping_temperature > 0.0:
            # NOTE use log_softmax first for more numerical stability
            weight = torch.log_softmax(weight / self.mapping_temperature, dim=-1).exp()
        return weight

    def char_sim_mat(self, lang1, lang2):
        x = normalize(self.mapping(lang1), dim=-1)
        y = normalize(self.mapping(lang2), dim=-1)
        mat = x.matmul(y.t())
        return mat

    def char_softmax(self, lang1, lang2):
        w1 = self.get_char_weight(lang1)
        w2 = self.get_char_weight(lang2)
        mat = w1.matmul(w2.t())
        l1_l2 = mat.log_softmax(dim=-1).exp()
        l2_l1 = mat.log_softmax(dim=0).exp().t()
        return l1_l2, l2_l1

    def char_mapping(self, l1, l2):
        l1_l2, l2_l1 = self.char_softmax(l1, l2)

        def get_topk(a2b, a_cs, b_cs):
            s, idx = a2b[4:].topk(3, dim=-1)
            a = a_cs.id2char(np.arange(4, len(a2b)).reshape(1, -1)).reshape(-1)
            b = b_cs.id2char(idx.cpu().numpy())
            d = {aa: ' '.join(bb) for aa, bb in zip(a, b)}
            return d, s
        l1_l2 = get_topk(l1_l2, self.charsets[l1], self.charsets[l2])
        l2_l1 = get_topk(l2_l1, self.charsets[l2], self.charsets[l1])
        return l1_l2, l2_l1

    def soft_emb(self, weight, lang):
        char_emb = self.get_char_weight(lang)
        return weight.matmul(char_emb)

    def get_start_emb(self, lang):
        char_emb = self.get_char_weight(lang)
        return char_emb[SOW_ID]


class MultiLayerLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(MultiLayerLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)

        cells = [nn.LSTMCell(input_size, hidden_size)] + \
                [nn.LSTMCell(hidden_size, hidden_size) for _ in range(self.num_layers - 1)]
        self.cells = nn.ModuleList(cells)

    def init_state(self, bs, encoding):
        states = list()
        for _ in range(self.num_layers):
            state = (encoding[0].mean(dim=0), encoding[1].mean(dim=0))
            states.append(state)
        return LSTMState(states)

    def forward(self, input_, states):
        assert len(states) == self.num_layers

        new_states = list()
        for i in range(self.num_layers):
            new_state = self.cells[i](input_, states.get(i))
            new_states.append(new_state)
            input_ = new_state[0]
            input_ = self.drop(input_)
        return LSTMState(new_states)

    def extra_repr(self):
        return '%d, %d, num_layers=%d' % (self.input_size, self.hidden_size, self.num_layers)


class GlobalAttention(nn.Module):

    def __init__(self, input_src_size, input_tgt_size, dropout=0.0):
        super(GlobalAttention, self).__init__()

        self.input_src_size = input_src_size
        self.input_tgt_size = input_tgt_size
        self.dropout = dropout

        self.Wa = nn.Parameter(torch.Tensor(input_src_size, input_tgt_size))
        self.drop = nn.Dropout(self.dropout)

    @cache(full=False)
    def _get_Wh_s(self, h_s):
        bs, l, _ = h_s.shape
        # There is some weird bug with dropout layer if dropout rate is zero
        Wh_s = self.drop(h_s).reshape(bs * l, -1).mm(self.Wa).view(bs, l, -1)
        return Wh_s

    def forward(self, h_t, h_s, mask_src):
        bs, sl, ds = h_s.size()
        dt = h_t.shape[-1]
        Wh_s = self._get_Wh_s(h_s)  # bs x sl x dt
        scores = Wh_s.matmul(self.drop(h_t).unsqueeze(dim=-1)).squeeze(dim=-1)  # bs x sl

        scores = scores * mask_src + (-9999.) * (1.0 - mask_src)
        almt_distr = nn.functional.log_softmax(scores, dim=-1).exp()  # bs x sl
        return almt_distr

    def extra_repr(self):
        return 'src=%d, tgt=%d' % (self.input_src_size, self.input_tgt_size)


class NormControlledResidual(nn.Module):

    def __init__(self, norms_or_ratios=None, multiplier=1.0, control_mode=None):
        super(NormControlledResidual, self).__init__()

        assert control_mode in ['none', 'relative', 'absolute']

        self.control_mode = control_mode
        self.norms_or_ratios = None
        if self.control_mode in ['relative', 'absolute']:
            self.norms_or_ratios = norms_or_ratios
            if self.control_mode == 'relative':
                assert self.norms_or_ratios[0] == 1.0

        self.multiplier = multiplier

    def anneal_ratio(self):
        if self.control_mode == 'relative':
            new_ratios = [self.norms_or_ratios[0]]
            for r in self.norms_or_ratios[1:]:
                r = min(r * self.multiplier, 1.0)
                new_ratios.append(r)
            self.norms_or_ratios = new_ratios
            logging.debug('Ratios are now [%s]' % (', '.join(map(lambda f: '%.2f' % f, self.norms_or_ratios))))

    def forward(self, *inputs):
        if self.control_mode == 'none':
            output = sum(inputs)
        else:
            assert len(inputs) == len(self.norms_or_ratios)
            outs = list()
            if self.control_mode == 'absolute':
                for inp, norm in zip(inputs, self.norms_or_ratios):
                    if norm >= 0.0:  # NOTE a negative value means no control applied
                        outs.append(normalize(inp, dim=-1) * norm)
                    else:
                        outs.append(inp)
            else:
                outs.append(inputs[0])
                norm_base = inputs[0].norm(dim=-1, keepdim=True)
                for inp, ratio in zip(inputs[1:], self.norms_or_ratios[1:]):
                    if ratio >= 0.0:  # NOTE same here
                        norm_actual = inp.norm(dim=-1, keepdim=True)
                        max_norm = norm_base * ratio
                        too_big = norm_actual > max_norm
                        adjusted_norm = torch.where(too_big, max_norm, norm_actual)
                        outs.append(normalize(inp, dim=-1) * adjusted_norm)
                    else:
                        outs.append(inp)
            output = sum(outs)
        return output
