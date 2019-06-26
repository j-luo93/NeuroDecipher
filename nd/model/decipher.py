import torch
import torch.nn as nn

from arglib import use_arguments_as_properties
from dev_misc import Map, clear_cache, get_tensor, get_zeros
from nd.dataset.charset import PAD_ID, get_charset
from nd.flow.edit_dist import compute_expected_edits
from nd.flow.min_cost_flow import min_cost_flow
from nd.magic_tensor.core import MagicTensor

from .lstm_state import LSTMState
from .modules import (GlobalAttention, MultiLayerLSTMCell,
                      NormControlledResidual, UniversalCharEmbedding)


@use_arguments_as_properties('char_emb_dim', 'hidden_size', 'num_layers', 'dropout', 'universal_charset_size', 'lost_lang', 'known_lang', 'norms_or_ratios', 'control_mode', 'residual')
class DecipherModel(nn.Module):

    def __init__(self, trie):
        super().__init__()

        self.encoder = nn.LSTM(self.char_emb_dim, self.hidden_size, num_layers=self.num_layers,
                               dropout=self.dropout, bidirectional=True, batch_first=True)
        self.decoder = MultiLayerLSTMCell(self.char_emb_dim + self.hidden_size,
                                          self.hidden_size, self.num_layers, self.dropout)
        self.attention = GlobalAttention(2 * self.hidden_size, self.hidden_size, dropout=self.dropout)
        langs = (self.lost_lang, self.known_lang)
        self.char_emb = UniversalCharEmbedding(langs, self.char_emb_dim, self.universal_charset_size)
        self.hidden = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.char_emb_dim),
            nn.LeakyReLU())
        self.drop = nn.Dropout(self.dropout)
        if self.residual:
            self.controlled_residual = NormControlledResidual(
                norms_or_ratios=self.norms_or_ratios, control_mode=self.control_mode)
        self.trie = trie

    def encode(self, id_seqs, lengths):
        inp_enc = self.char_emb(id_seqs, self.lost_lang)  # bs x L x d
        # inp_enc = self.drop(inp_enc)
        bs = inp_enc.shape[0]
        inp_packed = nn.utils.rnn.pack_padded_sequence(self.drop(inp_enc), lengths, batch_first=True)
        h = get_zeros(2 * self.num_layers, bs, self.hidden_size)  # NOTE bidirectional, therefore 2
        c = get_zeros(2 * self.num_layers, bs, self.hidden_size)
        h_s_packed, encoding = self.encoder(inp_packed, (h, c))
        h_s = nn.utils.rnn.pad_packed_sequence(h_s_packed, batch_first=True)[0]
        encoding = LSTMState.from_pytorch(encoding)
        return inp_enc, h_s, encoding

    def forward(self, batch):
        # Remember to clear cache.
        clear_cache()

        lost = batch.lost.lang
        known = batch.known.lang
        # Encode.
        emb_s, h_s, encoding = self.encode(batch.lost.id_seqs, batch.lost.lengths)
        mask_lost = (batch.lost.id_seqs != PAD_ID).float()  # bs x sl
        # Start decoding.
        bs, sl, _ = h_s.shape
        input_emb = self.char_emb.get_start_emb(known).expand(bs, -1)  # bs x d
        state = encoding
        h_tilde = get_zeros(bs, self.hidden_size)
        empty_ctx_s = get_zeros(bs, self.hidden_size * 2)
        max_len = max(batch.known.lengths)
        all_log_probs = list()
        all_almt_distrs = list()
        for dec_step in range(max_len):
            input_ = torch.cat([h_tilde, input_emb], dim=-1)
            input_ = self.drop(input_)
            state = self.decoder(input_, state)
            ctx_t = state.get_output()  # bs x d
            # get ctx_s
            almt_distr = self.attention(ctx_t, h_s, mask_lost)
            ctx_s = (almt_distr.view(bs, sl, 1) * h_s).sum(dim=1)  # bs x 2d
            # get h_tilde
            cat = torch.cat([ctx_s, ctx_t], dim=-1)
            h_tilde_rnn = self.hidden(self.drop(cat))
            if self.residual:
                ctx_s_emb = (almt_distr.view(bs, sl, 1) * emb_s).sum(dim=1)  # bs x d
                h_tilde = self.controlled_residual(ctx_s_emb, h_tilde_rnn)
            else:
                h_tilde = h_tilde_rnn
            # get probs
            logits = self.char_emb.project(self.drop(h_tilde), known)
            log_probs = torch.log_softmax(logits, dim=-1)  # bs x num_char
            probs = log_probs.exp()
            input_emb = self.char_emb.soft_emb(probs, known)
            # Collect stuff.
            all_log_probs.append(log_probs.t())
            all_almt_distrs.append(almt_distr)

        log_probs = torch.stack(all_log_probs, dim=0)  # tl x nc x bs
        almt_distr = torch.stack(all_almt_distrs, dim=1)  # bs x tl x sl

        ret = self.trie.analyze(log_probs, almt_distr,
                                batch.known.words, batch.lost.lengths)
        ret.log_probs = log_probs
        ret.valid_log_probs = MagicTensor(ret.valid_log_probs, batch.lost.words, batch.known.words)
        return ret


@use_arguments_as_properties('n_similar')
class DecipherModelWithFlow(DecipherModel):

    def forward(
            self,
            batch,
            num_cognates=None,
            mode='mle',
            edit=True,
            capacity=1):
        assert mode in ['mle', 'flow']
        if mode == 'mle':
            ret = super().forward(batch)
        else:
            assert not self.training
            with torch.no_grad():
                ret = super().forward(batch)
                known = batch.known.lang
                known_forms = batch.known.forms
                known_charset = get_charset(known)
                expected_edits = compute_expected_edits(
                    known_charset, ret.log_probs, known_forms, ret.valid_log_probs, edit=edit)
                flow, cost = min_cost_flow(expected_edits.cpu().numpy(), num_cognates,
                                           capacity=capacity, n_similar=self.n_similar)
                flow = MagicTensor(get_tensor(flow), batch.lost.words, batch.known.words)
                ret.update(flow=flow, cost=cost, expected_edits=expected_edits)
        return ret
