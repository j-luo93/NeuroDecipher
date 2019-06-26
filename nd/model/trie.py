import logging
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.sparse
from pytrie import SortedStringTrie

from arglib import has_properties
from dev_misc import Map, get_tensor, get_zeros
from nd.dataset.charset import EOW, EOW_ID, PAD_ID, get_charset
from nd.dataset.vocab import get_words

from .lstm_state import LSTMState


@has_properties('lang')
class Trie:
    '''
    A trie that efficiently computes the log probs for every word.
    '''

    def __init__(self, lang):

        words = get_words(lang)
        self._max_length = max(map(len, words))  # NOTE EOW has been taken care of by __len__
        self._prepare_weight()
        self.clear_cache()

    def clear_cache(self):
        self._eff_weight = self._weight
        self._eff_max_length = self._max_length

    def _prepare_weight(self):
        rows = list()
        cols = list()

        words = get_words(self.lang)
        charset = get_charset(self.lang)

        self._word2rows = defaultdict(list)
        for row, word in enumerate(words):
            for i, c in enumerate(word.char_seq):
                cid = charset.char2id(c)
                self._word2rows[word].append(len(rows))
                rows.append(row)
                cols.append(len(charset) * i + cid)
        data = np.ones(len(rows))
        # NOTE This is ugly, but it avoids this issue in 0.4.1: https://github.com/pytorch/pytorch/issues/8856.
        weight = torch.sparse.FloatTensor(
            get_tensor([rows, cols], dtype='l', use_cuda=False),
            get_tensor(data, dtype='f', use_cuda=False),
            (len(words), self._max_length * len(charset)))
        self._weight = get_tensor(weight)

    def _sample(self, words):
        all_words = get_words(self.lang)
        word_indices = list()
        old_to_new = np.zeros([len(all_words)], dtype='int64')
        self._eff_id2word = list()
        self._eff_word2id = dict()
        self._eff_max_length = max(map(len, words))
        for w in words:
            word_indices.extend(self._word2rows[w])
            old_to_new[w.idx] = len(self._eff_id2word)
            self._eff_id2word.append(w)
            self._eff_word2id[w] = len(self._eff_word2id)
        old_to_new = get_tensor(old_to_new)
        indices = get_tensor(word_indices, dtype='l')
        old_rows, cols = self._weight._indices()[:, word_indices].unbind(dim=0)
        rows = old_to_new[old_rows]
        data = self._weight._values()[word_indices]
        charset = get_charset(self.lang)
        weight = torch.sparse.FloatTensor(
            torch.stack([rows, cols], dim=0),
            data,
            (len(words), self._eff_max_length * len(charset)))
        self._eff_weight = get_tensor(weight)

    def analyze(self, log_probs, almt_distr, words, lost_lengths):
        self.clear_cache()
        self._sample(words)

        assert self._eff_max_length == len(log_probs)

        tl, nc, bs = log_probs.shape
        charset = get_charset(self.lang)
        assert nc == len(charset)

        # V x bs, or c_s x c_t -> bs x V
        valid_log_probs = self._eff_weight.matmul(log_probs.view(-1, bs)).t()

        sl = almt_distr.shape[-1]
        pos = get_tensor(torch.arange(sl).float(), requires_grad=False)
        mean_pos = (pos * almt_distr).sum(dim=-1)  # bs x tl
        mean_pos = torch.cat([get_zeros(bs, 1, requires_grad=False).fill_(-1.0), mean_pos],
                             dim=-1)
        reg_weight = lost_lengths.float().view(-1, 1) - 1.0 - mean_pos[:, :-1]
        reg_weight.clamp_(0.0, 1.0)
        rel_pos = mean_pos[:, 1:] - mean_pos[:, :-1]  # bs x tl
        rel_pos_diff = rel_pos - 1
        margin = rel_pos_diff != 0
        reg_loss = margin.float() * (rel_pos_diff ** 2)  # bs x tl
        reg_loss = (reg_loss * reg_weight).sum()

        return Map(reg_loss=reg_loss, valid_log_probs=valid_log_probs)
