import types
from functools import wraps

import torch

from arglib import has_properties
from dev_misc import log_this
from nd.dataset.vocab import Word

_SAFE_METHODS = {'numel', '__str__', 'data', 'shape', 'unsqueeze'}
_SAFE_METHODS_WITH_WRAPPER = {'log'}


@has_properties('tensor', 'row_words', 'col_words')
class MagicTensor:

    def __init__(self, tensor, row_words, col_words):
        assert tensor.ndimension() == 2
        assert isinstance(row_words[0], Word)
        assert isinstance(col_words[0], Word)
        assert len(row_words) == tensor.shape[0]
        assert len(col_words) == tensor.shape[1]

        def freeze(words):
            if isinstance(words, tuple):
                return words
            else:
                return tuple(words)

        # NOTE Freeze these to speed up checking procedure.
        self._row_words = freeze(row_words)
        self._col_words = freeze(col_words)
        # Fast indexing.
        self._word2row = {w: i for i, w in enumerate(self._row_words)}
        self._word2col = {w: i for i, w in enumerate(self._col_words)}

    def _check_value(self, other):
        if isinstance(other, MagicTensor):
            try:
                assert self.row_words == other.row_words
                assert self.col_words == other.col_words
                return other.tensor
            except AssertionError:
                self._permute(other)
                return other.tensor
        elif isinstance(other, (float, int)):
            return other
        else:
            raise NotImplementedError(f'Type {type(other)} not supported.')

    @log_this()
    def _permute(self, other):
        # Have to re-index the my own tensor. But make sure that the set of words are identical first.
        assert set(self.row_words) == set(other.row_words)
        assert set(self.col_words) == set(other.col_words)
        my_rows = [self._word2row[w] for w in other.row_words]
        my_cols = [self._word2col[w] for w in other.col_words]
        self._row_words = other.row_words
        self._col_words = other.col_words
        self._word2row = other._word2row
        self._word2col = other._word2col
        self._tensor = self._tensor[my_rows][:, my_cols]

    def __repr__(self):
        return f'MagicTensor({self.tensor!r})'

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            try:
                tensor = super().__getattribute__('_tensor')
            except AttributeError:
                raise

            orig = getattr(tensor, attr)
            if attr in _SAFE_METHODS:
                setattr(self, attr, orig)
            elif attr in _SAFE_METHODS_WITH_WRAPPER:

                @wraps(orig)
                def wrapper(self, *args, **kwargs):
                    ret = orig(*args, **kwargs)
                    return MagicTensor(ret, self.row_words, self.col_words)

                setattr(self, attr, types.MethodType(wrapper, self))
            else:
                raise AttributeError
            return getattr(self, attr)

    def __add__(self, other):
        value = self._check_value(other)
        return MagicTensor(self.tensor + value, self.row_words, self.col_words)

    def __mul__(self, other):
        value = self._check_value(other)
        return MagicTensor(self.tensor * value, self.row_words, self.col_words)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, key):
        new_tensor = self.tensor[key]
        if isinstance(key, tuple):
            assert len(key) == 2
            s0, s1 = key
            return MagicTensor(new_tensor, self.row_words[s0], self.col_words[s1])
        else:
            return MagicTensor(new_tensor, self.row_words[key], self.col_words)

    def select_rows(self, words):
        ids = [self._word2row[w] for w in words]
        return MagicTensor(self.tensor[ids], words, self.col_words)

    def select_cols(self, words):
        ids = [self._word2col[w] for w in words]
        return MagicTensor(self.tensor[:, ids], self.row_words, words)

    def get_best(self, nonzero=False):
        ret = dict()
        best_idx = self.tensor.max(dim=-1)[1].cpu().numpy()
        for lost_idx, known_idx in enumerate(best_idx):
            if not nonzero or self.tensor[lost_idx, known_idx].item() > 0:
                lost = self.row_words[lost_idx]
                known = self.col_words[known_idx]
                assert lost not in ret
                ret[lost] = known
        return ret
