from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arglib import has_properties
from dev_misc import cache

from .charset import EOW, get_charset
from .cognate import CognateList

_VOCABS = dict()
_COG_LIST = None


def get_vocab(lang):
    return _VOCABS[lang]


def get_vocab_size(lang):
    return len(get_vocab(lang))


def get_words(lang):
    return get_vocab(lang).words


def get_forms(lang):
    return get_vocab(lang).forms


def is_cognate(w1, w2):
    global _COG_LIST
    return _COG_LIST.is_cognate(w1, w2)


def has_cognate(w, lang):
    global _COG_LIST
    return _COG_LIST.has_cognate(w, lang)


def clear_vocabs():
    global _COG_LIST
    global _VOCABS
    _COG_LIST = None
    _VOCABS = dict()


def build_vocabs(path, lost_lang, known_lang, max_size=0):
    global _COG_LIST

    path = Path(path)
    if path.suffix != '.cog':
        raise ValueError(f'Do not recognize this file format {path.suffix}')

    assert _COG_LIST is None
    cog_list = CognateList(path, lost_lang, known_lang, max_size=max_size)

    for lang in [lost_lang, known_lang]:
        if lang in _VOCABS:
            raise ValueError(f'There already is a vocab for {lang}')
        _VOCABS[lang] = Vocab(cog_list.get_wordlist(lang), lang)
    _COG_LIST = cog_list


@dataclass(frozen=True, order=True)
class Word:
    __slots__ = ['lang', 'form', 'idx']
    lang: str
    form: str
    idx: int

    @property
    @cache(persist=True)
    def char_seq(self):
        return np.asarray(list(self.form) + [EOW])

    @property
    @cache(persist=True)
    def id_seq(self):
        return get_charset(self.lang).char2id(self.char_seq)

    def __len__(self):
        return len(self.form) + 1


@has_properties('lang')
class Vocab:

    def __init__(self, wordlist, lang):
        assert len(wordlist) == len(set(wordlist))  # Make sure they are all unique.
        self._build(wordlist)

    def _build(self, wordlist):
        self._id2word = list()
        self._word2id = dict()
        for w in wordlist:
            w = Word(self.lang, w, len(self._id2word))
            self._id2word.append(w)
            self._word2id[w.form] = len(self._word2id)

    def __len__(self):
        return len(self._id2word)

    def word2id(self, words):
        if isinstance(words, str):
            return self._word2id(w)
        func = np.vectorize(lambda w: self._word2id[w])
        word_ids = func(words)
        return word_ids

    @property
    @cache(persist=True)
    def words(self):
        return np.asarray(self._id2word)

    @property
    @cache(persist=True)
    def forms(self):
        return np.asarray([word.form for word in self.words])

    def cognate_to(self, lang):
        global _COG_LIST
        return np.asarray([w for w in self.words if _COG_LIST.has_cognate(w, lang)])