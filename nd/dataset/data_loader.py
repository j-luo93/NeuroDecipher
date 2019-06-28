import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from prettytable import PrettyTable as pt

from arglib import has_properties
from dev_misc import Map, cache, get_tensor, sort_all

from .charset import EOW, get_charset
from .vocab import Word, get_vocab, get_words


def pad_to_dense(a, dtype='f'):
    '''
    Modified from https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size.
    '''
    assert dtype in ['f', 'l']
    dtype = 'float32' if dtype == 'f' else 'int64'
    maxlen = max(map(len, a))
    ret = np.zeros((len(a), maxlen), dtype=dtype)
    for i, row in enumerate(a):
        ret[i, :len(row)] += row
    return ret


@has_properties('lang')
class WordlistDataset(Dataset):
    """This is for one language."""

    def __init__(self, words, lang):
        assert isinstance(words[0], Word)
        self._words = words

    def __len__(self):
        return len(self._words)

    @cache(persist=True, full=True)
    @cache(persist=True, full=True)
    def __getitem__(self, idx):
        word = self._words[idx]
        return Map(word=word, form=word.form, lang=self.lang, char_seq=word.char_seq, id_seq=word.id_seq)

    @property
    @cache(persist=True)
    def entire_batch(self):
        return collate_fn([self[i] for i in range(len(self))])


class VocabDataset(WordlistDataset):

    def __init__(self, lang):
        super().__init__(get_words(lang), lang)


def _get_item(key, batch):
    return np.asarray([record[key] for record in batch])


def collate_fn(batch):
    words = _get_item('word', batch)
    forms = _get_item('form', batch)
    char_seqs = _get_item('char_seq', batch)
    id_seqs = _get_item('id_seq', batch)
    lengths, words, forms, char_seqs, id_seqs = sort_all(words, forms, char_seqs, id_seqs)
    lengths = get_tensor(lengths, dtype='l')
    # Trim the id_seqs.
    max_len = max(lengths).item()
    id_seqs = pad_to_dense(id_seqs, dtype='l')
    id_seqs = get_tensor(id_seqs[:, :max_len])

    lang = batch[0].lang
    return Map(
        words=words, forms=forms, char_seqs=char_seqs, id_seqs=id_seqs, lengths=lengths, lang=lang)


def _prepare_stats(name, *rows):
    table = pt()
    table.field_names = 'lang', 'size'
    for row in rows:
        table.add_row(row)
    table.align = 'l'
    table.title = name
    return table


@has_properties('lost_lang', 'known_lang', 'cognate_only')
class LostKnownDataLoader(DataLoader):

    def __init__(self, lost_lang, known_lang, batch_size, cognate_only=False):
        self.datasets = dict()
        if not cognate_only:
            self.datasets[self.lost_lang] = VocabDataset(lost_lang)
        else:
            lost_words = get_vocab(lost_lang).cognate_to(known_lang)
            self.datasets[self.lost_lang] = WordlistDataset(lost_words, lost_lang)
        self.datasets[self.known_lang] = VocabDataset(known_lang)

        if batch_size:
            shuffle = True
        else:
            batch_size = len(self.datasets[self.known_lang])
            shuffle = False

        super().__init__(self.datasets[self.known_lang], batch_size=batch_size,
                         shuffle=shuffle, collate_fn=collate_fn)

    def __iter__(self):
        for known_batch in super().__iter__():
            lost_batch = self.datasets[self.lost_lang].entire_batch
            num_samples = len(known_batch.words)
            yield Map(lost=lost_batch, known=known_batch, num_samples=num_samples)

    @property
    @cache(persist=True)
    def entire_batch(self):
        """Return the entire dataset as a batch. This shold have a persistent order among the words."""
        return Map(known=self.datasets[self.known_lang].entire_batch, lost=self.datasets[self.lost_lang].entire_batch)

    def size(self, lang):
        return len(self.datasets[lang])

    def stats(self, name):
        row1 = [self.lost_lang, len(self.datasets[self.lost_lang])]
        row2 = [self.known_lang, len(self.datasets[self.known_lang])]
        table = _prepare_stats(name, row1, row2)
        return table

