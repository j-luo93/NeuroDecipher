import numpy as np

from dev_misc import TestCase

from .charset import EOW
from .data_loader import VocabDataset, WordlistDataset
from .vocab import build_vocabs, clear_vocabs, get_vocab


class TestVocabDataset(TestCase):

    def setUp(self):
        clear_vocabs()
        build_vocabs('data/test.es-fr-en.toy.cog', 'es', 'en')

    def test_basic(self):
        dataset = VocabDataset('es')
        ans = dataset[0].char_seq
        self.assertListEqual(ans.tolist(), np.asarray(['e', 's', '1', EOW]).tolist())


class TestWordlistDataset(TestCase):

    def setUp(self):
        clear_vocabs()
        build_vocabs('data/test.es-fr-en.toy.cog', 'es', 'en')

    def test_basic(self):
        vocab = get_vocab('es')
        dataset = WordlistDataset(vocab.words[1:], 'es')
        ans = dataset[0].char_seq
        self.assertListEqual(ans.tolist(), np.asarray(['e', 's', '2', EOW]).tolist())
