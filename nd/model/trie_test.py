import torch

from dev_misc import Mock, TestCase, patch
from nd.dataset.vocab import Word

from .trie import Trie


class TestTrie(TestCase):

    @patch('nd.model.trie.get_words')
    def setUp(self, patched_get_words):
        go = Word('en', 'go', 0)
        good = Word('en', 'good', 1)
        goods = Word('en', 'goods', 2)
        self.words = [go, good, goods]

        patched_get_words.side_effect = lambda lang: self.words
        self.sampled_words = [good, go]
        self.trie = Trie('en')

    def _get_probs(self, *shape):
        logits = torch.randn(shape)
        return torch.log_softmax(logits, dim=-1).exp()

    def test_analyze(self):
        log_probs = self._get_probs(6, 30, 64)
        almt_distr = self._get_probs(64, 6, 7)
        ret = self.trie.analyze(log_probs, almt_distr, self.words,
                                torch.LongTensor([7] * 32 + [6] * 16 + [2] * 16))
        self.assertHasShape(ret.valid_log_probs, (64, 3))

    def test_sample(self):
        log_probs = self._get_probs(5, 30, 64)
        almt_distr = self._get_probs(64, 5, 7)
        ret = self.trie.analyze(log_probs, almt_distr, self.sampled_words,
                                torch.LongTensor([7] * 32 + [6] * 16 + [2] * 16))
        self.assertHasShape(ret.valid_log_probs, (64, 2))
