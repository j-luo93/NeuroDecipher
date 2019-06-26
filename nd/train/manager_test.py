import sys

from dev_misc import TestCase, patch
from nd.dataset.vocab import clear_vocabs
from nd.main import parse_args

from .manager import Manager


class TestManager(TestCase):

    @patch('nd.main.parser.add_cfg_registry')
    def setUp(self, patched_add_cfg):
        clear_vocabs()
        # patched_add_cfg.return_value = None
        sys.argv = 'dummy.py -cp data/test.es-fr-en.toy.cog -l es -k en -nc 3'.split()
        parse_args()

    def test_train(self):
        manager = Manager()
        manager.train()
