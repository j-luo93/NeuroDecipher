import os

from torch.utils.data import DataLoader

from arglib import use_arguments_as_properties
from dev_misc import log_pp
from nd.dataset.data_loader import LostKnownDataLoader
from nd.dataset.vocab import build_vocabs
from nd.evaluate.evaluator import Evaluator
from nd.model.decipher import DecipherModelWithFlow
from nd.model.trie import Trie

from .trainer import Trainer


@use_arguments_as_properties('cog_path', 'lost_lang', 'known_lang', 'batch_size')
class Manager:

    def __init__(self):
        # Data
        build_vocabs(self.cog_path, self.lost_lang, self.known_lang)
        self.train_data_loader = LostKnownDataLoader(self.lost_lang, self.known_lang, self.batch_size)
        self.eval_data_loader = LostKnownDataLoader(self.lost_lang, self.known_lang, self.batch_size, training=False)
        # Model
        trie = Trie(self.known_lang)
        self.model = DecipherModelWithFlow(trie)
        log_pp(self.model)
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        # Trainer and evaluator
        self.trainer = Trainer(self.model, self.train_data_loader)
        self.evaluator = Evaluator(self.model, self.eval_data_loader)
        self.evaluator.add_setting(mode='mle', edit=False)
        self.evaluator.add_setting(mode='flow', edit=False)
        self.evaluator.add_setting(mode='flow', edit=True)
        log_pp(self.trainer.tracker.schedule_as_tree())
        log_pp(self.evaluator)

    def train(self):
        self.trainer.train(self.evaluator)
