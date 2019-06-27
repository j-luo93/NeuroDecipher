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

    model_cls = DecipherModelWithFlow
    trainer_cls = Trainer

    def __init__(self):
        self._get_data()
        self._get_model()
        self._get_trainer_and_evaluator()

    def _get_trainer_and_evaluator(self):
        self.trainer = type(self).trainer_cls(self.model, self.train_data_loader, self.flow_data_loader)
        self.evaluator = Evaluator(self.model, self.eval_data_loader)
        self.evaluator.add_setting(mode='mle', edit=False)
        self.evaluator.add_setting(mode='flow', edit=False)
        self.evaluator.add_setting(mode='flow', edit=True)
        log_pp(self.trainer.tracker.schedule_as_tree())
        log_pp(self.evaluator)

    def _get_data(self):
        self._get_data_loaders()
        self._show_data()

    def _get_data_loaders(self):
        build_vocabs(self.cog_path, self.lost_lang, self.known_lang)
        self.train_data_loader = LostKnownDataLoader(self.lost_lang, self.known_lang, self.batch_size, cognate_only=False)
        self.eval_data_loader = LostKnownDataLoader(self.lost_lang, self.known_lang, self.batch_size, cognate_only=True)
        self.flow_data_loader = self.train_data_loader # NOTE The flow instance shares its entire_batch property with train_data_loader.

    def _show_data(self):
        log_pp(self.train_data_loader.stats('train'))
        log_pp(self.eval_data_loader.stats('eval'))

    def _get_model(self):
        trie = Trie(self.known_lang)
        self.model = type(self).model_cls(trie)
        log_pp(self.model)
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()

    def train(self):
        self.trainer.train(self.evaluator)
