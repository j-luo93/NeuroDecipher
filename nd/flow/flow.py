import logging

import numpy as np

from arglib import has_properties
from dev_misc import get_tensor, log_this
from nd.dataset.vocab import get_forms, get_words, has_cognate, is_cognate
from nd.magic_tensor.core import MagicTensor


@has_properties('lost_lang', 'known_lang', 'momentum', 'num_cognates')
class Flow:

    def __init__(self, lost_lang, known_lang, momentum, num_cognates):
        super().__init__()
        lost_words = get_words(lost_lang)
        known_words = get_words(known_lang)
        flow = get_tensor(np.zeros([len(lost_words), len(known_words)]))
        self.flow = MagicTensor(flow, lost_words, known_words)
        self._warmed_up = False

    def state_dict(self):
        """Use words as the indices."""

        return {'lost_forms': get_forms(self.lost_lang),
                'known_forms': get_forms(self.known_lang),
                'flow': self.flow}

    def load_state_dict(self, state_dict):
        lost_forms = get_forms(self.lost_lang)
        known_forms = get_forms(self.known_lang)
        saved_lost_forms = state_dict['lost_forms']
        saved_known_forms = state_dict['known_forms']
        assert (lost_forms == saved_lost_forms).all()
        assert (known_forms == saved_known_forms).all()
        self.flow.data.copy_(state_dict['flow'])

    @log_this('IMP')
    def warm_up(self):
        value = self.num_cognates / self.flow.numel()
        self.flow.tensor[:] = value

    @log_this('IMP')
    def update(self, model, data_loader, num_cognates, edit, capacity):
        model.eval()
        entire_batch = data_loader.entire_batch
        model_ret = model(entire_batch, mode='flow', capacity=capacity, num_cognates=num_cognates, edit=edit)
        new_flow = model_ret.flow
        self._check_acc(new_flow)
        self.flow = self.momentum * self.flow + (1.0 - self.momentum) * new_flow

    def _check_acc(self, flow):
        preds = flow.get_best(nonzero=True)
        # Checking lost.
        acc = sum([has_cognate(w, self.known_lang) for w in preds.keys()])
        rate = acc / len(preds)
        logging.imp(f'Accuracy on the lost side {acc} / {len(preds)} = {rate:.3f} ')
        # Checking known.
        acc = sum([has_cognate(w, self.lost_lang) for w in preds.values()])
        rate = acc / len(preds)
        logging.imp(f'Accuracy on the known side {acc} / {len(preds)} = {rate:.3f} ')
        # Checking lost and known.
        acc = sum([is_cognate(w1, w2) for w1, w2 in preds.items()])
        rate = acc / len(preds)
        logging.imp(f'Accuracy for lost-known {acc} / {len(preds)} = {rate:.3f} ')

    def select(self, lost_words, known_words):
        """Take the subtensor, specified by the words."""
        flow = self.flow.select_rows(lost_words).select_cols(known_words)
        flow_k = flow.tensor.sum(dim=0)
        flow_l = flow.tensor.sum(dim=1)
        return {'flow': flow,
                'flow_k': flow_k,
                'flow_l': flow_l,
                'total_flow_k': flow_k.sum(),
                'total_flow_l': flow_l.sum()}
