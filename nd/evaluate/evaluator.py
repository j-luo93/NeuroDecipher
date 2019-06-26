from dataclasses import dataclass

from prettytable import PrettyTable as pt

from arglib import use_arguments_as_properties
from dev_misc import log_pp
from nd.dataset.vocab import is_cognate


@dataclass(frozen=True)
class EvalSetting:
    lost: str
    known: str
    lost_size: int
    known_size: int
    mode: str
    edit: bool
    capacity: int

    def __str__(self):
        return f'lost_{self.lost}__known_{self.known}__mode_{self.mode}__edit_{self.edit}__capacity_{self.capacity}'


@use_arguments_as_properties('lost_lang', 'known_lang', 'capacity', 'num_cognates')
class Evaluator:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self._settings = list()

    def add_setting(self, mode=None, edit=None):
        assert mode in ['mle', 'flow']
        assert edit in [True, False]

        lost_size = self.data_loader.size(self.lost_lang)
        known_size = self.data_loader.size(self.known_lang)
        if mode == 'mle':
            self._settings.append(
                EvalSetting(self.lost_lang, self.known_lang, lost_size, known_size, mode, None, None))
        else:
            for c in self.capacity:
                self._settings.append(
                    EvalSetting(self.lost_lang, self.known_lang, lost_size, known_size, mode, edit, c))

    def __str__(self):
        table = pt()
        table.field_names = 'lost', 'known', 'lost_size', 'known_size', 'mode', 'edit', 'capacity'
        for s in self._settings:
            table.add_row([getattr(s, field) for field in table.field_names])
        table.align = 'l'
        return str(table)

    def evaluate(self, epoch, num_cognates):
        self.model.eval()
        table = pt()
        table.field_names = 'lost', 'known', 'mode', 'edit', 'capacity', 'score'

        eval_scores = dict()
        for s in self._settings:
            batch = self.data_loader.entire_batch
            model_ret = self.model(batch, mode=s.mode, num_cognates=num_cognates, edit=s.edit, capacity=s.capacity)
            # Magic tensor to the rescue!
            almt = model_ret.valid_log_probs if s.mode == 'mle' else model_ret.flow
            preds = almt.get_best()
            acc = self._evaluate_one_setting(preds)
            score = acc / len(preds)
            fmt_score = f'{acc}/{len(preds)}={score:.3f}'
            table.add_row([getattr(s, field) for field in table.field_names[:-1]] + [fmt_score])
            eval_scores[str(s)] = score

        table.align = 'l'
        table.title = f'Epoch: {epoch}'
        log_pp(table)
        return eval_scores

    def _evaluate_one_setting(self, preds):
        acc = 0
        for lost, known in preds.items():
            if is_cognate(lost, known):
                acc += 1
        return acc
