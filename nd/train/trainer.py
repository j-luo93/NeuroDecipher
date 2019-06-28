import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from arglib import use_arguments_as_properties
from dev_misc import Map, Metric, Metrics, Tracker, log_this
from nd.flow.flow import Flow


@use_arguments_as_properties('num_rounds', 'num_epochs_per_M_step', 'saved_path', 'learning_rate', 'log_dir', 'num_cognates', 'inc', 'warm_up_steps', 'capacity', 'save_all', 'eval_interval', 'reg_hyper', 'lost_lang', 'known_lang', 'momentum', 'check_interval')
class Trainer:

    def __init__(self, model, train_data_loader, flow_data_loader):
        self.tracker = Tracker('decipher')
        stage = self.tracker.add_stage('round', self.num_rounds)
        stage.add_stage('E step')
        stage.add_stage('M step', self.num_epochs_per_M_step)
        self.tracker.fix_schedule()
        self.model = model
        self.train_data_loader = train_data_loader
        self.flow_data_loader = flow_data_loader
        self._init_optimizer()
        self.flow = Flow(self.lost_lang, self.known_lang, self.momentum, self.num_cognates)
        self.tb_writer = SummaryWriter(self.log_dir)

    @log_this('IMP')
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @log_this('IMP')
    def _init_params(self):
        for name, param in self._get_trainable_params(names=True):
            if param.ndimension() == 2:
                nn.init.xavier_uniform_(param)
                logging.debug('initialized %s' % name)
            else:
                if 'bias_ih' in name or 'bias_hh' in name:
                    size = param.size(0)
                    ind = torch.arange(size // 4, size // 2).long()
                    param.data[ind] = 1.0
                    logging.debug(f'Forget gate bias initialized to 1.0 for {name}')

    def _get_trainable_params(self, names=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if names:
                    yield name, param
                else:
                    yield param

    def load(self):
        ckpt = torch.load(self.saved_path)

        def try_load(name):
            src = ckpt[name]
            dest = getattr(self, name)
            try:
                dest.load_state_dict(src)
            except RuntimeError as e:
                logging.error(e)

        try_load('model')
        try_load('optimizer')
        try_load('tracker')
        try_load('flow')
        logging.imp(f'Loaded saved states from {self.saved_path}')

    def save(self, suffix='latest'):
        if self.log_dir:
            logging.info('Saving to %s' % self.log_dir)
            ckpt = {'model': self.model.state_dict(), 'optim': self.optimizer.state_dict(),
                    'tracker': self.tracker.state_dict(), 'flow': self.flow.state_dict()}
            torch.save(ckpt, self.log_dir + '/saved.%s' % suffix)
            logging.info('Finished saving decipher trainer')

    def train(self, evaluator):
        if self.saved_path:
            self.load()
        else:
            self._init_params()
        while not self.tracker.finished:
            self._train_loop(evaluator)

    @property
    def round_num(self):
        return self.tracker.get('round') + 1

    @property
    def stage(self):
        return self.tracker.current_stage

    def _train_loop(self, evaluator):
        if self.stage.name == 'E step':
            self._do_E_step()
        elif self.stage.name == 'M step':
            self._do_M_step(evaluator)
        else:
            raise RuntimeError(f'Not recognized stage name {self.stage.name}')
        self.tracker.update()

    def _do_E_step(self):
        num_cognates = min((self.round_num - 1) * self.inc, self.num_cognates)
        edit = self.round_num > self.warm_up_steps
        warm_up = self.stage.step == 0 and self.round_num == 1
        self._E_step_kernel(num_cognates, warm_up, edit)

    @log_this('IMP', arg_list=['num_cognates', 'warm_up', 'edit'])
    def _E_step_kernel(self, num_cognates, warm_up, edit):
        if warm_up:
            self.flow.warm_up()
        else:
            with torch.no_grad():
                self.flow.update(self.model, self.flow_data_loader, num_cognates, edit, self.capacity[0])
                self._init_params()
                self._init_optimizer()

    @property
    def epoch(self):
        return (self.round_num - 1) * self.num_epochs_per_M_step + self.stage.step + 1

    def _prepare_flow(self, batch):
        """Add flow-related info to the batch."""
        flow_info = self.flow.select(batch.lost.words, batch.known.words)
        batch.update(flow_info)

    def _do_M_step(self, evaluator):
        self._M_step_kernel()
        self._do_post_M_step(evaluator)

    def _M_step_kernel(self):
        for batch in self.train_data_loader:
            self._M_step_kernel_loop(batch)

    def _M_step_kernel_loop(self, batch, update=True):
        self._prepare_flow(batch)
        return self._do_M_step_batch(batch, update=update)

    def _do_post_M_step(self, evaluator):
        if self.epoch % self.eval_interval == 0:
            self._do_eval(evaluator)
        if self.epoch % self.check_interval == 0:
            self._do_check()

    def _do_eval(self, evaluator):
        num_cognates = min(self.round_num * self.inc, self.num_cognates)
        eval_scores = evaluator.evaluate(self.epoch, num_cognates)
        # Tensorboard
        for setting, score in eval_scores.items():
            self.tb_writer.add_scalar(setting, score, global_step=self.epoch)
        self.tb_writer.flush()
        # Save
        self.save()
        if self.save_all:
            self.save(suffix=self.epoch)

    def _do_check(self):
        self.tracker.check_metrics(self.epoch)
        self.tb_writer.add_scalar('loss', self.tracker.metrics.loss.mean, global_step=self.epoch)
        self.tracker.clear_metrics()

    def _do_M_step_batch(self, batch, update=True):
        self.model.train()
        self.optimizer.zero_grad()
        # Run it.
        model_ret = self.model(batch)
        if update:
            self._do_M_step_batch_update(model_ret, batch)
        return model_ret

    def _do_M_step_batch_update(self, model_ret, batch):
        # Get the metrics.
        num_samples = Metric('num_samples', batch.num_samples, 0, report_mean=False)
        metrics = self._analyze_model_return(model_ret, batch)
        # Compute gradients and backprop.
        metrics.loss.mean.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._get_trainable_params(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm * num_samples.total, num_samples.total)
        self.optimizer.step()
        # Update metrics.
        metrics += Metrics(num_samples, grad_norm)
        self.tracker.update_metrics(metrics)

    def _analyze_model_return(self, model_ret, batch):
        reg_loss = Metric('reg_loss', model_ret.reg_loss, batch.total_flow_k)
        # NOTE This means we are conditioning on one specific flow.
        nll_losses = torch.logsumexp((model_ret.valid_log_probs + (batch.flow + 1e-8).log()).tensor, dim=0)
        nll_losses = nll_losses * batch.flow_k
        nll_loss = Metric('nll_loss', -nll_losses.sum(), batch.total_flow_k)
        loss = Metric('loss', self.reg_hyper * reg_loss.mean + nll_loss.mean, 1)
        return Metrics(loss, nll_loss, reg_loss)
