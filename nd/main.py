import os
import random
from pprint import pformat

import numpy as np
import torch

from arglib import parser
from dev_misc import Map, create_logger, log_pp
from nd.config import registry
from nd.train.manager import Manager


def parse_args():
    """Define args here."""

    parser.add_argument('--num_rounds', '-nr', default=3, dtype=int, help='how many rounds of EM')
    parser.add_argument('--num_epochs_per_M_step', '-nm', default=5, dtype=int, help='how many epochs for each M step')
    parser.add_argument('--saved_path', '-sp', dtype=str, help='path to the saved model (and other metadata)')
    parser.add_argument('--learning_rate', '-lr', dtype=float, default=5e-3, help='initial learning rate')
    parser.add_argument('--num_cognates', '-nc', dtype=int, help='how many cognate pairs')
    parser.add_argument('--inc', dtype=int, default=50, help='increment of cognate pairs after each round')
    parser.add_argument('--warm_up_steps', '-wus', dtype=int, default=1,
                        help='how many steps at the start of training without edit distance')
    parser.add_argument('--capacity', default=(1, ), nargs='+', dtype=int,
                        help='capacity for the edges. The first value will be used for E step.')
    parser.add_argument('--save_all', dtype=bool, help='flag to save all models')
    parser.add_argument('--eval_interval', '-ei', default=250, dtype=int, help='evaluate once after this many steps')
    parser.add_argument('--check_interval', '-ci', default=50, dtype=int,
                        help='check and print metrics after this many steps')
    parser.add_argument('--cog_path', '-cp', dtype=str, help='path to the cognate file')
    parser.add_argument('--char_emb_dim', '-ced', default=250, dtype=int, help='dimensionality of character embeddings')
    parser.add_argument('--hidden_size', '-hs', default=250, dtype=int, help='hidden size')
    parser.add_argument('--num_layers', '-nl', default=1, dtype=int, help='number of layers for cipher model')
    parser.add_argument('--dropout', default=0.5, dtype=float, help='dropout rate between layers')
    parser.add_argument('--universal_charset_size', '-ucs', default=50, dtype=int,
                        help='size of the (universal) character inventory')
    parser.add_argument('--lost_lang', '-l', dtype=str, help='lost language code')
    parser.add_argument('--known_lang', '-k', dtype=str, help='known language code')
    parser.add_argument('--norms_or_ratios', '-nor', dtype=float, nargs='+', default=(1.0, 0.2),
                        help='norm or ratio values in control mode')
    parser.add_argument('--control_mode', '-cm', dtype=str, default='relative', help='norm control mode')
    parser.add_argument('--residual', dtype=bool, default=True, help='flag to use residual connection')
    parser.add_argument('--reg_hyper', default=1.0, dtype=float, help='hyperparameter for regularization')
    parser.add_argument('--batch_size', '-bs', dtype=int, help='batch size')
    parser.add_argument('--momentum', default=0.25, dtype=float, help='momentum for flow')
    parser.add_argument('--gpu', '-g', dtype=str, help='which gpu to choose')
    parser.add_argument('--random', dtype=bool, help='random, ignore seed')
    parser.add_argument('--seed', dtype=int, default=1234, help='random seed')
    parser.add_argument('--log_level', default='INFO', dtype=str, help='log level')
    parser.add_argument('--n_similar', dtype=int, help='number of most similar source tokens to keep')
    parser.add_cfg_registry(registry)
    args = Map(**parser.parse_args())

    if args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))  # HACK
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    create_logger(filepath=args.log_dir + '/log', log_level=args.log_level)
    log_pp(pformat(args))


def main():
    parse_args()
    train()


def train():
    manager = Manager()
    manager.train()


if __name__ == "__main__":
    main()
