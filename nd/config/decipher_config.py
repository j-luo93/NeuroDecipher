from typing import Tuple

from . import registry

register = registry.register


@register
class UgaHebSmallNoSpe:
    lost_lang: str = 'uga-no_spe'
    known_lang: str = 'heb-no_spe'
    cog_path: str = 'data/uga-heb.small.no_spe.cog'
    num_cognates: int = 221
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5


@register
class IberCoins:
    lost_lang: str = 'iber'
    known_lang: str = 'iber-latin'
    cog_path: str = 'data/iber-latin.coin.cog'
    num_cognates: int = 16
    num_epochs_per_M_step: int = 300
    eval_interval: int = 50
    check_interval: int = 50
    num_rounds: int = 30
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 10
    momentum: float = 0.95


@register
class IberProtoBasque(IberCoins):
    lost_lang: str = 'p_eu'
    known_lang: str = 'iber'
    num_cognates: int = 100
    capacity: int = 1
