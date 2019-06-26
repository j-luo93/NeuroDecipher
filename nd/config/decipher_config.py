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
