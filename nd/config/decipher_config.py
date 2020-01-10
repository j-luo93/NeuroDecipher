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
class GotNhdIpaSmall(UgaHebSmallNoSpe):
    lost_lang: str = 'got'
    known_lang: str = 'nhd_ipa'
    cog_path: str = 'data/got-nhd_ipa.cog'
    num_cognates: int = 369

@register
class GotNhdSmall(GotNhdIpaSmall):
    known_lang: str = 'nhd'
    cog_path: str = 'data/got-nhd.cog'

@register
class GotNhdNoPrefixSmall(GotNhdSmall):
    num_cognates: int = 365
    cog_path: str = 'data/got-nhd_no_pref.cog'

@register
class GotLemmaNhdNoPrefixSmall(GotNhdSmall):
    num_cognates: int = 213
    cog_path: str = 'data/got_lemma-nhd_no_pref.cog'

@register
class GotLemmaNhdNoPrefix(GotNhdSmall):
    num_cognates: int = 1196
    cog_path: str = 'data/got_lemma-nhd_no_pref.all.cog'

@register
class GotLemmaGermNoPrefix(GotNhdSmall):
    num_cognates: int = 1908
    known_lang: str = 'germ'
    cog_path: str = 'data/got_lemma-germ_no_pref.all.cog'

@register
class GotLemmaAENoPrefix(GotNhdSmall):
    num_cognates: int = 1443
    known_lang: str = 'ae'
    cog_path: str = 'data/got_lemma-ae_no_pref.all.cog'

@register
class GotNhdNDCog(GotNhdSmall):
    num_cognates: int = 1193
    cog_path: str = 'data/got-nhd.nd.cog'

@register
class GotGermNDCog(GotNhdSmall):
    num_cognates: int = 1908
    known_lang: str = 'germ'
    cog_path: str = 'data/got-germ.nd.cog'

@register
class GotAENDCog(GotNhdSmall):
    num_cognates: int = 1439
    known_lang: str = 'ae'
    cog_path: str = 'data/got-ae.nd.cog'

