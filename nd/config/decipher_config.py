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


@register
class GothicPgm(UgaHebSmallNoSpe):

    lost_lang: str = 'got'
    known_lang: str = 'pgm'
    cog_path: str = 'data/got-pgm.cog'
    num_cognates: int = 2285
    num_epochs_per_M_step: int = 250
    eval_interval: int = 50
    check_interval: int = 50
    num_rounds: int = 5
    inc: int = 500


@register
class GothicNon(GothicPgm):
    known_lang: str = 'non'
    cog_path: str = 'data/got-non.cog'
    num_cognates: int = 1787


@register
class GothicAng(GothicPgm):
    known_lang: str = 'ang'
    cog_path: str = 'data/got-ang.cog'
    num_cognates: int = 2278

@register
class GothicPgmIpa(GothicPgm):

    known_lang: str = 'pgm-ipa'
    cog_path: str = 'data/got-pgm_ipa.cog'


@register
class GothicNonIpa(GothicNon):
    known_lang: str = 'non-ipa'
    cog_path: str = 'data/got-non_ipa.cog'


@register
class GothicAngIpa(GothicAng):
    known_lang: str = 'ang-ipa'
    cog_path: str = 'data/got-ang_ipa.cog'
