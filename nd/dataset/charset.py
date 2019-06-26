#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch
import numpy as np

from dev_misc import Map

PAD_ID = 0
SOW_ID = 1
EOW_ID = 2
UNK_ID = 3

PAD = '<PAD>'
SOW = '<SOW>'
EOW = '<EOW>'
UNK = '<UNK>'
START_CHAR = [PAD, SOW, EOW, UNK]

_CHARSETS = dict()


def register_charset(lang):
    global _CHARSETS

    def decorated(cls):
        assert lang not in _CHARSETS
        _CHARSETS[lang] = cls
        return cls

    return decorated


def get_charset(lang):
    '''
    Make sure only one charset is ever created.
    '''
    global _CHARSETS
    cls_or_obj = _CHARSETS[lang]
    if isinstance(cls_or_obj, type):
        _CHARSETS[lang] = cls_or_obj()
    return _CHARSETS[lang]


def _recursive_map(func, lst):
    ret = list()
    for item in lst:
        if isinstance(item, (list, np.ndarray)):
            ret.append(_recursive_map(func, item))
        else:
            ret.append(func(item))
    return ret


class BaseCharset(object):

    _CHARS = u''
    _FEATURES = []

    def __init__(self):
        self._id2char = START_CHAR + list(self.__class__._CHARS)
        self._char2id = dict(zip(self._id2char, range(len(self._id2char))))
        self._feat_dict = {}
        for f in self.features:
            self._feat_dict['char'] = None
            self._feat_dict[f] = False

    def __len__(self):
        return len(self._id2char)

    def char2id(self, char):
        def map_func(c): return self._char2id.get(c, UNK_ID)
        if isinstance(char, str):
            return map_func(char)
        elif isinstance(char, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, char))
            # return np.asarray([np.asarray(list(map(map_func, ch))) for ch in char])
        else:
            raise NotImplementedError

    def id2char(self, id_):
        def map_func(i): return self._id2char[i]
        if isinstance(id_, int):
            return map_func(id_)
        elif isinstance(id_, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, id_))
            # id_.tolist()
            # if id_.ndim == 2:
            #     return np.asarray([np.asarray(list(map(map_func, i))) for i in id_])
            # elif id_.ndim == 3:
            #     return np.asarray([self.id2char(i) for i in id_])
        else:
            raise NotImplementedError

    def get_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy()
        chars = self.id2char(ids)

        def get_2d_tokens(chars):
            tokens = list()
            for char_seq in chars:
                token = ''
                for c in char_seq:
                    if c == EOW:
                        break
                    elif c in START_CHAR:
                        c = '|'
                    token += c
                tokens.append(token)
            return np.asarray(tokens)

        if chars.ndim == 3:
            a, b, _ = chars.shape
            chars = chars.reshape(a * b, -1)
            tokens = get_2d_tokens(chars).reshape(a, b)
        else:
            tokens = get_2d_tokens(chars)
        return tokens

    def process(self, word):
        # How to process chars in word. This function is language-dependent.
        raise NotImplementedError

    @property
    def features(self):
        return self._FEATURES


@register_charset('en')
class EnCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
            else:
                c_lower = c.lower()
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret


@register_charset('es')
class EsCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnñopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('es-ipa')
class EsIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnoprstuwxúɲɾʎʝʧ'
    _FEATURES = ['']


@register_charset('it')
class ItCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzàèéìïòöù'
    _FEATURES = ['capitalization']


@register_charset('it-ipa')
class ItIpaCharSet(BaseCharset):

    _CHARS = u'abdefghijklmnopqrstuvwzŋɔɛɲʃʎʤʧ'
    _FEATURES = ['']


@register_charset('pt')
class PtCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzáâãçéêíóôú'
    _FEATURES = ['capitalization']


@register_charset('pt-ipa')
class PtIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnopstuvzäɐɔɛɨɾʁʃʎʒ'
    _FEATURES = ['']


@register_charset('heb')
class HebCharSet(BaseCharset):

    _CHARS = u'#$&-<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga')
class UgaCharSet(BaseCharset):

    _CHARS = u'#$&*-<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


@register_charset('heb-no_spe')
class HebCharSetNoSpe(BaseCharset):

    _CHARS = u'$&<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga-no_spe')
class UgaCharSetNoSpe(BaseCharset):

    _CHARS = u'$&*<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


@register_charset('el')
class ElCharSet(BaseCharset):

    _CHARS = u'fhyαβγδεζηθικλμνξοπρςστυφχψω'
    _FEATURES = ['']


@register_charset('linb-latin')
class LinbLatinCharSet(BaseCharset):

    _CHARS = u'23adeijkmnopqrstuwz'
    _FEATURES = ['']


@register_charset('minoan')
class MinoanCharSet(BaseCharset):

    _CHARS = u'𐀀𐀁𐀂𐀃𐀄𐀅𐀆𐀇𐀈𐀉𐀊𐀋𐀍𐀏𐀐𐀑𐀒𐀓𐀔𐀕𐀖𐀗𐀘𐀙𐀚𐀛𐀜𐀝𐀞𐀟𐀠𐀡𐀢𐀣𐀤𐀥𐀦𐀨𐀩𐀪𐀫𐀬𐀭𐀮𐀯𐀰𐀱𐀲𐀳𐀴𐀵𐀶𐀷𐀸𐀹𐀺𐀼𐀽𐀿𐁀𐁁𐁂𐁄𐁅𐁆𐁇𐁈𐁉𐁊𐁋'
    _FEATURES = ['']


@register_charset('fr')
class FrCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('lost')
class LostCharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k1')
class K1CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k2')
class K2CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('de')
class DeCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzäöüß'
    _FEATURES = ['capitalization', 'umlaut']
    _UMLAUT = (u'ä', u'ö', u'ü')

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
                if c in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
            else:
                c_lower = c.lower()
                if c_lower in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret
