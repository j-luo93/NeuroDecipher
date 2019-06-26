from collections import defaultdict
from pathlib import Path

import pandas as pd

from dev_misc import cache, counter


class CognateSet:
    '''
    Similar to the concept of synset in wordnet. Each cognate set contains words that are cognates with each other.
    If A is cognate to B1 and B2, then all of them will be stored here.
    Note that this doesn't gurantee that B1 and B2 are semantically similar. The word "bank" might be cognate to
    two totally different words in another language. In other words, while the cognate relation across languages are
    perserved here, nothing can be definitely said wrt the relation within the same language.
    '''
    IDX = 0

    def __init__(self):
        self._data = defaultdict(set)
        self.idx = CognateSet.IDX
        CognateSet.IDX += 1

    def add(self, lang, *words):
        words = [w for w in words if w != '_']
        if words:
            self._data[lang].update(words)  # '_' is a placeholder.

    def is_in(self, word, lang):
        if lang not in self._data:  # Do this to avoid spurious keys for defaultdict.
            return False
        return word in self._data[lang]

    def __contains__(self, lang):
        return lang in self._data

    def items(self):
        return self._data.items()

    def __getitem__(self, lang):
        if not lang in self:
            raise KeyError
        else:
            return self._data[lang]

    def to_df(self):
        data = list()
        for l, s in self._data.items():
            for w in s:
                data.append((w, l, self.idx))
        return pd.DataFrame(data, columns=['word', 'lang', 'idx'])


class CognateDict:
    '''
    A big dictionary that stores every cognate set. This works by storing many CognateSet's, and each
    word is then mapped to one set.
    '''

    def __init__(self, langs):
        self._cs = dict()
        self._langs = langs
        # For each language, map a word to the CognateSet's it is in.
        self._keys = {l: defaultdict(list) for l in langs}

    @property
    def langs(self):
        return self._langs

    def add(self, *cognate_sets):
        for cognate_set in cognate_sets:
            for lang, words in cognate_set.items():
                for w in words:
                    self._keys[lang][w].append(cognate_set.idx)
            self._cs[cognate_set.idx] = cognate_set

    def find(self, word, lang):
        if word not in self._keys[lang]:
            raise KeyError(f'Word {word} in language {lang} not in the dictionary.')

        ret = defaultdict(set)
        for idx in self._keys[lang][word]:
            cs = self._cs[idx]
            for l, words in cs.items():
                ret[l].update(words)
        return ret

    @cache(persist=True, full=False)
    def to_df(self):
        dfs = [cs.to_df() for cs in self._cs.values()]
        return pd.concat(dfs, ignore_index=True)

    def get_wordlist(self, lang):
        df = self.to_df()
        return sorted(set(df[df['lang'] == lang]['word']))


class CognateList:
    '''
    List of cognates, possibly with noncognates as well. This is the main class used for the stream object.
    '''

    def __init__(self, cognate_path, lost_lang, known_lang, max_size=0):
        self.all_langs = set([lost_lang, known_lang])

        cognates = list()
        with Path(cognate_path).open(encoding='utf8') as fcog:
            header_langs = fcog.readline().strip().split("\t")
            for line in counter(fcog, max_size=max_size):
                tokens = line.strip().split('\t')
                cog = CognateSet()
                for l, t in zip(header_langs, tokens):
                    if l in self.all_langs:
                        cog.add(l, *t.split('|'))
                cognates.append(cog)
        self._cognates = cognates
        self._cognate_dict = CognateDict(self.all_langs)
        self._cognate_dict.add(*cognates)

    def get_wordlist(self, lang):
        return self._cognate_dict.get_wordlist(lang)

    def has_cognate(self, w, lang):
        cs = self._cognate_dict.find(w.form, w.lang)
        return lang in cs

    def is_cognate(self, w1, w2):
        cs = self._cognate_dict.find(w1.form, w1.lang)
        if w2.lang in cs:
            return w2.form in cs[w2.lang]
        else:
            return False
