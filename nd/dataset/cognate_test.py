from dev_misc import TestCase

from .cognate import CognateSet, CognateDict

class TestCognateSet(TestCase):
    
    def test_cognate_set(self):
        cog_set = CognateSet()
        cog_set.add('en', 'A1')
        cog_set.add('es', 'B1')
        cog_set.add('en', 'A2', 'A3')
        cog_set.add('de', 'C1', 'C2')
        self.assertTrue(cog_set.is_in('A1', 'en'))
        self.assertTrue(cog_set.is_in('A2', 'en'))
        self.assertTrue(cog_set.is_in('B1', 'es'))
        self.assertTrue(cog_set.is_in('C2', 'de'))
        self.assertFalse(cog_set.is_in('A4', 'en'))
        self.assertFalse(cog_set.is_in('A4', 'es'))
        self.assertTrue('en' in cog_set)
        self.assertTrue('es' in cog_set)
        self.assertFalse('hi' in cog_set)

    def test_placeholder(self):
        cog_set = CognateSet()
        cog_set.add('en', '_')
        cog_set.add('fr', 'A1')
        cog_set.add('es', 'B1')
        self.assertTrue('es' in cog_set)
        self.assertTrue('fr' in cog_set)
        self.assertFalse('en' in cog_set)

class TestCognateDict(TestCase):
    
    def _get_dummy_cs(self, **kwargs):
        cog_set = CognateSet()
        for l, ws in kwargs.items():
            cog_set.add(l, *ws)
        return cog_set
    
    def test_cognate_dict(self):
        cs0 = self._get_dummy_cs(**{'en': ['A1', 'A2'], 'es': ['B1', 'B2']})
        cs1 = self._get_dummy_cs(**{'en': ['A2'], 'es': ['B3']})
        cs2 = self._get_dummy_cs(**{'en': ['A3'], 'es': ['B1']})
        cd = CognateDict(['en', 'es'])
        cd.add(cs0, cs1, cs2)
        self.assertSetEqual(cd.find('A1', 'en')['es'], {'B1', 'B2'})
        self.assertSetEqual(cd.find('A2', 'en')['es'], {'B1', 'B2', 'B3'})
        self.assertSetEqual(cd.find('A3', 'en')['es'], {'B1'})
        self.assertSetEqual(cd.find('B1', 'es')['en'], {'A1', 'A2', 'A3'})
        self.assertSetEqual(cd.find('B2', 'es')['en'], {'A1', 'A2'})
        self.assertSetEqual(cd.find('B3', 'es')['en'], {'A2'})
        
