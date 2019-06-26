#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from . import charset


class TestEnCharSet(unittest.TestCase):
  def test_char2id(self):
    en_charset = charset.EnCharSet()
    self.assertEqual(en_charset.char2id('<PAD>'), 0)
    self.assertEqual(en_charset.char2id('w'), 26)
    self.assertEqual(en_charset.char2id('Ö'), en_charset.char2id('<UNK>'))

  def test_id2char(self):
    en_charset = charset.EnCharSet()
    self.assertEqual(en_charset.id2char(0), '<PAD>')
    self.assertEqual(en_charset.id2char(26), 'w')
    self.assertEqual(en_charset.id2char(3), '<UNK>')

  def test_process(self):
    en_charset = charset.EnCharSet()
    features = en_charset.process('Hello')
    self.assertEqual(features[0]['char'], 'h')
    self.assertEqual(features[0]['capitalization'], True)
    self.assertEqual(features[1]['char'], 'e')
    self.assertEqual(features[1]['capitalization'], False)


class TestDeCharSet(unittest.TestCase):
  def test_char2id(self):
    de_charset = charset.DeCharSet()
    self.assertEqual(de_charset.char2id('<PAD>'), 0)
    self.assertEqual(de_charset.char2id(u'ß'), 33)
    self.assertEqual(de_charset.char2id(u'Ö'), de_charset.char2id('<UNK>'))

  def test_id2char(self):
    de_charset = charset.DeCharSet()
    self.assertEqual(de_charset.id2char(0), '<PAD>')
    self.assertEqual(de_charset.id2char(26), 'w')
    self.assertEqual(de_charset.id2char(3), '<UNK>')

  def test_process(self):
    de_charset = charset.DeCharSet()
    features = de_charset.process(u'Öffnung')
    self.assertEqual(features[0]['char'], u'ö')
    self.assertEqual(features[0]['capitalization'], True)
    self.assertEqual(features[0]['umlaut'], True)
    self.assertEqual(features[3]['capitalization'], False)
    self.assertEqual(features[3]['umlaut'], False)


if __name__ == '__main__':
    unittest.main()
