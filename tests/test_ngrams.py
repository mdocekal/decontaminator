# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

:author:     Martin Dočekal
"""
from unittest import TestCase

from decontaminator.ngrams import ngrams


class Test(TestCase):
    def test_ngrams(self):
        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 1)),
            [['Ahoj,',], ['jak',], ['se',], ['máš?',]]
        )

        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 2)),
            [['Ahoj,', 'jak'], ['jak', 'se'], ['se', 'máš?']]
        )

        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 3)),
            [['Ahoj,', 'jak', 'se'], ['jak', 'se', 'máš?']]
        )

        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 4)),
            [['Ahoj,', 'jak', 'se', 'máš?']]
        )

        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 5)),
            []
        )

    def test_short(self):
        self.assertEqual(
            list(ngrams("Ahoj, jak se máš?".split(), 5, allow_shorter=True)),
            [['Ahoj,', 'jak', 'se', 'máš?']]
        )
