# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

:author:     Martin Dočekal
"""
from unittest import TestCase

from decontaminator.normalize import normalize_text


class Test(TestCase):
    def test_normalize_text(self):
        self.assertEqual("ahoj  jak se máš ", normalize_text("Ahoj, jak se máš?"))
        self.assertEqual("how are you", normalize_text("how are you"))
