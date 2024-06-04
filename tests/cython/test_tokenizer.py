# -*- coding: UTF-8 -*-
"""
Created on 03.06.24

:author:     Martin Dočekal
"""
from unittest import TestCase
from decontaminator.cython.tokenizer import whitespace_tokenizer


class TestWhitespaceTokenizer(TestCase):
    def test_whitespace_tokenizer_single_char(self):
        self.assertSequenceEqual([("a", 0, 1), ("b", 2, 3), ("c", 4, 5)], whitespace_tokenizer("a b c"))
        self.assertSequenceEqual([("a", 1, 2), ("b", 3, 4), ("c", 5, 6)], whitespace_tokenizer(" a b c "))
        self.assertSequenceEqual([("a", 1, 2), ("b", 4, 5), ("c", 7, 8)], whitespace_tokenizer(" a  b  c "))

    def test_whitespace_tokenizer_multi_char(self):
        self.assertSequenceEqual([("hello", 0, 5), ("world", 6, 11)], whitespace_tokenizer("hello world"))
        self.assertSequenceEqual([("hello", 1, 6), ("world", 7, 12)], whitespace_tokenizer(" hello world "))
