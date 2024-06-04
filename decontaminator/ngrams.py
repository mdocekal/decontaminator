# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

Module for working with n-grams.

:author:     Martin Dočekal
"""
from typing import Generator, Sequence, TypeVar

T = TypeVar('T')


def ngrams(tokens: Sequence[T], n: int, allow_shorter: bool = False) -> Generator[list[T], None, None]:
    """
    Splits text into n-grams.

    :param tokens: Tokens that will be used for n-grams.
    :param n: Size of n-gram.
    :param allow_shorter: If the sequence is shorter than n, it will generate one n-gram that will be the same as the sequence.
    :return: Generate n-grams.
    """

    if len(tokens) < n:
        if allow_shorter:
            yield list(tokens)
        return

    for i in range(len(tokens)-n+1):
        yield list(tokens[i:i+n])

