# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

:author:     Martin Dočekal
"""
import string

remove_punctuation_table = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_text(text: str) -> str:
    """
    Normalizes text, converts to lowercase and removes punctuation.

    :param text: Text to be normalized.
    :return: Normalized text.
    """
    return text.lower().translate(remove_punctuation_table)
