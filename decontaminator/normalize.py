# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

:author:     Martin Dočekal
"""
import string
from typing import Optional

remove_punctuation_table = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_text(text: str, sub_table: Optional[dict[int, int]] = None) -> str:
    """
    Normalizes text, converts to lowercase and removes punctuation.

    :param text: Text to be normalized.
    :param sub_table: Table for substitution of characters. By default, it removes punctuation.
    :return: Normalized text.
    """
    if sub_table is None:
        sub_table = remove_punctuation_table
    return text.lower().translate(sub_table)
