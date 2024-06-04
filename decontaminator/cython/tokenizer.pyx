# -*- coding: UTF-8 -*-
"""
Created on 03.06.24

:author:     Martin Dočekal
"""


cpdef list[tuple[str,int,int]] whitespace_tokenizer(str text):
    """
    Tokenizes text into words. Words are separated by white characters.
    
    :param text: Text to tokenize.
    :return: List of tuples. Each tuple contains word and its character start and end offset.
    """

    cdef int start=0
    cdef int end=0

    cdef list[tuple[str,int,int]] result=[]

    while end<len(text):
        if text[end].isspace():
            if start!=end:
                result.append((text[start:end],start,end))
            end+=1
            start=end
        else:
            end+=1

    if start!=end:
        result.append((text[start:end],start,end))

    return result
