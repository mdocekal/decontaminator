# -*- coding: UTF-8 -*-
"""
Created on 13.09.24

:author:     Martin Dočekal
"""
from typing import Generator, Union, Sequence
from unittest import TestCase

from decontaminator.reader import Reader


class MockReader(Reader):

    def __init__(self, samples, format_str=None):
        super().__init__(format_str)
        self.samples = samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self) -> Generator[Union[str, Sequence[str]], None, None]:

        for sample in self.samples:
            if self.multi_format:
                yield [j.render(sample) for j in self.jinja]
            else:
                yield self.jinja.render(sample)


class TestReader(TestCase):
    def setUp(self):
        self.samples = [
            {
                "text": "This is sample text with label:",
                "label": "A",
                "A": "choice A",
                "B": "choice B"
            },
            {
                "text": "This is sample text with label:",
                "label": "B",
                "A": "choice A",
                "B": "choice B"
            }
        ]

    def test_reader(self):
        with MockReader(self.samples, "{{ text }} {{ label }}") as reader:
            self.assertEqual(
                list(reader),
                ["This is sample text with label: A", "This is sample text with label: B"]
            )

    def test_multi_format(self):
        with MockReader(self.samples, ["{{ text }}", "{{ label }}"]) as reader:
            self.assertEqual(
                list(reader),
                [["This is sample text with label:", "A"], ["This is sample text with label:", "B"]]
            )

    def test_vars_function(self):
        with MockReader(self.samples, "{{ vars()[label] }}") as reader:
            self.assertEqual(
                list(reader),
                ["choice A", "choice B"]
            )


class TestReaderComplexFormats(TestCase):
    def setUp(self):
        self.samples = [
            {
                'question': 'Jaké je chemické složení vody?',
                'mc_answer1': 'Jeden atom vodíku a dva atomy kyslíku',
                'mc_answer2': 'Jeden atom vodíku a jeden atom kyslíku',
                'mc_answer3': 'Dva atomy vodíku a dva atomy kyslíku',
                'mc_answer4': 'Dva atomy vodíku a jeden atom kyslíku',
                'correct_answer_num': '4',
            },
        ]

    def test_vars_with_concat(self):
        with MockReader(self.samples, "{{ vars()['mc_answer' + correct_answer_num] }}") as reader:
            self.assertEqual(
                list(reader),
                ["Dva atomy vodíku a jeden atom kyslíku"]
            )
