# -*- coding: UTF-8 -*-
"""
Created on 13.09.24

:author:     Martin Dočekal
"""
from typing import Generator, Union, Sequence
from unittest import TestCase

from decontaminator.reader import Reader


class MockReader(Reader):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self) -> Generator[Union[str, Sequence[str]], None, None]:
        samples = [
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

        for sample in samples:
            if self.multi_format:
                yield [j.render(sample) for j in self.jinja]
            else:
                yield self.jinja.render(sample)


class TestReader(TestCase):

    def test_reader(self):
        with MockReader("{{ text }} {{ label }}") as reader:
            self.assertEqual(
                list(reader),
                ["This is sample text with label: A", "This is sample text with label: B"]
            )

    def test_multi_format(self):
        with MockReader(["{{ text }}", "{{ label }}"]) as reader:
            self.assertEqual(
                list(reader),
                [["This is sample text with label:", "A"], ["This is sample text with label:", "B"]]
            )

    def test_vars_function(self):
        with MockReader("{{ vars()[label] }}") as reader:
            self.assertEqual(
                list(reader),
                ["choice A", "choice B"]
            )


