# -*- coding: UTF-8 -*-
"""
Created on 03.06.24

:author:     Martin Dočekal
"""
from abc import ABC, abstractmethod
from typing import Generator, Optional

import orjson
from datasets import load_dataset


class Reader(ABC):
    """
    Abstract reader class.
    """

    def __init__(self, format_str: str):
        """
        Initializes reader.

        :param format_str: Format string that will be used for formatting the output.
        """
        self.format_str = format_str

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    @abstractmethod
    def __iter__(self) -> Generator[str, None, None]:
        ...


class HFDatasetReader(Reader):
    """
    Reader for the HF dataset.
    """

    def __init__(self, dataset: str, split: str, config_name: str, format_str: str, hf_cache: Optional[str] = None):
        """
        Initializes reader.

        :param dataset: Path to the HF dataset.
        :param split: Split name of the dataset.
        :param config_name: Name of the configuration.
        :param format_str: Format string that will be used for formatting the output.
        :param hf_cache: Path to the HF cache.
        """
        super().__init__(format_str)
        self.dataset = dataset
        self.config_name = config_name
        self._reader = load_dataset(dataset, config_name, cache_dir=hf_cache)[split]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        for data in self._reader:
            yield self.format_str.format(**data)

    def __len__(self):
        return len(self._reader)


class JSONLReader(Reader):
    """
    Reader for the JSONL dataset.
    """

    def __init__(self, dataset: str, format_str: Optional[str] = None):
        """
        Initializes reader.

        :param dataset: Path to the JSONL dataset.
        :param format_str: Format string that will be used for formatting the output.
            If None whole line will be returned.
        """
        super().__init__(format_str)
        self.dataset = dataset
        self.file = None

    def __enter__(self):
        self.file = open(self.dataset, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        self.file.seek(0)
        if self.format_str is None:
            while line := self.file.readline():
                yield line
            return
        else:
            while line := self.file.readline():
                yield self.format_str.format(**orjson.loads(line))
