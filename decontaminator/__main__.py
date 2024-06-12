# -*- coding: UTF-8 -*-
"""
Created on 31.05.24
Main module of the decontaminator.

:author:     Martin Dočekal
"""
import json
import logging
import math
import multiprocessing
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictWriter
from typing import Optional, Iterable, Tuple

import orjson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from windpyutils.parallel.own_proc_pools import FunctorWorker, FunctorPool
from windpyutils.structures.maps import ImmutIntervalMap

from decontaminator.cython.tokenizer import whitespace_tokenizer
from decontaminator.myjson import json_dumps, json_loads, json_dump, json_load
from decontaminator.ngrams import ngrams
from decontaminator.normalize import normalize_text
from decontaminator.reader import HFDatasetReader, JSONLReader


def create_ngram_map(dataset: str, n: int, output: str, allow_shorter: bool = False, format_str: str = None,
                     split: Optional[str] = None, dataset_config: str = None, hf_cache: str = None):
    """
    Creates n-gram map from the dataset.

    :param dataset: Path to the hf dataset or jsonl dataset.
    :param n: Size of n-gram.
    :param output: Path to the output.
    :param allow_shorter: If the sequence is shorter than n, it will generate one n-gram that will be the same as the sequence.
    :param format_str: This argument allows to select only certain columns and marge them using this formating string.
        E.g.: '{firstname} {lastname}'
    :param split: HF dataset split name for test dataset.
    :param dataset_config: HF dataset configuration name for test dataset.
    :param hf_cache: Path to the HF cache.
    """
    ngram_map = defaultdict(list)    # ngram -> number of documents where the ngram is present
    jsonl_read = dataset.endswith(".jsonl")
    reader = JSONLReader(dataset, format_str) if jsonl_read else HFDatasetReader(dataset, split, dataset_config,
                                                                                 format_str, hf_cache)
    with reader as r, tqdm(desc="Creating n-gram map", total=os.path.getsize(dataset) if jsonl_read else len(reader)) as pbar:

        for i, line in enumerate(r):
            doc_ng = set()
            for ng in ngrams(normalize_text(line).split(), n, allow_shorter):
                str_ng = json_dumps(ng)
                if str_ng in doc_ng:
                    continue

                doc_ng.add(str_ng)

                ngram_map[str_ng].append(i)

            if jsonl_read:
                pbar.update(r.file.tell() - pbar.n)
            else:
                pbar.update(1)

    ngram_map["metadata"] = {"n": n, "allow_shorter": allow_shorter, "format_str": format_str, "samples": i + 1}
    with open(output, "w") as f:
        json_dump(ngram_map, f)


def create_ngram_map_entry_point(args):
    create_ngram_map(args.dataset, args.n, args.output, args.allow_shorter, args.format_str, args.split,
                     args.dataset_config, args.hf_cache)


def merge_maps(maps: Iterable[str], output: str):
    """
    Merges n-gram maps.

    :param maps: Paths to the n-gram maps.
    :param output: Path where the merged map will be stored.
    """

    ngram_map = defaultdict(list)
    offset = 0
    ngram_map["metadata"] = {}
    for map_path in maps:
        with open(map_path, "r") as f:
            map_data = json_load(f)

            for k, v in map_data.items():
                if k != "metadata":
                    ngram_map[k].extend([i + offset for i in v])

            map_data["metadata"]["start_offset"] = offset
            offset += map_data["metadata"]["samples"]
            map_data["metadata"]["end_offset"] = offset

            ngram_map["metadata"][Path(map_path).stem] = map_data["metadata"]

    with open(output, "w") as f:
        json_dump(ngram_map, f)


def merge_map_parser_entry_point(args):
    merge_maps(args.maps, args.output)


def cnt_ngram_sizes(ngram_map: dict[str, int]) -> list[int]:
    """
    Returns the sizes of n-grams in the map.

    :param ngram_map: N-gram map.
    :return: Sizes of n-grams in the map in descending order.
    """

    return sorted(set(len(json_loads(k)) for k in ngram_map if k != "metadata"), reverse=True)


class DecontaminateWorker(FunctorWorker):
    """
    Parallel worker for decontamination of the dataset.
    """
    def __init__(self, forbidden_ngrams: set[str], ngram_sizes: list[int],
                 field: str, window_size: int, removal_char_boundary: int):
        """
        :param forbidden_ngrams: ngrams that contaminates the dataset, this should be already filtered final set
            of ngrams
        :param ngram_sizes: sizes of ngrams in the forbidden_ngrams
        :param field: content field
        :param window_size: Size of the window for n-grams removal in chars (might be longer as it will not break words).
        :param removal_char_boundary:  If we are about te remove more than removal_char_boundary characters then whole document is discarded.
        """
        super().__init__()
        self.forbidden_ngrams = forbidden_ngrams
        self.ngram_sizes = ngram_sizes
        self.field = field
        self.window_size = window_size
        self.removal_char_boundary = removal_char_boundary

    def __call__(self, proc: tuple[int, str]) -> Tuple[int, Optional[str]]:
        """
        Decontaminates the record on line_offset.

        :param proc:
            - number of read bytes
            - line
        :return:
            - number of read bytes
            - decontaminated line or None if the line was removed
        """

        proc_bytes = proc[0]
        line = json_loads(proc[1])
        content = line[self.field]

        tokens = whitespace_tokenizer(normalize_text(content))  # for normalized string the offsets remain the same

        contaminated_intervals = []

        for n in self.ngram_sizes:
            for ng in ngrams(tokens, n, allow_shorter=False):
                str_ng = json_dumps([t[0] for t in ng])
                if str_ng in self.forbidden_ngrams:
                    contaminated_intervals.append((ng[0][1], ng[-1][2]))

        # create windows for removal
        windows = []

        for start, end in contaminated_intervals:

            if end - start < self.window_size:
                remove_shift = (self.window_size - (end - start)) // 2
                start = max(0, start - remove_shift)
                end = min(len(content), end + remove_shift)

            # respect word boundaries
            while start > 1 and not content[start - 1].isspace():
                start -= 1
            while end < len(content) and not content[end].isspace():
                end += 1

            if not windows or (windows[-1][1] < start and start - windows[-1][1] > self.window_size):
                windows.append([start, end])
            else:
                windows[-1][1] = end

        number_of_chars_to_remove = sum(end - start for start, end in windows)

        if number_of_chars_to_remove > self.removal_char_boundary:
            return proc_bytes, None

        if windows:
            # add empty interval at the beginning
            windows.insert(0, [0, 0])
            new_content = " ".join(
                content[end_before:start].strip() for (_, end_before), (start, _) in zip(windows, windows[1:]))
            new_content = new_content.strip()  # there might be space at the start when the first window starts at the beginning of the document
            end_suffix = content[windows[-1][1]:].strip()  # add the rest of the document
            if len(new_content) > 0 and len(end_suffix) > 0:
                new_content += " "
            new_content += end_suffix

            line[self.field] = new_content

            if not line[self.field]:
                return proc_bytes, None

        return proc_bytes, json_dumps(line)


def decontaminate(dataset_path: str, ngram_map_file: str, output: str, field: str, ignore_above: int = math.inf,
                  window_size: int = math.inf, removal_char_boundary: int = math.inf, workers: int = -1):
    """
    Decontaminates the dataset.

    :param dataset_path: Path to jsonl dataset.
    :param ngram_map_file: Path to n-gram map with n-grams that should be removed and their frequency among original documents.
    :param output: Path where the decontaminated dataset will be stored.
    :param field: content field
    :param ignore_above: Ignore n-grams that are more frequent than this value.
    :param window_size: Size of the window for n-grams removal in chars (might be longer as it will not break words).
    Also. if there will be a piece (e.g., between two windows) that is shorter than this value then it will be discarded.
    :param removal_char_boundary: If we are about te remove more than removal_char_boundary characters then whole document is discarded.
    :param workers: Number of parallel workers. If -1 then it will use all available CPUs.
    """
    workers = workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with open(ngram_map_file, "r") as f:
        forbidden_ngrams = {k for k, v in json_load(f).items() if len(v) <= ignore_above}

    ngram_sizes = cnt_ngram_sizes(forbidden_ngrams)

    with open(dataset_path, "r") as dataset, \
            tqdm(desc="Decontaminating", total=os.path.getsize(dataset_path)) as pbar, open(output, "w") as out:

        workers = [
            DecontaminateWorker(forbidden_ngrams, ngram_sizes, field, window_size, removal_char_boundary)
            for _ in range(workers)
        ]

        def read_dataset():
            line_offset = 0
            while cur_line := dataset.readline():
                yield dataset.tell() - line_offset, cur_line
                line_offset = dataset.tell()

        with FunctorPool(workers) as pool:
            for proc_bytes, line in pool.imap_unordered(read_dataset(), 100):
                if line is not None:
                    out.write(line)
                    out.write("\n")
                pbar.update(proc_bytes)


def decontaminate_entry_point(args):
    decontaminate(args.dataset, args.ngram_map, args.output, args.field, args.ignore_above, args.window_size,
                  args.removal_char_boundary, workers=args.workers)


class SearchContaminatedWorker(FunctorWorker):
    """
    Parallel worker for searching contaminated samples.
    """
    def __init__(self, forbidden_ngrams: dict[str, list[int]], ngram_sizes: list[int], field: str):
        """
        :param forbidden_ngrams:
            key - ngram
            value - list of indices of documents where the ngram is present
                all those indexes are marked as contaminated when the ngram is found in input string
        :param ngram_sizes: sizes of ngrams in the forbidden_ngrams
        :param field: content field
        """
        super().__init__()
        self.forbidden_ngrams = forbidden_ngrams
        self.ngram_sizes = ngram_sizes
        self.field = field

    def __call__(self, proc: tuple[int, str]) -> Tuple[int, list[int], list[str]]:
        """
        Decontaminates the record on line_offset.

        :param proc:
            - number of read bytes
            - line
        :return:
            - number of read bytes
            - contaminated indices
            - contaminated ngrams
        """

        proc_bytes = proc[0]
        line = json_loads(proc[1])
        content = line[self.field]

        tokens = normalize_text(content).split()  # for normalized string the offsets remain the same

        contaminated_indices = set()
        contaminated_ngrams = set()

        for n in self.ngram_sizes:
            for ng in ngrams(tokens, n, allow_shorter=False):
                str_ng = json_dumps(ng)
                if str_ng in contaminated_ngrams:
                    continue

                if str_ng in self.forbidden_ngrams:
                    contaminated_indices.update(self.forbidden_ngrams[str_ng])
                    contaminated_ngrams.add(str_ng)

        return proc_bytes, list(contaminated_indices), list(contaminated_ngrams)


def search_contaminated(dataset_path: str, ngram_map_file: str, contaminated_indices_path: str, contaminated_ngrams_path: str,
                        field: str, ignore_above: int = math.inf,
                        workers: int = -1):
    """
    Decontaminates the dataset.

    :param dataset_path: Path to jsonl dataset.
    :param ngram_map_file: Path to n-gram map with n-grams that should be removed and their frequency among original documents.
    :param contaminated_indices_path: Path where the contaminated sample indices will be stored. The indice is considered contaiminated when associated ngram to indice occurs in the input dataset.
    :param contaminated_ngrams_path: Path where the contaminated ngrams will be stored.
    :param field: content field
    :param ignore_above: Ignore n-grams that are more frequent than this value.
    :param workers: Number of parallel workers. If -1 then it will use all available CPUs.
    """

    workers = workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with open(ngram_map_file, "r") as f:
        forbidden_ngrams = {k: v for k, v in json_load(f).items() if len(v) <= ignore_above}

    ngram_sizes = cnt_ngram_sizes(forbidden_ngrams)

    with open(dataset_path, "r") as dataset, \
            tqdm(desc="Searching contaminated", total=os.path.getsize(dataset_path)) as pbar:

        workers = [
            SearchContaminatedWorker(forbidden_ngrams, ngram_sizes, field)
            for _ in range(workers)
        ]

        def read_dataset():
            line_offset = 0
            while cur_line := dataset.readline():
                yield dataset.tell() - line_offset, cur_line
                line_offset = dataset.tell()

        contaminated_indices = set()
        contaminated_ngrams = set()
        with FunctorPool(workers) as pool:
            for proc_bytes, current_contaminated_indices, current_contaminated_ngrams in pool.imap_unordered(read_dataset(), 100):
                contaminated_indices.update(current_contaminated_indices)
                contaminated_ngrams.update(current_contaminated_ngrams)
                pbar.update(proc_bytes)

        with open(contaminated_indices_path, "w") as out:
            json_dump(sorted(contaminated_indices), out)

        with open(contaminated_ngrams_path, "w") as out:
            json_dump(sorted(contaminated_ngrams), out)


def search_contaminated_entry_point(args):
    search_contaminated(args.dataset, args.ngram_map, args.contaminated_indices, args.contaminated_ngrams,
                        args.field, args.ignore_above,
                        workers=args.workers)


def indices_2_dataset_indices(indices: str, ngram_map: str, output: str):
    """
    Converts indices of merged dataset into indices of original datasets.

    :param indices: Path to json file with indices of merged dataset.
    :param ngram_map: Path to merged n-gram map. Its metadata should contain information about the original datasets.
    :param output: Path to results.
    It will create a json file with map having dataset names as keys and list of indices as values.
    """

    with open(indices, "r") as f:
        indices = json_load(f)

    with open(ngram_map, "r") as f:
        ngram_map = json_load(f)

    indices = sorted(indices)

    dataset_indices = defaultdict(list)

    dataset_names = sorted(ngram_map["metadata"].keys(), key=lambda x: ngram_map["metadata"][x]["start_offset"])

    current_dataset = 0

    for i in indices:
        while i >= ngram_map["metadata"][dataset_names[current_dataset]]["end_offset"] and current_dataset < len(dataset_names):
            current_dataset += 1

        dataset_indices[dataset_names[current_dataset]].append(
            i - ngram_map["metadata"][dataset_names[current_dataset]]["start_offset"]
        )

    with open(output, "w") as f:
        json_dump(dataset_indices, f)


def indices_2_dataset_indices_entry_point(args):
    indices_2_dataset_indices(args.indices, args.ngram_map, args.output)


def contaminated_ngrams_per_dataset(ngrams_path: str, ngram_map: str, output: str):
    """
    From a list of contaminated ngrams creates a map of ngrams per each dataset, and it assigns associated
    indices of contaminated samples

    :param ngrams_path: Path to json file with list of contaminated ngrams.
    :param ngram_map: Path to merged n-gram map. Its metadata should contain information about the original datasets.
    :param output: Path to results.
    """

    with open(ngrams_path, "r") as f:
        contaminated_ngrams = json_load(f)

    with open(ngram_map, "r") as f:
        ngram_map = json_load(f)

    res = defaultdict(lambda: defaultdict(list))

    interval_2_dataset = ImmutIntervalMap({
        (metadata["start_offset"], metadata["end_offset"]-1): dataset for dataset, metadata in ngram_map["metadata"].items()
    })

    for ngram in contaminated_ngrams:
        indices = ngram_map[ngram]

        for i in indices:
            dataset = interval_2_dataset[i]
            res[dataset][ngram].append(i-ngram_map["metadata"][dataset]["start_offset"])

    with open(output, "w") as f:
        json_dump(res, f)


def contaminated_ngrams_per_dataset_entry_point(args):
    contaminated_ngrams_per_dataset(args.ngrams_path, args.ngram_map, args.output)


def filter_hf_dataset(dataset: str, config: Optional[str], split: Optional[str], output: str, contaminated_indices: str,
                      contaminated_indices_selector: Optional[str] = None, hf_cache: Optional[str] = None,
                      write_csv_header: bool = True):
    """
    Filters the HF dataset.

    :param dataset: Path to the HF dataset.
    :param config: Name of the configuration.
    :param split: Split name of the dataset.
    :param output: Path to the output.
    :param contaminated_indices: Path to the contaminated indices.
    :param contaminated_indices_selector: When the contaminated_indices file contains indices for multiple datasets
        then this argument allows to select only indices for the current dataset.
    :param hf_cache: Path to the HF cache.
    :param write_csv_header: If True then it will write the header of the CSV.
    """
    with open(contaminated_indices, "r") as f:
        contaminated_indices = json_load(f)
        if contaminated_indices_selector is not None:
            try:
                contaminated_indices = contaminated_indices[contaminated_indices_selector]
            except KeyError:
                contaminated_indices = []

        contaminated_indices = set(contaminated_indices)

    dataset_path = dataset
    dataset = load_dataset(dataset, config, cache_dir=hf_cache)

    if len(contaminated_indices):
        split_of_interest = dataset if split is None else dataset[split]
        filtered = split_of_interest.select([i for i in range(len(split_of_interest)) if i not in contaminated_indices])

        csv_writer = DictWriter(sys.stdout, fieldnames=["dataset", "config", "split", "number of contaminated indices", "contamination percentage"])
        if write_csv_header:
            csv_writer.writeheader()

        csv_writer.writerow({
            "dataset": dataset_path,
            "config": config,
            "split": split,
            "number of contaminated indices": len(contaminated_indices),
            "contamination percentage": len(contaminated_indices)/len(split_of_interest) * 100
        })

        if split is None:
            dataset = filtered
        else:
            dataset[split] = filtered

    for split_name, split_dataset in dataset.items():
        split_dataset.to_json(Path(output) / f"{split_name}.jsonl", force_ascii=False)


def filter_hf_dataset_entry_point(args):
    filter_hf_dataset(args.dataset, args.dataset_config, args.split, args.output, args.contaminated_indices,
                      args.contaminated_indices_selector, args.hf_cache, args.write_csv_header)


def main():
    logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s', level=logging.WARNING)

    parser = ArgumentParser(description="Tool for decontamination of training datasets.")
    subparsers = parser.add_subparsers()

    make_ngram_map_parser = subparsers.add_parser("make_ngram_map", help="Creates n-gram map from the dataset.")
    make_ngram_map_parser.add_argument("dataset", help="Path to the hf dataset or jsonl dataset.")
    make_ngram_map_parser.add_argument("output", help="Path where the map will be stored.")
    make_ngram_map_parser.add_argument("n", help="Size of n-gram.", type=int)
    make_ngram_map_parser.add_argument("--allow_shorter", help="If the sequence is shorter than n, it will generate one n-gram that will be the same as the sequence.", action="store_true")
    make_ngram_map_parser.add_argument("--format_str", help="This argument allows to select only certain columns and marge them using this formating string. E.g.: '{firstname} {lastname}'", default=None)
    make_ngram_map_parser.add_argument("--split", help="HF dataset split name for test dataset.", default=None)
    make_ngram_map_parser.add_argument("--dataset_config", help="HF dataset configuration name for test dataset.", default=None)
    make_ngram_map_parser.add_argument("--hf_cache", help="Path to the HF cache.", default=None)
    make_ngram_map_parser.set_defaults(func=create_ngram_map_entry_point)

    merge_map_parser = subparsers.add_parser("merge_map", help="Merges n-gram maps.")
    merge_map_parser.add_argument("maps", nargs="+", help="Path to n-gram maps.")
    merge_map_parser.add_argument("--output", help="Path where the merged map will be stored.")
    merge_map_parser.set_defaults(func=merge_map_parser_entry_point)

    decontaminate_parser = subparsers.add_parser("decontaminate", help="Decontaminates the dataset.")
    decontaminate_parser.add_argument("dataset", help="Path to jsonl dataset.")
    decontaminate_parser.add_argument("ngram_map", help="Path to n-gram map with n-grams that should be removed")
    decontaminate_parser.add_argument("output", help="Path where the decontaminated dataset will be stored.")
    decontaminate_parser.add_argument("--field", help="content field")
    decontaminate_parser.add_argument("--ignore_above", help="Ignore n-grams that are more frequent than this value.", type=int, default=None)
    decontaminate_parser.add_argument("--window_size", help="Size of the window for n-grams removal in chars (might be longer as it will not break words). Also. if there will be a piece (e.g., between two windows) that is shorter than this value then it will be discarded.", type=int, default=math.inf)
    decontaminate_parser.add_argument("--removal_char_boundary", help="If we are about te remove more than removal_char_boundary characters then whole document is discarded.", type=int, default=math.inf)
    decontaminate_parser.add_argument("--workers", help="Number of parallel workers. If -1 then it will use all available CPUs.", type=int, default=-1)
    decontaminate_parser.set_defaults(func=decontaminate_entry_point)

    search_contaminated_parser = subparsers.add_parser("search_contaminated", help="Searches contaminated samples and ngrams.")
    search_contaminated_parser.add_argument("dataset", help="Path to jsonl dataset. Every ngram in this dataset is considered to be as contamination.")
    search_contaminated_parser.add_argument("ngram_map", help="Path to n-gram map with n-grams and associated sample indices.")
    search_contaminated_parser.add_argument("contaminated_indices", help="Path where the contaminated sample indices will be stored. The indice is considered contaiminated when associated ngram to indice occurs in the input dataset.")
    search_contaminated_parser.add_argument("contaminated_ngrams", help="Path where the contaminated ngrams will be stored.")
    search_contaminated_parser.add_argument("--field", help="content field")
    search_contaminated_parser.add_argument("--ignore_above", help="Ignore n-grams that are more frequent than this value.", type=int, default=None)
    search_contaminated_parser.add_argument("--workers", help="Number of parallel workers. If -1 then it will use all available CPUs.", type=int, default=-1)
    search_contaminated_parser.set_defaults(func=search_contaminated_entry_point)

    indices_2_dataset_indices_parser = subparsers.add_parser("indices_2_dataset_indices", help="Converts indices of merged dataset into indices of original datasets.")
    indices_2_dataset_indices_parser.add_argument("indices", help="Path to json file with indices of merged dataset.")
    indices_2_dataset_indices_parser.add_argument("ngram_map",
                                                  help="Path to merged n-gram map. Its metadata should contain information about the original datasets.")
    indices_2_dataset_indices_parser.add_argument("output", help="Path to results. It will create a json file with map having dataset names as keys and list of indices as values.")
    indices_2_dataset_indices_parser.set_defaults(func=indices_2_dataset_indices_entry_point)

    contaminated_ngrams_per_dataset_parser = subparsers.add_parser("contaminated_ngrams_per_dataset", help="From a list of contaminated ngrams creates a map of ngrams per each dataset and it assigns associated indices of contaminated samples.")
    contaminated_ngrams_per_dataset_parser.add_argument("ngrams_path", help="Path to json file with list of contaminated ngrams.")
    contaminated_ngrams_per_dataset_parser.add_argument("ngram_map", help="Path to merged n-gram map. Its metadata should contain information about the original datasets.")
    contaminated_ngrams_per_dataset_parser.add_argument("output", help="Path to results.")
    contaminated_ngrams_per_dataset_parser.set_defaults(func=contaminated_ngrams_per_dataset_entry_point)

    filter_hf_dataset_parser = subparsers.add_parser("filter_hf_dataset", help="Filters the HF dataset.")
    filter_hf_dataset_parser.add_argument("dataset", help="Path to the HF dataset.")
    filter_hf_dataset_parser.add_argument("output", help="Path to the output.")
    filter_hf_dataset_parser.add_argument("contaminated_indices", help="Path to the contaminated indices.")
    filter_hf_dataset_parser.add_argument("--dataset_config", help="Name of the configuration.", default=None)
    filter_hf_dataset_parser.add_argument("--split", help="Split name of the dataset.", default=None)
    filter_hf_dataset_parser.add_argument("--contaminated_indices_selector", help="When the contaminated_indices file contains indices for multiple datasets then this argument allows to select only indices for the current dataset.", default=None)
    filter_hf_dataset_parser.add_argument("--hf_cache", help="Path to the HF cache.", default=None)
    filter_hf_dataset_parser.add_argument("--write_csv_header", help="If True then it will write the header of the CSV.", action="store_true")
    filter_hf_dataset_parser.set_defaults(func=filter_hf_dataset_entry_point)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
