# -*- coding: UTF-8 -*-
"""
Created on 31.05.24

:author:     Martin Dočekal
"""
import json
import os
from pathlib import Path
from shutil import rmtree
from unittest import TestCase

from datasets import load_dataset, load_from_disk
import string

from decontaminator.__main__ import create_ngram_map, decontaminate, search_contaminated, indices_2_dataset_indices, \
    filter_hf_dataset, contaminated_ngrams_per_dataset
from decontaminator.myjson import json_dumps

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp")
FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures")


class TestBase(TestCase):
    def tearDown(self) -> None:
        super().tearDown()
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                if f.is_dir():
                    rmtree(f)
                else:
                    os.remove(f)

    def compare_jsonl(self, gt: str, res: str, unordered: bool = False):
        """
        Checks two jsonl files that they have the same content.

        :param gt: path to ground truth file
        :param res: path to file with results
        :param unordered: All lists are changed to sets
        """
        with open(gt, "r") as gt_f, open(res, "r") as res_f:
            if unordered:
                def convert(line):
                    def convert_var(v):
                        if isinstance(v, list):
                            return frozenset(convert_var(x) for x in v)
                        elif isinstance(v, dict):
                            return frozenset((k, convert_var(x)) for k, x in v.items())
                        else:
                            return v

                    return {k: convert_var(v) for k, v in json.loads(line).items()}

                for i, (gt, res) in enumerate(zip(gt_f.readlines(), res_f.readlines())):
                    self.assertDictEqual(convert(gt), convert(res), msg=f"Problem with line {i}")
            else:
                gt_lines = [json.loads(line) for line in gt_f.readlines()]
                res_lines = [json.loads(line) for line in res_f.readlines()]

                self.assertListEqual(gt_lines, res_lines)


DATASET_FOR_MAP_CREATE_PATH = os.path.join(FIXTURES_PATH, "dataset_for_map_create.jsonl")
CREATED_MAP_PATH = os.path.join(TMP_PATH, "created_map.db")


class TestCreateNgramMap(TestBase):

    def test_create_ngram_map(self):
        create_ngram_map(DATASET_FOR_MAP_CREATE_PATH, 2, CREATED_MAP_PATH, format_str="{content}",
                         hf_cache=None)

        with open(CREATED_MAP_PATH, "r") as f:
            db = json.load(f)
            self.assertEqual(len(db), 9)
            self.assertEqual(db[json_dumps(["hello", "world"])], [0, 1])
            self.assertEqual(db[json_dumps(["world", "how"])], [1])
            self.assertEqual(db[json_dumps(["how", "are"])], [1])
            self.assertEqual(db[json_dumps(["are", "you"])], [1])
            self.assertEqual(db[json_dumps(["i", "m"])], [2])
            self.assertEqual(db[json_dumps(["m", "fine"])], [2])
            self.assertEqual(db[json_dumps(["fine", "thank"])], [2])
            self.assertEqual(db[json_dumps(["thank", "you"])], [2])
            self.assertEqual(db["metadata"], {
                "n": 2,
                "allow_shorter": False,
                "format_str": "{content}",
                "samples": 3,
                "sub_chars": [string.punctuation]
            })


DECONTAMINATE_MAP_PATH = os.path.join(FIXTURES_PATH, "decontaminate_map.json")
DATASET_FOR_DECONTAMINATE_PATH = os.path.join(FIXTURES_PATH, "decontaminate.jsonl")
DECONTAMINATED_DATASET_PATH = os.path.join(FIXTURES_PATH, "decontaminated.jsonl")
DECONTAMINATED_COMMON_DATASET_PATH = os.path.join(FIXTURES_PATH, "decontaminated_common.jsonl")

DECONTAMINATED_OUTPUT_PATH = os.path.join(TMP_PATH, "decontaminated.jsonl")


class TestDecontaminate(TestBase):

    def test_decontaminate(self):
        decontaminate(DATASET_FOR_DECONTAMINATE_PATH, DECONTAMINATE_MAP_PATH, DECONTAMINATED_OUTPUT_PATH,
                      "content", window_size=25, removal_char_boundary=70, workers=1)

        self.compare_jsonl(DECONTAMINATED_DATASET_PATH, DECONTAMINATED_OUTPUT_PATH)

    def test_decontaminate_common(self):
        decontaminate(DATASET_FOR_DECONTAMINATE_PATH, DECONTAMINATE_MAP_PATH, DECONTAMINATED_OUTPUT_PATH,
                      "content", window_size=25, removal_char_boundary=70, ignore_above=9, workers=1)

        self.compare_jsonl(DECONTAMINATED_COMMON_DATASET_PATH, DECONTAMINATED_OUTPUT_PATH)


CONTAMINATED_SEARCH_MAP_PATH = os.path.join(FIXTURES_PATH, "contaminated_search_map.json")
CONTAMINATED_SEARCH_INDICES_RESULTS_PATH = os.path.join(FIXTURES_PATH, "contaminated_search_indices_results.json")
CONTAMINATED_SEARCH_INDICES_RESULTS_COMMON_PATH = os.path.join(FIXTURES_PATH, "contaminated_search_indices_results_common.json")
CONTAMINATED_SEARCH_NGRAM_RESULTS_PATH = os.path.join(FIXTURES_PATH, "contaminated_search_ngram_results.json")
CONTAMINATED_SEARCH_NGRAM_RESULTS_COMMON_PATH = os.path.join(FIXTURES_PATH, "contaminated_search_ngram_results_common.json")
CONTAMINATED_SEARCH_INDICES_OUTPUT_PATH = os.path.join(TMP_PATH, "contaminated_search_indices_results.json")
CONTAMINATED_SEARCH_NGRAM_OUTPUT_PATH = os.path.join(TMP_PATH, "contaminated_search_ngram_results.json")


class TestSearchContaminated(TestBase):

    def test_search_contaminated(self):
        search_contaminated(DATASET_FOR_DECONTAMINATE_PATH, CONTAMINATED_SEARCH_MAP_PATH,
                            CONTAMINATED_SEARCH_INDICES_OUTPUT_PATH, CONTAMINATED_SEARCH_NGRAM_OUTPUT_PATH,
                            "content", workers=1)

        self.compare_jsonl(CONTAMINATED_SEARCH_INDICES_RESULTS_PATH, CONTAMINATED_SEARCH_INDICES_OUTPUT_PATH)
        self.compare_jsonl(CONTAMINATED_SEARCH_NGRAM_RESULTS_PATH, CONTAMINATED_SEARCH_NGRAM_OUTPUT_PATH)

    def test_search_contaminated_common(self):
        search_contaminated(DATASET_FOR_DECONTAMINATE_PATH, CONTAMINATED_SEARCH_MAP_PATH,
                            CONTAMINATED_SEARCH_INDICES_OUTPUT_PATH, CONTAMINATED_SEARCH_NGRAM_OUTPUT_PATH,
                            "content", ignore_above=2, workers=1)

        self.compare_jsonl(CONTAMINATED_SEARCH_INDICES_RESULTS_COMMON_PATH, CONTAMINATED_SEARCH_INDICES_OUTPUT_PATH)
        self.compare_jsonl(CONTAMINATED_SEARCH_NGRAM_RESULTS_COMMON_PATH, CONTAMINATED_SEARCH_NGRAM_OUTPUT_PATH)


INDICES_PATH = os.path.join(FIXTURES_PATH, "indices.json")
MAP_METADATA_FOR_INDICES_CONVERSION_PATH = os.path.join(FIXTURES_PATH, "map_metadata_for_indices_conversion.json")
CONVERTED_INDICES_PATH = os.path.join(FIXTURES_PATH, "converted_indices.json")
CONVERTED_INDICES_OUTPUT_PATH = os.path.join(TMP_PATH, "converted_indices.json")


class TestIndices2DatasetIndices(TestBase):
    def test_indices_2_dataset_indices(self):
        indices_2_dataset_indices(INDICES_PATH, MAP_METADATA_FOR_INDICES_CONVERSION_PATH, CONVERTED_INDICES_OUTPUT_PATH)
        self.compare_jsonl(CONVERTED_INDICES_PATH, CONVERTED_INDICES_OUTPUT_PATH)


CONTAMINATED_NGRAMS_PER_DATASET_PATH = os.path.join(FIXTURES_PATH, "contaminated_ngrams_per_dataset.json")
CONTAMINATED_NGRAMS_PER_DATASET_OUTPUT_PATH = os.path.join(TMP_PATH, "contaminated_ngrams_per_dataset.json")


class TestContaminatedNGramsPerDataset(TestBase):
    def test_contaminated_ngrams_per_dataset(self):
        contaminated_ngrams_per_dataset(CONTAMINATED_SEARCH_NGRAM_RESULTS_PATH, CONTAMINATED_SEARCH_MAP_PATH,
                                        CONTAMINATED_NGRAMS_PER_DATASET_OUTPUT_PATH)
        self.compare_jsonl(CONTAMINATED_NGRAMS_PER_DATASET_PATH, CONTAMINATED_NGRAMS_PER_DATASET_OUTPUT_PATH)


HF_DATASET_PATH = os.path.join(FIXTURES_PATH, "hf_dataset")
FILTERING_INDICES_PATH = os.path.join(FIXTURES_PATH, "filtering_indices.json")
FILTERED_HF_DATASET_PATH = os.path.join(FIXTURES_PATH, "filtered_hf_dataset")
FILTERED_HF_DATASET_OUTPUT_PATH = os.path.join(TMP_PATH, "filtered_hf_dataset")


class TestFilterHFDataset(TestBase):
    def test_filter_hf_dataset(self):
        filter_hf_dataset(HF_DATASET_PATH, None, "test", FILTERED_HF_DATASET_OUTPUT_PATH, FILTERING_INDICES_PATH,
                          "hf_dataset")

        gt_dataset = load_dataset(FILTERED_HF_DATASET_PATH)
        res_dataset = load_dataset(FILTERED_HF_DATASET_OUTPUT_PATH)

        self.assertEqual(len(gt_dataset), len(res_dataset))
        self.assertEqual(len(gt_dataset["train"]), len(res_dataset["train"]))
        self.assertEqual(len(gt_dataset["validation"]), len(res_dataset["validation"]))
        self.assertEqual(len(gt_dataset["test"]), len(res_dataset["test"]))

        for split in ["train", "validation", "test"]:
            for gt_data, res_data in zip(gt_dataset[split], res_dataset[split]):
                self.assertDictEqual(gt_data, res_data, msg=f"Problem with {split}")

