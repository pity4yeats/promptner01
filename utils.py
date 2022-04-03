import dataclasses
import os
import sys
import json
import random
import pandas as pd
from dataclasses import dataclass
from typing import Optional

NGRAM = 0


@dataclass
class InputExample:
    guid: int
    sentence: str  # "United States can win if Donald Trump was never elected president ."
    labelseq: str  # "B-ORG I-ORG O O O B-PER I-PER O O O O O"
    golds: Optional[dict] = None  # {"United States": "ORG", "Donald Trump": "PER"}
    nonentity_indices: Optional[dict] = None  # [2, 3, 4, 7, 8, 9, 10, 11]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


def is_separate_line(line):
    return line.startswith('-DOCSTART-') or line == '' or line == '\n' or line == '\t\n'


def ngram(tokens_list, k):
    # Get the ngram set from kth index of given tokens_list(sentence)
    kgram = []
    for i in range(1, min(NGRAM, len(tokens_list) - k + 1)):
        kgram.append(' '.join(tokens_list[k:k + i]))
    return kgram


def save_dataset(dataset, path):
    # @dataset: A list of InputExamples
    # @path: Target file path to save the dataset

    for data in dataset:
        prompts = fill_template()
