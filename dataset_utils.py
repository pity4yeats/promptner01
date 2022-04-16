import dataclasses
import os
import sys
import json
import random
import pandas as pd
from dataclasses import dataclass
from typing import Optional

NGRAM = 9


@dataclass
class InputExample:
    """
    A list of InputExamples represents a single dataset.
    """
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


def get_golds(sentence, labelseq):
    gold_entities = {}
    tokens = sentence.rstrip().split(' ')
    labels = labelseq.rstrip().split(' ')
    num_tokens = len(tokens)
    num_labels = len(labels)
    if num_tokens != num_labels:
        sys.exit('Tokens\' number({}) does not comply with labels\' number({}).'.format(num_tokens, num_labels))
    gold_entity = ''
    for i, label in enumerate(labels):
        if label[0] == 'B':
            # Put last gold entity into result dictionary
            if gold_entity:
                gold_entities[gold_entity.rstrip()] = labels[i - 1][2:]
                gold_entity = ''
            # Append current token to the gold entity
            gold_entity += tokens[i] + ' '
            # Put current gold entity into result dictionary if last label
            if i == num_labels - 1:
                gold_entities[gold_entity.rstrip()] = label[2:]
        elif label[0] == 'I':
            # Always append current token to the gold entity first
            gold_entity += tokens[i] + ' '
            # Put current gold entity into result dictionary if last label
            if i == num_labels - 1:
                gold_entities[gold_entity.rstrip()] = label[2:]
        elif label[0] == 'O':
            if gold_entity:
                gold_entities[gold_entity.rstrip()] = labels[i - 1][2:]
                gold_entity = ''
        else:
            # print(f'{tokens[i]=}\n{labels[i]=}')
            print('token: {}\nlabel: {}'.format(tokens[i], labels[i]))
            sys.exit('Expecting B/I/O, instead got {}'.format(label[0]))
    return gold_entities


def get_input_examples(filepath):
    input_examples = []
    with open(filepath, 'r', encoding='utf-8') as filename:
        current_sentence = ''
        current_labelseq = ''
        guid = 0
        # Iterate over all lines in file, construct sentence and labels sequence
        for i, line in enumerate(filename):
            if not is_separate_line(line):
                # line: "United B-ORG"
                words = line.rstrip().replace('\t', ' ').split(' ')
                current_sentence += words[0] + ' '
                current_labelseq += words[1] + ' '
            else:  # When separate line encountered
                if not current_sentence:
                    continue
                sentence = current_sentence.rstrip()
                labelseq = current_labelseq.rstrip()
                golds = get_golds(sentence, labelseq)
                nonentity_indices = [i for i, x in enumerate(labelseq.split(' ')) if x == "O"]
                input_examples.append(
                    InputExample(
                        guid=guid,
                        sentence=sentence,
                        labelseq=labelseq,
                        golds=golds,
                        nonentity_indices=nonentity_indices
                    )
                )
                current_sentence = ''
                current_labelseq = ''
                guid += 1
    return input_examples


def get_prompts(input_example, template):
    label_map = {"LOC": "a location", "PER": "a person", "ORG": "an organization", "MISC": "an other"}
    # template = templates[0]
    sentence = input_example.sentence
    golds = input_example.golds
    tokens = sentence.split(' ')
    NT_SIZE = round(len(golds) * 2)
    prompts = {}

    neg_token_spans = []
    for i in range(len(tokens)):
        ith_ngram = ngram(tokens, i)
        # Exclude all golds from the ngram candidates
        neg_token_spans.extend([igram for igram in ith_ngram if igram not in golds])
    neg_token_spans = list(set(neg_token_spans))

    # Fill PT with all gold entities in sentence
    pos_prompts = []
    for gold, label in golds.items():
        prompt = template['pos'].replace('<token_span>', gold)
        prompt = prompt.replace('<mask>', label_map[label])
        pos_prompts.append(prompt)

    # Fill NT with randomly selected token spans
    neg_prompts = []
    if len(neg_token_spans) >= NT_SIZE:
        for token_span in random.sample(neg_token_spans, NT_SIZE):
            neg_prompts.append(template['neg'].replace('<token_span>', token_span))
    else:
        for token_span in neg_token_spans:
            neg_prompts.append(template['neg'].replace('<token_span>', token_span))

    # Keep positive and negative prompts separated in case of unexpected needs
    prompts['pos'] = pos_prompts
    prompts['neg'] = neg_prompts
    return prompts


def save_dataset(dataset, filepath, template):
    # @dataset: A list of InputExamples
    # @filepath: Target file path to save the dataset
    panda_data = []
    for input_example in dataset:
        prompts = get_prompts(input_example, template)
        for pos_prompt in prompts['pos']:
            panda_data.append((input_example.sentence, pos_prompt))
        for neg_prompt in prompts['neg']:
            panda_data.append((input_example.sentence, neg_prompt))
    data_frame = pd.DataFrame(data=panda_data, columns=['sentence', 'prompt'])
    data_frame.to_csv(filepath, sep=',', index=False, columns=['sentence', 'prompt'])


def describe_dataset(dataset, print_sample=False):
    gold_length_distribution = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0,
                                'extra': 0}
    total_num_tokens = 0
    total_num_labels = 0
    total_num_golds = 0
    labels = []
    for input_example in dataset:
        # Get all ner tags in given dataset
        for gold, label in input_example.golds.items():
            if label not in labels:
                labels.append(label)
            size = len(gold.split(' '))
            if size == 1:
                gold_length_distribution['1'] += 1
            elif size == 2:
                gold_length_distribution['2'] += 1
            elif size == 3:
                gold_length_distribution['3'] += 1
            elif size == 4:
                gold_length_distribution['4'] += 1
            elif size == 5:
                gold_length_distribution['5'] += 1
            elif size == 6:
                gold_length_distribution['6'] += 1
            elif size == 7:
                gold_length_distribution['7'] += 1
            elif size == 8:
                gold_length_distribution['8'] += 1
            elif size == 9:
                gold_length_distribution['9'] += 1
            elif size == 10:
                gold_length_distribution['10'] += 1
            else:
                gold_length_distribution['extra'] += 1
        # Tokens and labels might not share the same length, this two variables are for checking purpose
        total_num_tokens += len(input_example.sentence.split(' '))
        total_num_labels += len(input_example.labelseq.split(' '))
        total_num_golds += len(input_example.golds)

    data_summary = {
        "total_num_sentences": len(dataset),
        "total_num_tokens": total_num_tokens,
        "total_num_labels": total_num_labels,
        # There exists identical entities in a single sentence, which results in
        # the inconsistency between the number of occurrences of "B-*"(3377) and
        # `total_num_golds`(3315). to illustrate, consider following sentence:
        # e.g. "Trump and Obama were never friends because CNN claims that Obama
        # used to talk cheap about Trump."
        "total_num_golds": total_num_golds,
        "total_num_nonentity": total_num_labels - total_num_golds,
        "labels": sorted(labels),
        "gold_length_distribution": gold_length_distribution,
    }
    if print_sample:
        print("Randomly selected sample data:\n{}".format(random.choice(dataset)))
    for key, val in data_summary.items():
        print("{}: {}".format(key, val))


def build_dataset():
    original = {
        "conll03_train": "data/original/CoNLL03/train.txt",
        "conll03_devel": "data/original/CoNLL03/devel.txt",
        # "conll03_test": "data/original/CoNLL03/test.txt",

        "conll04_train": "data/original/CoNLL04/train.txt",
        "conll04_devel": "data/original/CoNLL04/devel.txt",
        # "conll04_test": "data/original/CoNLL04/test.txt",
    }

    for dataset_name, filepath in original.items():
        dataset = get_input_examples(filepath)
        size = len(dataset)
        portions = [20, 40, 60, 80]

        print("\n=========={}==========".format(dataset_name))
        describe_dataset(dataset)
        # Save the complete dataset to file
        # save_dataset(dataset, os.path.join('data/processed/template01', dataset_name))

        # for portion in portions:
        #     tmp_dataset = random.sample(dataset, round(size * portion / 100))
        #     tmp_dataset_name = '{}_{}.txt'.format(dataset_name, portion)
        #     print("\n=========={}==========".format(tmp_dataset_name))
        #     describe_dataset(tmp_dataset)
        #     save_dataset(tmp_dataset, os.path.join('data/processed/', tmp_dataset_name))


def build_dataset_templates():
    original = {
        "conll03_train": "data/original/CoNLL03/train.txt",
        "conll03_devel": "data/original/CoNLL03/devel.txt",
        # "conll03_test": "data/original/CoNLL03/test.txt",

        "conll04_train": "data/original/CoNLL04/train.txt",
        "conll04_devel": "data/original/CoNLL04/devel.txt",
        # "conll04_test": "data/original/CoNLL04/test.txt",
    }
    templates = [
        {
            "pos": "<token_span> is <mask> .",
            "neg": "<token_span> is not an entity .",
        },
        {
            "pos": "TL;DR , <token_span> is <mask> entity .",
            "neg": "TL;DR , <token_span> is not a named entity .",
        },
        {
            "pos": "<token_span> is <mask> entity , right ? Yes .",
            "neg": "<token_span> is not a named entity, right ? Sure ."
        },
        {
            "pos": "<token_span> can be a named entity , and it's <mask> entity .",
            "neg": "<token_span> cannot be a named entity .",
        },
        {
            "pos": "What entity would you call <token_span> ? <mask> .",
            "neg": "Is <token_span> a named entity ? No .",
        },
        {
            "pos": "<token_span> == <mask> .",
            "neg": "<token_span> == None .",
        },
        {
            "pos": "<token_span> is <mask> entity .",
            "neg": "<token_span> is not a named entity ."
        },
        {
            "pos": "The entity type of <token_span> is <mask> .",
            "neg": "The entity type of <token_span> is none entity ."
        },
        {
            "pos": "<token_span> belongs to <mask> category .",
            "neg": "<token_span> belongs to none category ."
        },
        {
            "pos": "<token_span> should be tagged as <mask> .",
            "neg": "<token_span> should be tagged as none entity ."
        },
        {
            "pos": "::: <token_span> == <mask> :::",
            "neg": "::: <token_span> == none :::"
        },
        {
            "pos": "<u1> <u2> <u3> <token_span> <u4> <u5> <u6> <mask> <u7> <u8> <u9> <u10>",
            "neg": "<u1> <u2> <u3> <token_span> <u4> <u5> <u6> <u7> <u8> <mask> <u9> <u10>"
        },
    ]

    for i in range(9, 10):
        template = templates[i]
        for dataset_name, filepath in original.items():
            dataset = get_input_examples(filepath)
            print("\n=========={}==========".format(dataset_name))
            describe_dataset(dataset)
            save_dataset(dataset,
                         # os.path.join('data/processed/template0{}'.format(i+1), dataset_name),
                         os.path.join('data/processed/template10', dataset_name),
                         template)
            portions = [20, 40, 60, 80]
            size = len(dataset)
            for portion in portions:
                tmp_dataset = random.sample(dataset, round(size * portion / 100))
                tmp_dataset_name = '{}_{}.txt'.format(dataset_name, portion)
                print("\n=========={}==========".format(tmp_dataset_name))
                describe_dataset(tmp_dataset)
                save_dataset(dataset,
                             os.path.join('data/processed/template10', tmp_dataset_name),
                             template)


# if __name__ == '__main__':
#     build_dataset_templates()
