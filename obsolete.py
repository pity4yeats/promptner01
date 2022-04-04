import os
# os.chdir('/content/drive/MyDrive/promptner/code_dir')
import time
import math
import pandas
import logging
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from utils_metrics import *
from seq2seq_model import Seq2SeqModel
from tqdm import tqdm, trange
epochs = 3
batch_size = 32
output_dir = './saved_models'
# code_dir = os.path.dirname(os.path.realpath('.'))
code_dir = '~/PycharmProjects/promptner/'

use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('./best_model')


class InputExample:
    def __init__(self, tokens_list, labels):
        self.tokens_list = tokens_list
        self.labels = labels


def train(train_data, devel_data):
    train_dataset = os.path.join(code_dir, train_data)
    devel_dataset = os.path.join(code_dir, devel_data)

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    raw_training_dataset = pandas.read_csv(train_dataset, sep=',').values.tolist()
    raw_evaluation_dataset = pandas.read_csv(devel_dataset, sep=',').values.tolist()
    training_dataset = pandas.DataFrame(raw_training_dataset, columns=["input_text", "target_text"])
    evaluation_dataset = pandas.DataFrame(raw_evaluation_dataset, columns=["input_text", "target_text"])

    # print(training_dataset)
    # print(evaluation_dataset)
    # exit(0)

    model_args = {
        "reprocess_input_data": True, "overwrite_output_dir": True, "use_multiprocessing": False,
        "max_seq_length": 50, "train_batch_size": batch_size, "num_train_epochs": epochs,
        "save_eval_checkpoints": True, "save_model_every_epoch": True,
        "evaluate_during_training": True, "evaluate_generated_text": True, "evaluate_during_training_verbose": True,
        "max_length": 25, "manual_seed": 4, "save_steps": 11898, "gradient_accumulation_steps": 1,
        "output_dir": output_dir,
    }

    model = Seq2SeqModel(
        # encoder_type='roberta',
        # encoder_name='roberta-base',
        # decoder_name='roberta-base',
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        use_cuda=use_cuda
    )

    model.train_model(training_dataset, eval_data=evaluation_dataset)


def get_input_examples(file):
    input_examples = []
    with open(file, 'r', encoding='utf-8') as test_data:
        tokens_list = []
        labels = []
        for line in test_data:
            if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                if tokens_list:
                    input_examples.append(InputExample(tokens_list=tokens_list, labels=labels))
                    tokens_list = []
                    labels = []
            else:
                words = line.split(' ')
                tokens_list.append(words[0])
                labels.append(words[-1].replace('\n', '')) if len(words) > 1 else labels.append('O')

        if tokens_list:
            input_examples.append(InputExample(tokens_list=tokens_list, labels=labels))
    return input_examples


# get the ngram set from kth index of given tokens_list(sentence)
def ngram(tokens_list, k):
    n = 9
    kgram = []
    for i in range(1, min(n, len(tokens_list) - k + 1)):
        kgram.append(' '.join(tokens_list[k:k + i]))
    return kgram


# predict
def predict_kernel(tokens_spans, tokens_list, igram):
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # model = BartForConditionalGeneration.from_pretrained('./outputs/best_model')
    ner_tags = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
    template_tails = [
        " is a location entity .",
        " is a person entity .",
        " is an organization entity .",
        " is an other entity .",
        " is not a named entity ."
    ]
    num_tails = len(template_tails)
    num_spans = len(tokens_spans)

    model.to(device)

    inputs = ' '.join(tokens_list[:])
    inputs = [inputs] * (num_tails * num_spans)
    input_ids = tokenizer(inputs, return_tensors='pt')['input_ids']
    igram_prompts = []
    for i in range(num_spans):
        for j in range(num_tails):
            igram_prompts.append(tokens_spans[i] + template_tails[j])
    igram_prompts_input_ids = tokenizer(igram_prompts, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # TODO what does this do?
    # refer: https://huggingface.co/docs/transformers/main/en/model_doc/bart#transformers.BartTokenizer
    # 2 here stands for </s> token, means using separator token to replace the initial token of every prompt
    igram_prompts_input_ids[:, 0] = 2

    # zero-initialize a list the same length of the prompts list
    tmp = [0] * num_tails * num_spans
    # Iterate over all igram_prompts
    for i in range(len(igram_prompts) // num_tails):
        base_len = (
                       (tokenizer(igram_prompts[i * 5], return_tensors='pt', padding=True, truncation=True)[
                           'input_ids']).shape
                   )[1] - 4
        tmp[i * 5: i * 5 + 5] = [base_len] * 5
        tmp[i * 5 + 4] += 1

    score = [1] * 5 * num_spans
    decoder_input_ids = igram_prompts_input_ids[:, :igram_prompts_input_ids.shape[1] - 2]
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=decoder_input_ids.to(device)).logits
        for i in range(igram_prompts_input_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 5 * num_spans):
                if i < tmp[j]:
                    score[j] = score[j] * logits[j][int(igram_prompts_input_ids[j][i + 1])]
    end = igram + (score.index(max(score)) // num_tails)
    return [igram, end, ner_tags[(score.index(max(score)) % 5)], max(score)]


def predict(tokens_list):
    predicted_labels = []
    i = 0
    # from the first item in tokens_list(sentence), predict
    for i in range(len(tokens_list)):
        ith_ngram = ngram(tokens_list, i)
        # print('Just finished extracting {}-gram.'.format(igram))
        # print(f'{tokens_spans=}')
        entities = predict_kernel(ith_ngram, tokens_list, i)
        # print(f'{entities=}')
        if entities[1] >= len(tokens_list):
            entities[1] = len(tokens_list) - 1
        if entities[2] != 'O':
            predicted_labels.append(entities)

    if len(predicted_labels) > 1:
        while i < len(predicted_labels):
            j = i + 1
            while j < len(predicted_labels):
                if (predicted_labels[i][1] < predicted_labels[j][0]) or (
                        predicted_labels[i][0] > predicted_labels[j][1]):
                    j += 1
                else:
                    if predicted_labels[i][3] < predicted_labels[j][3]:
                        predicted_labels[i], predicted_labels[j] = predicted_labels[j], predicted_labels[i]
                        predicted_labels.pop(j)
                    else:
                        predicted_labels.pop(j)
            i += 1
    results = ['O'] * len(tokens_list)
    for label in predicted_labels:
        results[label[0]: label[1] + 1] = ['I-' + label[2]] * (label[1] - label[0] + 1)
        results[label[0]] = 'B-' + label[2]
    return results


def inference(test_data):
    input_examples = get_input_examples(test_data)

    targets = []
    predicts = []

    i = 0
    print('Running inference...')
    pbar = tqdm(total=len(input_examples))
    for input_example in input_examples:
        predicts.append(predict(input_example.tokens_list))
        targets.append(input_example.labels)
        # print(f'{input_example.tokens_list=}')
        # print(f'{predicts[i]=}')
        # print(f'{targets[i]=}')
        i += 1
        pbar.update(1)
    pbar.close()

    targets = get_entities_bio(targets)
    predicts = get_entities_bio(predicts)

    results = {
        'acc': precision_score(targets, predicts),
        'rec': recall_score(targets, predicts),
        'f1': f1_score(targets, predicts),
    }
    for k, v in results.items():
        print('{}: {}'.format(k, v))


if __name__ == '__main__':
    conll03_train_file = 'data/tmp/conll03_train.csv'
    conll03_devel_file = 'data/tmp/conll03_devel.csv'
    conll03_test_file = 'data/tmp/conll03_test.txt'
    conll04_train_file = 'data/tmp/conll04_train.csv'
    conll04_devel_file = 'data/tmp/conll04_devel.csv'
    conll04_test_file = 'data/tmp/conll04_test.txt'

    # print(os.getcwd())
    train(conll04_train_file, conll04_devel_file)

    # model.eval()
    # model.config.use_cache = False
    # inference(conll04_test_file)
