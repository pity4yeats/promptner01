import torch
from tqdm import tqdm
import logging
import pandas

import seq2seq_model
from dataset_utils import *
from utils_metrics import *
from transformers import BartTokenizer

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def train(batch_size, epochs, output_dir, best_model_dir, train_dataset, devel_dataset):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_dataset = pandas.read_csv(train_dataset, sep=',').values.tolist()
    devel_dataset = pandas.read_csv(devel_dataset, sep=',').values.tolist()
    train_dataset = pandas.DataFrame(train_dataset, columns=["input_text", "target_text"])
    devel_dataset = pandas.DataFrame(devel_dataset, columns=["input_text", "target_text"])

    model_args = {
        "reprocess_input_data": True, "overwrite_output_dir": True, "use_multiprocessing": False,
        "max_seq_length": 50, "train_batch_size": batch_size, "num_train_epochs": epochs,
        "save_eval_checkpoints": False, "save_model_every_epoch": False, "best_model_dir": best_model_dir,
        "evaluate_during_training": True, "evaluate_generated_text": True, "evaluate_during_training_verbose": True,
        "max_length": 25, "manual_seed": 4, "save_steps": 11898, "gradient_accumulation_steps": 8,
        "output_dir": output_dir,
    }

    model = seq2seq_model.Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        use_cuda=use_cuda
    )

    model.train_model(train_dataset, eval_data=devel_dataset)


def evaluate(model, test_dataset):
    # @test_dataset: The path of test dataset
    # @model: The model prepared for inference
    # @tokenizer: The tokenizer to be used for inference
    dataset = get_input_examples(test_dataset)

    targets = []
    predicts = []

    pbar = tqdm(total=len(dataset))
    for step, data in enumerate(dataset):
        labels = data.labelseq.rstrip().split(' ')
        tokens = data.sentence.rstrip().split(' ')
        targets.append(labels)
        # The model and tokenizer are passed all the way down to the predict_kernel
        predicts.append(predict(model, tokens))
        pbar.update(1)
    pbar.close()

    # Convert sequence result for metric calculation
    true_entities = get_entities_bio(targets)
    pred_entities = get_entities_bio(predicts)

    metrics = {
        'acc': precision_score(true_entities, pred_entities),
        'rec': recall_score(true_entities, pred_entities),
        'f1': f1_score(true_entities, pred_entities),
    }

    for metric, value in metrics.items():
        print('{}: {}'.format(metric, value))


def predict(model, tokens):
    predicts = []
    i = 0
    for i in range(len(tokens)):
        ith_ngram = ngram(tokens, i)
        entities = predict_kernel(model, ith_ngram, tokens, i)
        # print(f'{entities=}')
        if entities[1] >= len(tokens):
            entities[1] = len(tokens) - 1
        if entities[2] != 'O':
            predicts.append(entities)

    if len(predicts) > 1:
        while i < len(predicts):
            j = i + 1
            while j < len(predicts):
                if (predicts[i][1] < predicts[j][0]) or (
                        predicts[i][0] > predicts[j][1]):
                    j += 1
                else:
                    if predicts[i][3] < predicts[j][3]:
                        predicts[i], predicts[j] = predicts[j], predicts[i]
                        predicts.pop(j)
                    else:
                        predicts.pop(j)
            i += 1
    results = ['O'] * len(tokens)
    for label in predicts:
        results[label[0]: label[1] + 1] = ['I-' + label[2]] * (label[1] - label[0] + 1)
        results[label[0]] = 'B-' + label[2]
    return results


def predict_kernel(model, ith_ngram, tokens, igram):
    ner_tags = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
    template_tails = [
        " is a location entity .",
        " is a person entity .",
        " is an organization entity .",
        " is an other entity .",
        " is not a named entity ."
    ]
    num_tails = len(template_tails)
    num_spans = len(ith_ngram)

    model.to(device)

    inputs = ' '.join(tokens[:])
    inputs = [inputs] * (num_tails * num_spans)
    input_ids = tokenizer(inputs, return_tensors='pt')['input_ids']
    igram_prompts = []
    for i in range(num_spans):
        for j in range(num_tails):
            igram_prompts.append(ith_ngram[i] + template_tails[j])
    igram_prompts_input_ids = tokenizer(igram_prompts,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True)['input_ids']
    # refer: https://huggingface.co/docs/transformers/main/en/model_doc/bart#transformers.BartTokenizer
    # 2 here stands for </s> token, means using separator token to replace the initial token of every prompt
    igram_prompts_input_ids[:, 0] = 2

    # Zero-initialize a list the same length of the prompts list
    tmp = [0] * num_tails * num_spans
    # Iterate over all igram_prompts
    for i in range(len(igram_prompts) // num_tails):
        igram_prompts_shape = (tokenizer(igram_prompts[i * 5],
                                         return_tensors='pt',
                                         padding=True,
                                         truncation=True
                                         )['input_ids']).shape
        base_len = igram_prompts_shape[1] - 4
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
