# Helper script to avoid runtime-restart in colab run.
# Get Eval F1 of All Templates on Full Data Scenario for Both CoNLL2003 and CoNLL2004.

# Accept two arguments:
#   @dataset: string - conll04
#   @template: string - template06

# Usage:
#   python templates_train.py conll03 template02

"""
I have four machines, each running one method. Each one holding 10 training and 10 inference.
That'll do for the other template's results.
"""
# --->Training<---
# @dataset_name: conll03_20.txt, conll04_80.txt, conll04_100.txt
# @template: template03, template05

# --->Inference<---

from prompt_model import *
import sys

dataset = sys.argv[1]
template = sys.argv[2]

epochs = 5
batch_size = 32
portions = [20, 40, 60, 80, 100]
template_path = os.path.join('./data/processed/', template)

for portion in portions:
    portion = str(portion)
    output_dir = os.path.join('./saved_models', dataset, template, portion)
    best_model_dir = os.path.join(output_dir, 'best_model')
    train_dataset = os.path.join(template_path, '{}_train_{}.txt'.format(dataset, portion))
    devel_dataset = os.path.join(template_path, '{}_devel_{}.txt'.format(dataset, portion))

    print('\n\n\n\n\n--------------------------------\nTraining: {}/{}/{}\n'.format(dataset, template, portion))
    train(batch_size, epochs, output_dir, best_model_dir, train_dataset, devel_dataset)
