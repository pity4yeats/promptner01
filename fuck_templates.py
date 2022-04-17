# Helper script to avoid runtime-restart in colab run.
# Get Eval F1 of All Templates on Full Data Scenario for Both CoNLL2003 and CoNLL2004.

# Accept two arguments:
#   @dataset: string - conll04
#   @template: string - template06

# Usage:
#   python fuck_templates.py conll03 template02

'''

I have four machines, each running one method. Each one holding 10 training and 10 inference.

That'll do for the other template's results.
'''


# --->Training<---
# @dataset_name: conll03_20.txt, conll04_80.txt, conll04_100.txt
# @template: template03, template05

# --->Inference<---



from prompt_model import *
import sys

epochs = 5
batch_size = 32
# dataset = 'conll04'   # modify this to load another dataset
# template_path = './data/processed/template06'   # modify this to use different templates

dataset = sys.argv[1]
template = sys.argv[2]

template_path = os.path.join('./data/processed/', template)
output_dir = os.path.join('./saved_models', dataset, template)
best_model_dir = os.path.join('./saved_models', dataset, template, 'best_model')
train_dataset = os.path.join(template_path, '{}_train'.format(dataset))
devel_dataset = os.path.join(template_path, '{}_devel'.format(dataset))

print('\n\n\n\n\n--------------------------------\nTraining: {}/{}\n'.format(template_path, dataset))
train(batch_size, epochs, output_dir, best_model_dir, train_dataset, devel_dataset)
