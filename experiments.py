import transformers
from prompt_model import *

portions = [20, 40, 60, 80]
epochs = 3
batch_size = 32


# Train CoNLL2003
for portion in portions:
    output_dir = os.path.join('./saved_models', 'conll2003_{}'.format(portion))
    train_dataset = os.path.join('./data/processed', 'conll03_train_{}.txt'.format(portion))
    devel_dataset = os.path.join('./data/processed', 'conll03_devel_{}.txt'.format(portion))
    print('\n==========Training CoNLL2003_{}=========='.format(portion))
    train(batch_size, epochs, output_dir, train_dataset, devel_dataset)

# Inference CoNLL2003
for portion in portions:
    test_dataset = os.path.join('./data/original/CoNLL03/test.txt')
    model_path = os.path.join('./saved_models', 'conll03_{}'.format(portion))
    model = transformers.BartForConditionalGeneration.from_pretrained(model_path)
    print('\n==========Evaluating CoNLL2003_{}=========='.format(portion))
    evaluate(model, test_dataset)

# Train CoNLL2004
for portion in portions:
    output_dir = os.path.join('./saved_models', 'conll2004_{}'.format(portion))
    train_dataset = os.path.join('./data/processed', 'conll04_train_{}.txt'.format(portion))
    devel_dataset = os.path.join('./data/processed', 'conll04_devel_{}.txt'.format(portion))
    print('\n==========Training CoNLL2004_{}=========='.format(portion))
    train(batch_size, epochs, output_dir, train_dataset, devel_dataset)

# Inference CoNLL2004
for portion in portions:
    test_dataset = os.path.join('./data/original/CoNLL04/test.txt')
    model_path = os.path.join('./saved_models', 'conll04_{}'.format(portion))
    model = transformers.BartForConditionalGeneration.from_pretrained(model_path)
    print('\n==========Evaluating CoNLL2004_{}=========='.format(portion))
    evaluate(model, test_dataset)
