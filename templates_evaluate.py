import transformers
from prompt_model import *

dataset = sys.argv[1]
template = sys.argv[2]

portions = [20, 40, 60, 80, 100]

if dataset == 'conll03':
    test_dataset = os.path.join('./data/original/CoNLL03/test.txt')
elif dataset == 'conll04':
    test_dataset = os.path.join('./data/original/CoNLL04/test.txt')

for portion in portions:
    portion = str(portion)
    model_path = os.path.join('./saved_models', dataset, template, portion)
    model = transformers.BartForConditionalGeneration.from_pretrained(model_path)
    print('\n\n\n\n\n--------------------------------\nEvaluating: {}/{}/{}\n'.format(template, dataset, portion))
    evaluate(model, test_dataset)
