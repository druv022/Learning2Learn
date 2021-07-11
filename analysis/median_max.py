from datasets import load_dataset
import datasets
import yaml
from transformers import BertTokenizerFast
import numpy as np


with open('./src/config/params.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

dataset = {'ag_news':['agnews_cache_dir','text'], 'yahoo_answers_topics':['yahoo_cache_dir','best_answer'],'dbpedia_14':['dbpedia_cache_dir','content'],'yelp_review_full':['yelp_cache_dir','text']}
tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])
for name, cache in dataset.items():
    dataset = load_dataset(
                    name, cache_dir=config[cache[0]])

    train_texts = dataset['train'][cache[1]]
    test_texts = dataset['test'][cache[1]]

    train_texts_2 = [i.split(' ') for i in train_texts]
    test_texts_2 = [i.split(' ') for i in test_texts]

    lengths_train = [len(i.split(' ')) for i in train_texts]
    lengths_test = [len(i.split(' ')) for i in test_texts]

    print("###############################################")
    print(name)
    print(f'Train simple median:{np.median(np.asarray(lengths_train))}; max: {max(lengths_train)}')
    print(f'Test simple median:{np.median(np.asarray(lengths_test))}; max: {max(lengths_test)}')

    print('Using tokenizer')
    train_texts_3= tokenizer.batch_decode(tokenizer(train_texts)['input_ids'])
    test_texts_3 = tokenizer.batch_decode(tokenizer(test_texts)['input_ids'])

    lengths_train_2 = [len(i.split(' ')) for i in train_texts_3]
    lengths_test_2 = [len(i.split(' ')) for i in test_texts_3]

    print(f'Train bert median:{np.median(np.asarray(lengths_train_2))}; max: {max(lengths_train_2)}')
    print(f'Test bert median:{np.median(np.asarray(lengths_test_2))}; max: {max(lengths_test_2)}')