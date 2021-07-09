import os
import pickle
import itertools

from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from src.utils.preprocess import Tokenizer, trim_text


class YahooAnswers(Dataset):
    def __init__(self, config, split='train', sample_size=-1):
        self.config = config

        if split == 'train':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='train[:95%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='train[95%:]')
        elif split == 'test':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        if sample_size < 0:
            self.sample_size = len(self.dataset)
        else:
            self.sample_size = sample_size if len(self.dataset) > sample_size else len(self.dataset)

        tokenizer = Tokenizer(self.config)
        self.encodings = tokenizer.tokenize(self.dataset['best_answer'][0:self.sample_size])
        self.num_labels = len(set(self.dataset['topic']))

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = data['best_answer']
        label = data['topic']
        encoding = self.encodings[idx]
        ids = torch.tensor(encoding.ids)
        attention = torch.tensor(encoding.attention_mask)
        type_ids = torch.tensor(encoding.type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return attention, ids, type_ids, label


class YahooAnswers14NLI(Dataset):
    def __init__(self, config, split='train', sample_size=-1):
        self.split = split
        self.config = config

        if split == 'train':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='train[:95%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='train[95%:]')
        elif split == 'test':
            self.dataset = load_dataset(
                'yahoo_answers_topics', cache_dir=self.config['yahoo_cache_dir'], split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")
        
        if sample_size < 0:
            self.sample_size = len(self.dataset)
        else:
            self.sample_size = sample_size if len(self.dataset) > sample_size else len(self.dataset)

        self.num_labels = len(set(self.dataset['topic']))
        self.extended_labels = {i: config['prepend_topic'] + i.lower() if i.lower() not in config['yahoo_remapping'] else
                                config['prepend_topic'] + config['yahoo_remapping'][i.lower()] for i in self.dataset.features['topic'].names}
        self.extended_labels={}
        for i in self.dataset.features['topic'].names:
            str=config['prepend_topic']
            if(i in config['yahoo_remapping']):
                str=str+config['yahoo_remapping'][i]
            else:
                str = str+[i]
            self.extended_labels[i]=str

        trim_length = config['max_text_length']['yahoo_ans'] - 20

        self.label_text = list(
            self.extended_labels.values()) * len(self.dataset['best_answer'][0:sample_size])

        self.new_text = [i for i in itertools.chain.from_iterable(itertools.repeat(
            trim_text(x, trim_length), len(self.extended_labels)) for x in self.dataset['best_answer'][0:self.sample_size])]
        self.new_labels = self.dataset['topic'][0:self.sample_size]

        self.tokenizer = Tokenizer(self.config,config['max_text_length']['yahoo_ans'])

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        new_idx = idx*self.num_labels
        concat_ids = []
        concat_attn = []
        concat_type_ids = []
        encodings = self.tokenizer.tokenize(
            self.new_text[new_idx:new_idx+self.num_labels], self.label_text[new_idx:new_idx+self.num_labels])
        for i in range(0, self.num_labels):
            encoding = encodings[i]
            concat_ids.append(torch.tensor(encoding.ids))
            concat_attn.append(torch.tensor(encoding.attention_mask))
            concat_type_ids.append(torch.tensor(encoding.type_ids))

        label = torch.tensor(self.new_labels[idx])
        concat_ids = torch.stack(concat_ids)
        concat_attn = torch.stack(concat_attn)
        concat_type_ids = torch.stack(concat_type_ids)
        return concat_attn, concat_ids, concat_type_ids, label
