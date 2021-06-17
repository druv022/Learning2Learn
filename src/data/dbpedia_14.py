import os
import pickle
import itertools

from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from src.utils.preprocess import tokenize, trim_text


class DBPedia14(Dataset):
    def __init__(self, config, split='train'):
        if split == 'train':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='train[:80%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='train[80%:100%]')
        elif split == 'test':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        self.encodings = tokenize(config, self.dataset['content'])
        self.num_labels = len(set(self.dataset['label']))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = data['content']
        label = data['label']
        encoding = self.encodings[idx]
        ids = torch.tensor(encoding.ids)
        attention = torch.tensor(encoding.attention_mask)
        type_ids = torch.tensor(encoding.type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return attention, ids, type_ids, label


class DBPedai14NLI(Dataset):
    def __init__(self, config, split='train'):
        self.split = split
        self.config = config

        if split == 'train':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='train[:90%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='train[90%:]')
        elif split == 'test':
            self.dataset = load_dataset(
                'dbpedia_14', cache_dir=self.config['cache_dir'], split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        self.num_labels = len(set(self.dataset['label']))

        self.extended_labels = {i: config['prepend'] + i.lower() if i.lower() != 'sci/tech' else
                                config['prepend'] + 'science or technology' for i in self.dataset.features['label'].names}
        self.label_text = list(
            self.extended_labels.values()) * len(self.dataset['content'])
        
        self.new_text = [i for i in itertools.chain.from_iterable(itertools.repeat(
            trim_text(x), len(self.extended_labels)) for x in self.dataset['content'])]
        self.new_labels = self.dataset['label']

        self.encodings = tokenize(config, self.new_text, self.label_text)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        new_idx = idx*self.num_labels
        concat_ids = []
        concat_attn = []
        concat_type_ids = []
        for i in range(new_idx, new_idx+self.num_labels):
            encoding = self.encodings[i]
            concat_ids.append(torch.tensor(encoding.ids))
            concat_attn.append(torch.tensor(encoding.attention_mask))
            concat_type_ids.append(torch.tensor(encoding.type_ids))

        label = torch.tensor(self.new_labels[idx])
        concat_ids = torch.stack(concat_ids)
        concat_attn = torch.stack(concat_attn)
        concat_type_ids = torch.stack(concat_type_ids)
        return concat_attn, concat_ids, concat_type_ids, label
