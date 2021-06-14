import os
import pickle

from datasets import load_dataset
import datasets
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from src.utils.preprocess import tokenize

class AGNews(Dataset):
    def __init__(self, config, split='train'):
        if split == 'train':
            self.dataset = load_dataset(
                'ag_news', cache_dir='src/resources/ag_news', split='train[:80%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'ag_news', cache_dir='src/resources/ag_news', split='train[80%:100%]')
        elif split == 'test':
            self.dataset = load_dataset(
                'ag_news', cache_dir='src/resources/ag_news', split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        self.encodings = tokenize(config, self.dataset['text'])
        self.num_labels = len(set(self.dataset['label']))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = data['text']
        label = data['label']
        encoding = self.encodings[idx]
        ids = torch.tensor(encoding.ids)
        attention = torch.tensor(encoding.attention_mask)
        type_ids = torch.tensor(encoding.type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return attention, ids, type_ids, label


class AGNewsNLI(Dataset):
    def __init__(self, config, split='train'):
        self.split = split
        self.config = config
        self.dump_file_path = os.path.join(self.config['cache_dir'], self.split +'_dump.pkl')
        
        if split == 'train':
            self.dataset = load_dataset(
                'ag_news', cache_dir=self.config['cache_dir'], split='train[:80%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'ag_news', cache_dir=self.config['cache_dir'], split='train[80%:100%]')
        elif split == 'test':
            self.dataset = load_dataset(
                'ag_news', cache_dir=self.config['cache_dir'], split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        self.num_labels = len(set(self.dataset['label']))

        self._save_and_load()
        self.encodings = tokenize(config, self.new_text)

    def _save_and_load(self):
        if os.path.exists(self.dump_file_path):
            with open(self.dump_file_path, 'rb') as f:
                data = pickle.load(f)
            self.new_text = data['text']
            self.new_labels = data['label']
            self.num_labels = data['num_labels']
        else:
            self.extended_labels = {i:' This text is about '+ i.lower() for i in self.dataset.features['label'].names}
            self.new_text = []
            self.new_labels = []
            for idx, text in tqdm(enumerate(self.dataset['text'])):
                label = self.dataset['label'][idx]
                for label2, label_with_text in self.extended_labels.items():
                    self.new_text.append(text+label_with_text)
                self.new_labels.append(label)
            data = {'text': self.new_text, 'label': self.new_labels, 'num_labels': self.num_labels}
            with open(self.dump_file_path, 'wb') as f:
                pickle.dump(data, f)
            del data

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
