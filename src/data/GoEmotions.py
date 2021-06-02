from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

from src.utils.preprocess import tokenize


class GoEmotionsDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        if split == 'train':
            self.dataset = load_dataset(
                'go_emotions', cache_dir='src/resources/emotions', split='train')
        elif split == 'text':
            self.dataset = load_dataset(
                'go_emotions', cache_dir='src/resources/emotions', split='test')
        else:
            print(
                "Please choose one of the following ['train', 'test']")
        
        self._prepare_meta_dataset()
        

    def _prepare_meta_dataset(self):
        text = self.dataset['text']

        self.process_dataset = []
        for i, label in enumerate(self.dataset['labels']):
            for l in label:
                self.process_dataset.append((l, text[i]))

    def set_dataset(self, task=-1):
        if task == -1:
            self.new_dataset = self.dataset
        elif task in self.config["task_mapping"]:
            label_list = self.config["task_mapping"][task]
            texts = []
            meta_labels = []
            for label, text in self.process_dataset:
                if label in label_list:
                    texts.append(text)
                    meta_labels.append(self.config["meta_mapping"][label])
            
            self.new_dataset = {'text': texts, 'labels': meta_labels}
        else:
            print(f"Wrong task number: {task}")
            return

        self.new_encodings = tokenize(self.config, self.new_dataset['text'])


    def __getitem__(self, idx):
        label = self.new_dataset['labels'][idx]
        encoding = self.new_encodings[idx]
        ids = torch.tensor(encoding.ids)
        attention = torch.tensor(encoding.attention_mask)
        type_ids = torch.tensor(encoding.type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return attention, ids, type_ids, label

    def __len__(self):
        return len(self.new_dataset['text'])
