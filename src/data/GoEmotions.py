from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

from src.utils.preprocess import tokenize


class GoEmotionsDataset(Dataset):
    def __init__(self, config, split='train'):
        if split == 'train':
            self.dataset = load_dataset(
                'go_emotions', cache_dir='src/resources/emotions', split='train[:80%]')
        elif split == 'val':
            self.dataset = load_dataset(
                'go_emotions', cache_dir='src/resources/emotions', split='train[80%:100%]')
        elif split == 'text':
            self.dataset = load_dataset(
                'go_emotions', cache_dir='src/resources/emotions', split='test')
        else:
            print(
                "Please choose one of the following ['train', 'val', 'test']")

        self.encodings = tokenize(config, self.dataset['text'])
        self.num_labels = len(set(self.dataset['label']))

    def __getitem__(self, idx):
        data = self.dataset[idx]
        label = data['meta_label']
        encoding = self.encodings[idx]
        ids = torch.tensor(encoding.ids)
        attention = torch.tensor(encoding.attention_mask)
        type_ids = torch.tensor(encoding.type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return attention, ids, type_ids, label

    def __len__(self):
        return self.pdf.shape[0]
