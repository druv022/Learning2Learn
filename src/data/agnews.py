from datasets import load_dataset
import datasets
from torch.utils.data import Dataset
import torch

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
