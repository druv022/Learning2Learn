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


class AGNewsNLI(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
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
        self.extended_labels = {i:' This text is about '+ i for i in self.dataset.features['label'].names}
        self.ext_labels_encodings = tokenize(config, list(self.extended_labels.values()))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = data['text']
        true_label = data['label']
        encoding = self.encodings[idx]

        concat_ids = []
        concat_attn = []
        concat_type_ids = []
        concat_label = []
        for i, label in enumerate(self.extended_labels):
            concat_ids.append(torch.tensor(encoding.ids+self.ext_labels_encodings[i].ids))
            concat_attn.append(torch.tensor(encoding.attention_mask + self.ext_labels_encodings[i].attention_mask))
            concat_type_ids.append(torch.tensor(encoding.type_ids + self.ext_labels_encodings[i].type_ids))
            label_id = 1 if self.dataset.features['label'].names[true_label] == label else 0
            concat_label.append(torch.tensor(label_id))
        
        concat_ids = torch.stack(concat_ids)
        concat_attn = torch.stack(concat_attn)
        concat_type_ids = torch.stack(concat_type_ids)
        concat_label = torch.stack(concat_label)
        return concat_attn, concat_ids, concat_type_ids, concat_label
