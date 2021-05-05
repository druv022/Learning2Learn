from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

class IMDBDataset(Dataset):
    def __init__(self, encodings,pdf):
        self.pdf=pdf
        self.encodings=encodings

    def __getitem__(self, idx):
        data=self.pdf.iloc[idx]
        label=data['label']
        encoding=self.encodings[idx]
        ids=torch.tensor(encoding.ids)
        attention=torch.tensor(encoding.attention_mask)
        type_ids=torch.tensor(encoding.type_ids)
        label=torch.tensor(label,dtype=torch.long)
        return attention,ids,type_ids, label

    def __len__(self):
        return self.pdf.shape[0]


def read_imdb_split(split_dir):
    print("Reading IMDB dataset")
    df=pd.read_csv(split_dir)
    train_df=df.loc[df['data_for']=='train']
    val_df=df.loc[df['data_for']=='validation']
    test_df = df.loc[df['data_for'] == 'test']
    return train_df,val_df,test_df