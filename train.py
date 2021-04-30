from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AdamW

from src.model.bertclassifier2 import BertClassification
from src.data.IMDB import IMDBDataset, read_imdb_split

CUDA_LAUNCH_BLOCKING=1

def train(config):
    # get dataset from line no. 16-27
    train_texts, train_labels = read_imdb_split(config["train"])

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=config["split"])

    tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])

    train_encodings = tokenizer(train_texts, truncation=config["truncation"], padding=config["padding"])
    val_encodings = tokenizer(val_texts, truncation=config["truncation"], padding=config["padding"])
    

    train_dataset = IMDBDataset(train_encodings, train_labels)
    val_dataset = IMDBDataset(val_encodings, val_labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertClassification(config["modelname"])
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])

    optim = AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()



if __name__ == "__main__":
    with open('./src/config/params.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(config=config)




    # test_texts, test_labels = read_imdb_split(args.path.test)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # test_dataset = IMDBDataset(test_encodings, test_labels)