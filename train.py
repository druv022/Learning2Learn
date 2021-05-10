from pathlib import Path
import yaml

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AdamW

from src.model.bertclassifier2 import BertClassification
from src.data.IMDB import IMDBDataset as Dataset
from src.data.IMDB import read_imdb_split
from src.utils.preprocess import tokenize
from test import test

CUDA_LAUNCH_BLOCKING=1

def train(config, model, train_texts, train_labels, val_texts, val_labels, device=torch.device("cpu")):
    # tokenize
    train_encodings = tokenize(config, train_texts)
    # prepare dataset and data loader
    train_dataset = Dataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], \
                                num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    optim = AdamW(model.parameters(), lr=config["lr"])

    print("Begin training:...")
    for epoch in range(config["epochs"]):
        # set model to train
        model.train()

        epoch_loss = []
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())
            print(loss.item())
        
        epoch_loss = np.mean(epoch_loss)
        print(f"Epoch:{epoch}\tEpoch_loss: {epoch_loss}")
        # TODO: add all to tensorboardx

        # call validation on the model
        test(model, config, val_texts, val_labels, device)



if __name__ == "__main__":
    with open('./src/config/params.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # read 
    train_texts, train_labels = read_imdb_split(config["train"])
    # train test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=config["split"])

    # build model
    model = BertClassification(config["modelname"])
    model.to(device)
    # train
    train(config, model, train_texts, train_labels, val_texts, val_labels, device)
