from pathlib import Path
import yaml

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from src.model.bertclassifier import BertClassification
from src.data.IMDB import IMDBDataset as Dataset
from src.data.IMDB import read_imdb_split
from src.utils.preprocess import tokenize
from test import test
import torch.nn as nn

def train(config, model, train_df,val_df,test_df, device=torch.device("cpu")):
    writer=SummaryWriter()
    train_text=train_df['lext'].tolist()
    # tokenize
    train_encodings = tokenize(config,train_text)
    # prepare dataset and data loader
    train_dataset = Dataset(train_encodings, train_df)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], \
                                num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    optim1 = AdamW(
        model.parameters(), lr=2e-5, correct_bias=False)
    optimizer_ft = optim1
    ##weights=torch.FloatTensor([0.3,1]).cuda()
    loss = nn.CrossEntropyLoss()
    training_steps = int((config['epochs'] * train_df.shape[0]) / config['batch_size'])
    num_warmup_steps = 0  ##int(training_steps*0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer_ft, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=training_steps)
    steps = 0
    model.train()
    data_loss=0
    while (steps < training_steps):
        for attention, input_id, token_id, label in tqdm(train_loader):
            if (device == "cuda:0"):
                attention = attention.cuda()
                input_id = input_id.cuda()
                token_id = token_id.cuda()
                label = label.cuda()
            logits = model.forward(attention, input_id, token_id)
            ##class_cat=class_cat.unsqueeze(1)
            model_loss = loss(logits, label)
            ##print(model_loss)
            optimizer_ft.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_ft.step()
            data_loss = data_loss + model_loss.item()
            if (steps % 10 == 0):
                writer.add_scalar("loss", data_loss / 10, steps)
                data_loss = 0
            if(steps%30==0):
                acc=test(model,config,val_df,"cpu")
                writer.add_scalar("accuracy", acc, steps)
                model.train()




if __name__ == "__main__":
    with open('./src/config/params.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Done reading data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # read 
    train_df,val_df,test_df = read_imdb_split(config["data"])
    # train test split
    num_labels=len(train_df.label.unique())
    # build model
    model = BertClassification(config,num_labels=num_labels)
    model.to(device)
    # train
    train(config, model, train_df,val_df,test_df, device)
