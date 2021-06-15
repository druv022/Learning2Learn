from pathlib import Path
import yaml
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from src.model.bertclassifier import BertClassification
# from src.data.GoEmotions import GoEmotionsDataset as Dataset
from src.data.agnews import AGNewsNLI as Dataset

from src.utils.preprocess import tokenize
from test import test
import torch.nn as nn
import time


def train(config, model, train_dataset, val_dataset,test_dataset, device=torch.device("cpu")):
    writer = SummaryWriter()
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    optim1 = AdamW(
        model.parameters(), lr=2e-5, correct_bias=False)
    optimizer_ft = optim1
    # weights=torch.FloatTensor([0.3,1]).cuda()
    loss = nn.CrossEntropyLoss()
    training_steps = int(
        (config['epochs'] * len(train_dataset)) / config['batch_size'])
    num_warmup_steps = 0  # int(training_steps*0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer_ft, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=training_steps)
    steps = 0
    model.train()
    data_loss = 0
    while (steps < training_steps):
        for attention, input_id, token_id, label in tqdm(train_loader):
            if (torch.cuda.is_available()):
                attention = attention.cuda()
                input_id = input_id.cuda()
                token_id = token_id.cuda()
                label = label.cuda()

            logits = model.forward(attention, input_id, token_id)
            # class_cat=class_cat.unsqueeze(1)
            model_loss = loss(logits, label)
            # print(model_loss)
            optimizer_ft.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_ft.step()
            data_loss = data_loss + model_loss.item()
            if (steps > 0 and steps % 100 == 0):
                writer.add_scalar("loss", data_loss / 100, steps)
                data_loss = 0
            if(steps > 0 and steps % 500 == 0):
                acc, report = test(model, val_dataset)
                writer.add_scalar("validation_accuracy", acc, steps)
                writer.add_text("validation_classification_report", str(report), steps)
                model.train()
            steps = steps+1
    acc, report = test(model, test_dataset)
    writer.add_scalar("test_accuracy", acc, steps)
    writer.add_text("test_classification_report", str(report), steps)
    time.sleep(100)





def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Dataset(config, split='train')
    val_dataset = Dataset(config, split='val')
    test_dataset=Dataset(config,split='test')

    num_labels = train_dataset.num_labels

    model = BertClassification(config, num_labels=num_labels)
    model.to(device)

    train(config, model, train_dataset, val_dataset,test_dataset, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='./src/config/params.yml', help='Path to config file')
    args = parser.parse_args()

    main(args)
