from pathlib import Path
from pandas.core.indexing import convert_from_missing_indexer_tuple
import yaml
import argparse
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from src.model.bertclassifier import BertClassification
from src.data.agnews import AGNewsNLI
from src.data.dbpedia_14 import DBPedai14NLI
from src.data.yahoo_answers import YahooAnswers14NLI
from src.data.yelp_review import YelpReview14NLI
from test import test
from src.data.utils import pad_input


def train(config, writer, model, train_dataset, val_dataset, test_dataset=None, device=torch.device("cpu")):
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"], collate_fn=pad_input)

    optim1 = AdamW(
        model.parameters(), lr=3e-5, correct_bias=False)
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
    best_acc = 0
    best_model = model
    model.train()
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
            if (steps > 0 and steps % 200 == 0):
                writer.add_scalar("loss", data_loss / 200, steps)
                data_loss = 0
            if(steps > 0 and steps % config['val_step'] == 0):
                acc, report = test(config, model, val_dataset, device=device)
                writer.add_scalar("validation_accuracy", acc, steps)
                writer.add_text(
                    "validation_classification_report", str(report), steps)

                if best_acc < acc:
                    best_model = model

            steps = steps+1

    time.sleep(100)
    return best_model


def test_model(config, model, writer, test_data, device):
    acc, report = test(config, model, test_data, device=device)
    writer.add_scalar("test_accuracy", acc, 0)
    writer.add_text("test_classification_report", str(report), 0)


def get_dataset(config, name, sample_size):
    if name == 'ag_news':
        train_dataset = AGNewsNLI(
            config, split='train', sample_size=sample_size)
        val_dataset = AGNewsNLI(config, split='val', sample_size=config['test_num_samples'])
        test_dataset = AGNewsNLI(config, split='test', sample_size=config['test_num_samples'])
    elif name == 'dbpedia_14':
        train_dataset = DBPedai14NLI(
            config, split='train', sample_size=sample_size)
        val_dataset = DBPedai14NLI(config, split='val', sample_size=config['test_num_samples'])
        test_dataset = DBPedai14NLI(config, split='test', sample_size=config['test_num_samples'])
    elif name == 'yahoo_ans':
        train_dataset = YahooAnswers14NLI(
            config, split='train', sample_size=sample_size)
        val_dataset = YahooAnswers14NLI(config, split='val', sample_size=config['test_num_samples'])
        test_dataset = YahooAnswers14NLI(config, split='test', sample_size=config['test_num_samples'])
    elif name == 'yelp_review':
        train_dataset = YelpReview14NLI(
            config, split='train', sample_size=sample_size)
        val_dataset = YelpReview14NLI(config, split='val', sample_size=config['test_num_samples'])
        test_dataset = YelpReview14NLI(config, split='test', sample_size=config['test_num_samples'])
    else:
        print("Please check the name of the dataset.")

    return train_dataset, val_dataset, test_dataset


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = get_dataset(
        config, config['dataset_name'], config['train_num_samples'])

    writer = SummaryWriter()
    num_labels = train_dataset.num_labels

    model = BertClassification(config, num_labels=num_labels)
    model.to(device)

    model = train(config, writer, model, train_dataset,
                  val_dataset, device=device)

    # test separately after saving the model
    test_model(config, model, writer, test_dataset, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='./src/config/params.yml', help='Path to config file')
    args = parser.parse_args()

    main(args)
