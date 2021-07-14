from pathlib import Path
import yaml

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from src.model.bertclassifier import BertClassification
from src.data.agnews import AGNewsNLI
from src.data.dbpedia_14 import DBPedai14NLI
from src.data.yahoo_answers import YahooAnswers14NLI
from src.data.yelp_review import YelpReview14NLI
import torch.optim as optim
import torch.nn as nn
import random
from copy import deepcopy
from meta_test import meta_test_train
from sklearn.utils import shuffle
from src.data.utils import pad_input
import torch

dataset_dic={'ag_news':AGNewsNLI,'dbpedia_14':DBPedai14NLI,'yahoo_ans':YahooAnswers14NLI,'yelp_review':YelpReview14NLI}

def meta_train(config, model, device=torch.device("cpu")):
    writer = SummaryWriter()
    loss = nn.CrossEntropyLoss()
    dataset_data={}
    outerstepsize0 = 0.1
    niterations = 50000
    total_tasks=len(config['max_text_length'])
    task_steps={}
    for i in range(0,total_tasks):
        task_steps[i]=0
    total_steps = 0
    for iteration in range(0, niterations):
        weights_before = deepcopy(model.state_dict())
        sample_task_index = random.randint(0, 2)
        sample_task_name=config['train_meta_dataset'][sample_task_index]
        ##print("Sampled Task", config['train_meta_dataset'])
        optimizer_ft = optim.SGD([
            {'params': model.parameters()}
        ], lr=2e-5)
        steps = 0
        model.train()
        data_loss = 0
        if(sample_task_name not in dataset_data):
            train_dataset = dataset_dic[sample_task_name](config, split='train', sample_size=1000)
            dataset_data[sample_task_name]=train_dataset
        train_dataset=dataset_data[sample_task_name]
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                                  num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                                  collate_fn=pad_input)
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
            if (steps > 0 and steps % config["loss_plot_step"] == 0):
                print(data_loss / config["loss_plot_step"])
                writer.add_scalar(f'loss/task'+sample_task_name, data_loss /
                                  config["loss_plot_step"], task_steps[sample_task_index])
                data_loss = 0
            steps = steps + 1
            task_steps[sample_task_index] = task_steps[sample_task_index]+1
        weights_after = model.state_dict()
        outerstepsize = outerstepsize0 * \
            (1 - iteration / niterations)  # linear schedule
        model.load_state_dict({name:
                               weights_before[name] + (weights_after[name] -
                                                       weights_before[name]) * outerstepsize
                               for name in weights_before})
        if (iteration > 0 and iteration % config["acc_plot_step"] == 0):
            meta_test_train(config, model, writer, iteration,
                            niterations, task_steps,dataset_data ,device)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    with open('./src/config/params.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Done reading data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # read
    # train test split
    num_labels = 5
    # build model
    model = BertClassification(config, num_labels=num_labels)
    model = nn.DataParallel(model)
    model.to(device)
    # train
    meta_train(config, model, device)
