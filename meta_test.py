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
import torch.optim as optim
import torch.nn as nn
import random
from copy import deepcopy
from sklearn.metrics import classification_report, accuracy_score


def test_meta_data(model, test_df,config,device):
    model.eval()
    data_text = test_df['lext'].tolist()
    data_encodings = tokenize(config, data_text)

    test_dataset = Dataset(data_encodings, test_df)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    pred_labels = []
    target_labels = []
    print("Begin testing")
    for attention, ids, type_ids, label in tqdm(test_loader):
        input_ids = ids.to(device)
        attention_mask = attention.to(device)
        type_ids = type_ids.to(device)
        labels = label.to(device)
        outputs = model.forward(attention_mask, input_ids, type_ids)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        value, indices = torch.max(pred, dim=1)
        pred_labels.extend(indices.cpu().tolist())
        target_labels.extend(labels.cpu().tolist())

    return accuracy_score(target_labels, pred_labels), classification_report(target_labels, pred_labels)

def meta_test_train(config, model,df,writer,iteration,niterations,task_steps,device=torch.device("cpu")):
    loss = nn.CrossEntropyLoss()
    outerstepsize0 = 0.1
    weights_before = deepcopy(model.state_dict())
    sample_task = 5
    print("Sampled TaSK Test", sample_task)
    task_data = df.loc[df['task_id'] == sample_task].head(450)
    train_text = task_data['lext'].tolist()
    train_encodings = tokenize(config, train_text)
    optimizer_ft = optim.SGD([
        {'params': model.bert.parameters()},
        {'params': model.classifier.parameters(), 'lr': 2e-2}
    ], lr=1e-5)
    steps = 0
    model.train()
    data_loss = 0
    train_dataset = Dataset(train_encodings, task_data)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], \
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    for attention, input_id, token_id, label in tqdm(train_loader):
        if (torch.cuda.is_available()):
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
        if (steps > 0 and steps % config["loss_plot_test_step"] == 0):
            print(data_loss / config["loss_plot_test_step"])
            writer.add_scalar(f'loss/task' + str(sample_task), data_loss / config["loss_plot_test_step"], task_steps[sample_task])
            data_loss = 0
        steps = steps + 1
        task_steps[sample_task] = task_steps[sample_task] + 1
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations)  # linear schedule
    model.load_state_dict({name:
                               weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                           for name in weights_before})
    acc,report=test_meta_data(model,df.tail(50),config,device)
    writer.add_scalar('accuracy' + str(sample_task), acc, task_steps[sample_task])
    writer.add_text('report' + str(sample_task), report, task_steps[sample_task])
    model.train()
    print("Testing Done")




