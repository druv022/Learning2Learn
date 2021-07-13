import yaml

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import classification_report, accuracy_score
from src.data.agnews import AGNewsNLI
from src.data.dbpedia_14 import DBPedai14NLI
from src.data.yahoo_answers import YahooAnswers14NLI
from src.data.yelp_review import YelpReview14NLI
from src.data.utils import pad_input

dataset_dic={'ag_news':AGNewsNLI,'dbpedia_14':DBPedai14NLI,'yahoo_ans':YahooAnswers14NLI,'yelp_review':YelpReview14NLI}


def test_meta_data(model, config, device,writer,task_steps,type):
    model.eval()
    pred_labels = []
    target_labels = []
    print("Begin testing")
    sample_task_name = config['test_meta_dataset']
    test_dataset = dataset_dic[sample_task_name](config, split='test', sample_size=config['test_num_samples'])
    test_test_loader= DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                                  num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                                  collate_fn=pad_input)
    for attention, ids, type_ids, label in tqdm(test_test_loader):
        input_ids = ids.to(device)
        attention_mask = attention.to(device)
        type_ids = type_ids.to(device)
        labels = label.to(device)
        outputs = model.forward(attention_mask, input_ids, type_ids)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        value, indices = torch.max(pred, dim=1)
        pred_labels.extend(indices.cpu().tolist())
        target_labels.extend(labels.cpu().tolist())

    acc=accuracy_score(target_labels, pred_labels)
    report=classification_report(target_labels, pred_labels)
    writer.add_scalar('accuracy ' + sample_task_name+' '+type,
                      acc, task_steps[3])
    writer.add_text('report' + sample_task_name+' '+type,
                    report, task_steps[3])

def meta_test_train(config, model, writer, iteration, niterations, task_steps, device=torch.device("cpu")):
    test_meta_data(model,config,device,writer,task_steps,'zero')
    model.train()
    loss = nn.CrossEntropyLoss()
    outerstepsize0 = 0.1
    weights_before = deepcopy(model.state_dict())
    sample_task_name = config['test_meta_dataset']
    print("Sampled TaSK Test", sample_task_name)

    optimizer_ft = optim.SGD([
        {'params': model.module.bert.parameters()},
        {'params': model.module.pre_score_layer.parameters(), 'lr': 2e-2},
        {'params': model.module.score_layer.parameters(), 'lr': 2e-2}
    ], lr=1e-5)
    steps = 0
    model.train()
    data_loss = 0
    train_dataset = dataset_dic[sample_task_name](config, split='train', sample_size=config['train_num_samples'])
    test_train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"],
                                  num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                                  collate_fn=pad_input)
    for attention, input_id, token_id, label in tqdm(test_train_loader):
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
        if (steps > 0 and steps % config["loss_plot_test_step"] == 0):
            print(data_loss / config["loss_plot_test_step"])
            writer.add_scalar(f'loss/task' + sample_task_name, data_loss /
                              config["loss_plot_test_step"], task_steps[3])
            data_loss = 0
        steps = steps + 1
        task_steps[3] = task_steps[3] + 1
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * \
        (1 - iteration / niterations)  # linear schedule
    model.load_state_dict({name:
                           weights_before[name] + (weights_after[name] -
                                                   weights_before[name]) * outerstepsize
                           for name in weights_before})
    test_meta_data(model, config, device,writer,task_steps,'trained')
    model.train()
    print("Testing Done")
