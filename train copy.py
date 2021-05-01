from pathlib import Path
import yaml

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AdamW, Trainer, TrainingArguments

from src.model.bertclassifier2 import BertClassification
from src.data.IMDB import IMDBDataset as Dataset
from src.data.IMDB import read_imdb_split
from src.utils.preprocess import tokenize
from test import test

CUDA_LAUNCH_BLOCKING=1

def train(config, model, train_texts, train_labels, val_texts, val_labels, device=torch.device("cpu")):
    # tokenize
    train_encodings = tokenize(config, train_texts)
    val_encodings = tokenize(config, val_texts)

    # prepare dataset and data loader
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    print("Begin training:...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

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
