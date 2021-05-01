from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

from src.model.bertclassifier2 import BertClassification
from src.utils.preprocess import tokenize
from src.data.IMDB import IMDBDataset as Dataset


def test(model, config, test_texts, test_labels, device=torch.device("cpu")):
    model.eval()
    test_encodings = tokenize(config, test_texts)

    test_dataset = Dataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pred_labels= []
    print("Begin testing")
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        pred = torch.nn.functional.softmax(outputs[1], dim=-1)
        value, indices = torch.max(pred, dim=-1)

        pred_labels.extend(indices.cpu().tolist())

    print(accuracy_score(test_labels, pred_labels))
    print(classification_report(test_labels, pred_labels))




