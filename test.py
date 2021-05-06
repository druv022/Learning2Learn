from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

from src.model.bertclassifier2 import BertClassification
from src.utils.preprocess import tokenize
from src.data.IMDB import IMDBDataset as Dataset


def test(model, config, data_df, device=torch.device("cuda")):
    model.eval()
    data_text = data_df['lext'].tolist()
    data_encodings = tokenize(config, data_text)

    test_dataset = Dataset(data_encodings, data_df)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    pred_labels= []
    target_labels=[]
    print("Begin testing")
    for attention,ids,type_ids, label in tqdm(test_loader):
        input_ids = ids.to(device)
        attention_mask = attention.to(device)
        type_ids=type_ids.to(device)
        labels = label.to(device)
        outputs =model.forward(attention_mask, input_ids, type_ids)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        value, indices = torch.max(pred, dim=1)
        pred_labels.extend(indices.cpu().tolist())
        target_labels.extend(labels.cpu().tolist())

    return accuracy_score(target_labels, pred_labels)
    ##print(classification_report(target_labels, pred_labels))




