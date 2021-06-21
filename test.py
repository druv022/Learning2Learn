from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from src.data.utils import pad_input


def test(config, model, test_dataset, device=torch.device("cuda")):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=config["pin_memory"], collate_fn=pad_input)

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
    ##print(classification_report(target_labels, pred_labels))
