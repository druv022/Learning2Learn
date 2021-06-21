import torch
import torch.nn.functional as F


def pad_input(batch):
    attention, input_id, token_id, label = [], [], [], []
    max_length = max([len(item[0][0]) for item in batch])
    for item in batch:
        pad_length = max_length - len(item[0][0])
        attention.append(F.pad(item[0], (0,pad_length)))
        input_id.append(F.pad(item[1], (0,pad_length)))
        token_id.append(F.pad(item[2], (0,pad_length)))
        label.append(item[3])

    attention = torch.stack(attention)
    input_id = torch.stack(input_id)
    token_id = torch.stack(token_id)
    label = torch.stack(label)

    return attention, input_id, token_id, label