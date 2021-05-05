from torch import nn
from transformers import BertTokenizerFast, BertModel, BertConfig,BertPreTrainedModel, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch

class BertClassification(nn.Module):
    def __init__(self, config, num_labels=2, hidden_dropout_prob=0.1):
        super(BertClassification,self).__init__()
        self.bert = BertModel.from_pretrained(config['modelname'])
        self.config = BertConfig.from_pretrained(config['modelname'])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, attention, input, tokens):
        output= self.bert(input_ids=input, attention_mask=attention, token_type_ids=tokens)
        last_state = torch.mean(output.last_hidden_state, dim=1)
        drop_output = self.dropout(last_state)
        print(drop_output.shape)
        logits = self.classifier(drop_output)
        return logits
