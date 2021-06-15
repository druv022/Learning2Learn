from torch import nn
from transformers import BertTokenizerFast, BertModel, BertConfig, BertPreTrainedModel, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch


class BertClassification(nn.Module):
    def __init__(self, config, num_labels=2, hidden_dropout_prob=0.1):
        super(BertClassification, self).__init__()
        self.bert = BertModel.from_pretrained(config['modelname'])
        self.config = BertConfig.from_pretrained(config['modelname'])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.pre_score_layer = nn.Linear(self.config.hidden_size, 48)
        self.score_layer = nn.Linear(48, 1)

    def forward(self, attention, input, tokens):
        num_labels=attention.shape[1]
        batch_size=attention.shape[0]
        attention=attention.reshape(attention.shape[0]*attention.shape[1],attention.shape[2])
        input = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
        tokens=tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2])
        output = self.bert(
            input_ids=input, attention_mask=attention, token_type_ids=tokens)
        last_state=output.pooler_output
        ##last_state = torch.mean(output.last_hidden_state, dim=1)
        drop_output = self.dropout(last_state)
        pre_score = torch.tanh(self.pre_score_layer(drop_output))
        score=self.score_layer(pre_score)
        score=score.reshape(batch_size,num_labels)
        return score
