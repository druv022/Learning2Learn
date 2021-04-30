from torch import nn
from transformers import BertTokenizerFast, BertModel, BertConfig,BertPreTrainedModel, BertForSequenceClassification, Trainer, TrainingArguments, AdamW

class BertClassification(nn.Module):
    def __init__(self, config, num_labels=2, hidden_dropout_prob=0.1, hidden_size=768):
        super().__init__()
        # self.num_labels = config.num_labels
        self.num_labels = num_labels

        self.config = BertConfig()
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits