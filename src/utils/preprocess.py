from transformers import BertTokenizerFast

def tokenize(config, texts,label_text):
    tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])
    encodings = tokenizer(texts, label_text,truncation=config["truncation"], padding=config["padding"],max_length=128)
    return encodings

