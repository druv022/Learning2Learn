from transformers import BertTokenizerFast

def tokenize(config, texts):
    tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])

    encodings = tokenizer(texts, truncation=config["truncation"], padding=config["padding"])

    return encodings

