from transformers import BertTokenizerFast


class Tokenizer:

    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])

    def tokenize(self, texts, label_text=None):
        if label_text:
            encodings = self.tokenizer(
                texts, label_text, truncation=self.config["truncation"], padding=self.config["padding"], max_length=256)
        else:
            encodings = self.tokenizer(
                texts, truncation=self.config["truncation"], padding=self.config["padding"], max_length=256)
        return encodings


def trim_text(text, trim_length):
    text_arr = text.split(" ")
    if(len(text_arr) > trim_length):
        text_arr = text_arr[:trim_length]
        st = ''
        for word in text_arr:
            st = st+word+' '

        st = st.strip()
        return st
    else:
        return text
