from transformers import BertTokenizerFast


class Tokenizer:

    def __init__(self, config,max_length) -> None:
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])
        self.max_length=max_length

    def tokenize(self, texts, label_text=None):
        print("Max lenght",self.max_length)
        if label_text:
            encodings = self.tokenizer(
                texts, label_text, truncation=self.config["truncation"], padding=self.config["padding"], max_length=self.max_length)
        else:
            encodings = self.tokenizer(
                texts, truncation=self.config["truncation"], padding=self.config["padding"], max_length=self.max_length)
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
