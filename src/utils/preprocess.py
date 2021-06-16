from transformers import BertTokenizerFast

def tokenize(config, texts,label_text):
    tokenizer = BertTokenizerFast.from_pretrained(config["modelname"])
    encodings = tokenizer(texts, label_text,truncation=config["truncation"], padding=config["padding"],max_length=128)
    return encodings

def trim_text(text):
    text_arr=text.split(" ")
    if(len(text_arr)>110):
        text_arr=text_arr[:110]
        st=''
        for word in text_arr:
            st=st+word+' '

        st=st.strip()
        return st
    else:
        return text

