from transformers import DistilBertTokenizerFast


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer
