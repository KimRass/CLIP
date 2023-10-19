from transformers import DistilBertTokenizerFast


def train_tokenizer(corpus, vocab_size, vocab_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=vocab_size, length=len(corpus))
    tokenizer.save_pretrained(vocab_dir)


def load_tokenizer(vocab_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(vocab_dir)
    return tokenizer
