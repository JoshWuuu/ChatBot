from torchtext.legacy.datasets import Multi30k
# from torchtext.data import Field, BucketIterator
from collections import Counter
from torchtext.vocab import Vocab
import config


def tokenize_ger(text):
    """
    tokenize_ger(text) -> list of tokens, using spacy_ger tokenizer

    Input:
    - text: string
    """
    return [tok.text for tok in config.spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    """
    tokenize_ger(text) -> list of tokens, using spacy_eng tokenizer

    Input:
    - text: string
    """
    return [tok.text for tok in config.spacy_eng.tokenizer(text)]

def data_preprocessing():
    """
    data preprocessing, field, tokenization
    build vocab, build vocabularies,
    bucket iterator, dataloader for vocab corpus

    Input:
    - text: string
    """

    train_iter = Multi30k(split='train')
    counter = Counter()
    for (label, line) in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=10, specials=('', '', '', ''))
    
    # german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

    # english = Field(
    #     tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
    # )
    # print(german, english)
    # train_data, valid_data, test_data = Multi30k.splits(
    #     exts=(".de", ".en"), fields=(german, english)
    # )
    # print(train_data)
    # german.build_vocab(train_data, max_size=10000, min_freq=2)
    # english.build_vocab(train_data, max_size=10000, min_freq=2)
    # print(german.vocab)
    # print(train_data)

    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    #     batch_size=config.batch_size,
    #     sort_within_batch=True,
    #     sort_key=lambda x: len(x.src),
    #     device=config.device,
    # )
    # iterator, for loop, batch.de and batch.en

    return train_iterator, valid_iterator, test_iterator

def main():
    train_iterator, valid_iterator, test_iterator = data_preprocessing()
    print("Done")

if __name__ == "__main__":
    main()