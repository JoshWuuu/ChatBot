from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import unicodedata
from io import open
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter
from torchtext.vocab import Vocab
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

spacy_eng = spacy.load("en_core_web_sm")

class PairsDataset(Dataset):
    def __init__(self, pairs, vocab) -> None:
        super().__init__()
        self.pairs = pairs
        self.vocab = vocab
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input = [self.vocab[token] for token in pair[0]]
        target = [self.vocab[token] for token in pair[1]]

        return torch.tensor(input), torch.tensor(target)

def tokenize_eng(text):
    """
    tokenize_ger(text) -> list of tokens, using spacy_eng tokenizer

    Input:
    - text: string
    """
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def unicodeToAscii(s):
    """
    if word count below a certain count threshold, rm it
        
    Input:
    - min_count: int
    Returns:
    """
    return "".join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    lowercase, trim, and rm non-letter characters
        
    - s
    Returns:
    """
    # strip remove all the leading and trailing space 
    s = unicodeToAscii(s.lower().strip())
    s = re.compile("[.;:!\'?,\"()\[\]]").sub("", s) # remove punctuation
    s = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", s) # remove punctuation
    lemmatizer = WordNetLemmatizer()
    s = ' '.join([lemmatizer.lemmatize(word) for word in s.split()])
    return s

def readLines(datafile):
    """
    lowercase, trim, and rm non-letter characters
        
    Input:
    - datafile: string, path to datafile

    Returns:
    - voc: obj
    - pairs: list, [[pair of conversation]]
    """
    print('Reading lines')
    # splits the file into lines
    lines = open(datafile, encoding ='utf-8').read().strip().split('\n')
    # splits the lines into the pairs
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    return pairs

def tokenize_pairs(pairs):
    """
    lowercase, trim, and rm non-letter characters
        
    Input:
    - s

    Returns:
    - voc: obj
    - pairs: list, [[pair of conversation]]
    """
    print('Tokenizing pairs')
    # splits the file into lines
    pairs = [[['<SOS>'] + tokenize_eng(s) + ['<EOS>'] for s in line] for line in pairs]
    return pairs

def vocab_from_pairs(pairs):
    """
    build vocab from pairs

    Input:
    - pairs: list, [[pair of conversation]]

    Returns:
    - vocab: obj
    """
    print('Building vocab')
    counter = Counter()
    for lines in tqdm(pairs):
        for word in lines[0]:
            counter.update((word,))
        for word in lines[1]:
            counter.update((word,))
    vocab = Vocab(counter, min_freq=5)
    return vocab

def textToIndex(text, vocab):
    """
    transform text to index

    Input:
    - text: list, [token]
    - vocab: obj
    """
    indexes = [vocab[token] for token in text]
    return indexes

def indexToText(indexes, vocab):
    """
    transform index to text

    Input:
    - indexes: list, [index]
    - vocab: obj
    """
    new_itos = vocab.itos
    text = [new_itos[index] for index in indexes]
    return text

def collate_batch(batch):
    """
    padding the batch in dataloader function

    Input:
    - batch: list, [[input, target]]
    """
    target_list, input_list = [], []
    for input, target in batch:
        input_list.append(input)
        target_list.append(target)
    return pad_sequence(input_list, padding_value=0.0), pad_sequence(target_list, padding_value=0.0)

def batch_sampler_fn(pairs, batch_size):
    """
    batch sampler for the dataloader, to make sure the length of the sentence in the batch is similar

    Input:
    - pairs: list, [[pair of conversation]]
    - batch_size: int
    """
    indices = [(i, len(line[0])) for i, line in enumerate(pairs)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths 
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

def main():
    save_dir = os.path.join("data", "save")
    corpus_name = "movie-corpus"
    MAX_LENGTH = 10  # Maximum sentence length to consider
    corpus = os.path.join("data", corpus_name)
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    print('Preparing training data')
    pairs = readLines(datafile)
    pairs = tokenize_pairs(pairs)
    vocab = vocab_from_pairs(pairs)
    
    print("The length of the new vocab is", len(vocab))
    new_stoi = vocab.stoi
    print("The index of a is", new_stoi['a'])
    new_itos = vocab.itos
    print("The token at index 3 is", new_itos[3])
    print('the token for different indexes', new_itos[0:150])

    pairdataset = PairsDataset(pairs, vocab)
    print("dataset[0]:", pairdataset[0])
    train_dataloader = DataLoader(pairdataset, batch_sampler=batch_sampler_fn(pairs, 1), 
                              collate_fn=collate_batch)
    
    for line, target in train_dataloader:
        print("dataloader input index: ", line.view(-1, 1).squeeze(1).tolist())
        print("dataloader input string: ",indexToText(line.view(-1, 1).squeeze(1).tolist(), vocab))
        print("dataloader target index: ", target.view(-1, 1).squeeze(1).tolist())
        print("dataloader target string: ", indexToText(target.view(-1, 1).squeeze(1).tolist(), vocab))
        break

if __name__ == '__main__':
    main()