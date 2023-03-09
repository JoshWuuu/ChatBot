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

spacy_eng = spacy.load("en_core_web_sm")

def tokenize_eng(text):
    """
    tokenize_ger(text) -> list of tokens, using spacy_eng tokenizer

    Input:
    - text: string
    """
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def readLines(datafile, corpus_name):
    """
    lowercase, trim, and rm non-letter characters
        
    Input:
    - s

    Returns:
    - voc: obj
    - pairs: list, [[pair of conversation]]
    """
    print('Reading lines')
    # splits the file into lines
    lines = open(datafile, encoding ='utf-8').read().strip().split('\n')
    # splits the lines into the pairs
    pairs = [[s for s in line.split('\t')] for line in lines]
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
    pairs = [[tokenize_eng(s) for s in line] for line in pairs]
    return pairs

def vocab_from_pairs(pairs):
    """
    
    Input:
    

    Returns:
    """
    print('Building vocab')
    counter = Counter()
    for (line1, line2) in pairs:
        counter.update(tokenize_eng(line1))
        counter.update(tokenize_eng(line2))
    vocab = Vocab(counter, min_freq=10, specials=('', '', '', ''))
    return vocab

def main():
    save_dir = os.path.join("data", "save")
    corpus_name = "movie-corpus"
    MAX_LENGTH = 10  # Maximum sentence length to consider
    corpus = os.path.join("data", corpus_name)
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    print('Preparing training data')
    pairs = readLines(datafile, corpus_name)
    vocab = tokenize_pairs(pairs)

    print("The length of the new vocab is", len(vocab))
    new_stoi = vocab.stoi
    print("The index of '' is", new_stoi[''])
    new_itos = vocab.itos
    print("The token at index 2 is", new_itos[2])

if __name__ == '__main__':
    main()