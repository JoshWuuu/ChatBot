from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

def indexesFromSentence(voc, sentence):
    """
    index the sentence
        
    Input:
    - voc: obj
    - sentence: str

    Returns:
    - list of list, with index of word in each sentence
    """
    EOS_token = 2
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=0):
    """
    combine the value of the corresponding index, and fill 0 if the length is different
    zip_longest act as transpose method for the matrix
        
    Input:
    - l: list, index list
    - fillvalue: int, default 0

    Returns:
    - same shape of list in list with each value of string in same index of the sentences
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    """
    padding mask, 1 no padding 0 padding
        
    Input:
    - l: list, index list + fillvalue 0
    - sentence: str

    Returns:
    - m: list, list o padding mask
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    """
    return padded input seqence tensor and lengths
        
    Input:
    - l: list, list of sentence
    - voc: doc

    Returns:
    - padVar: torch tensor, sentence index
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    """
    return padded target seqence tensor and max lengths
        
    Input:
    - l: list, list of sentence
    - voc: doc

    Returns:
    - padVar: torch tensor, sentence index
    - mask: torch boolean, mask for the input
    - max_target_len: int, max length of the target sentence
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    """
    return all items for a batch of pairs
        
    Input:
    - voc: doc
    - pair_batch: list, list of pair sentence
    
    Returns:
    - input_tensor, input_length, output_tensor, output_mask, output_max_length: param of all above funcs
    """
    pair_batch.sort(key = lambda x: len(x[0].split(" ")), reversed = True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        input_tensor, input_length = inputVar(input_batch, voc)
        output_tensor, output_mask, output_max_length = outputVar(output_batch, voc)
        return input_tensor, input_length, output_tensor, output_mask, output_max_length

def main():
    
    corpus_name = "movie-corpus"
    MAX_LENGTH = 10  # Maximum sentence length to consider
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    print('Preparing training data')
    voc, pairs = readVocs(datafile, corpus_name)
    print('Filter sentence pairs with MAX_LENGTH: {}'.format(MAX_LENGTH))
    filtered_pairs = filterPairs(pairs)
    print('Filtered ratio (shorted sentense) {} / {} : {:.4f}'.format(
        len(filtered_pairs), len(pairs), len(filtered_pairs) / len(pairs)
    ))
    print('Put word into object')
    for pair in filtered_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Count words:', voc.num_words)

    MIN_COUNT = 3
    # Trim voc and pairs
    trimmed_pairs = trimRareWords(voc, filtered_pairs, MIN_COUNT)   
    print('Trimmed ratio (rare word) {} / {} : {:.4f}'.format(
        len(trimmed_pairs), len(filtered_pairs), len(trimmed_pairs) / len(filtered_pairs)
    ))

    # Example for validation
    batch_size = 5
    batches = batch2TrainData(voc,  [random.choice(trimmed_pairs) for _ in range(small_batch_size)])

    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

if __name__ == "__main__":
    main()