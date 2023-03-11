from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import os

from json_preprocess import *
from data_cleaning import *
from model_build import *
from model_train import *
from model_eval import *
from config import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print('the token for 0:150 indexes', new_itos[0:10])

    print('Constructing the dataset and dataloader...')
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
    
    print('Configuring the model...')

    load_model = True
    save_model = True

    # Training hyperparameters
    num_epochs = 100
    learning_rate = 3e-4
    batch_size = 64

    model_name = 0
    # Model hyperparameters
    src_vocab_size = len(vocab)
    trg_vocab_size = len(vocab)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 100
    forward_expansion = 4
    src_pad_idx = vocab.stoi["<pad>"]

    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)

    train_dataloader = DataLoader(pairdataset, batch_sampler=batch_sampler_fn(pairs, batch_size), 
                              collate_fn=collate_batch)
    
    print('Training the model...')
    model_train(model_name, model, vocab, learning_rate, num_epochs, train_dataloader, save_dir,
                corpus_name, hidden_size, device)

    print('Chatbot...')
    model.eval()
    evaluateInput(model, vocab, device)

if __name__ == "__main__":
    main()