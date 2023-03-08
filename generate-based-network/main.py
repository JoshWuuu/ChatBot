from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import csv
import random
import os
import codecs
from io import open

from json_preprocess import *
from data_cleaning import *
from text_to_matrix import *
from model_build import *
from model_train import *
from model_eval import *
from argparse import ArgumentParser

import config

def main():
    
    parser = ArgumentParser()
    print("\nchecking device..")
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")    
    # take a look on the data 
    save_dir = os.path.join("data", "save")
    corpus_name = "movie-corpus"
    corpus = os.path.join("data", corpus_name)

    print("\nchecking some samples in utterances.jsonl..")
    printLines(os.path.join(corpus, "utterances.jsonl"))

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict and conversations dict
    lines = {}
    conversations = {}
    # Load lines and conversations
    print("\nProcessing corpus into lines and conversations...")
    lines, conversations = linesAndConversations(os.path.join(corpus, "utterances.jsonl"))

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)

    print('Preparing training data')
    voc, pairs = readVocs(datafile, corpus_name)
    MAX_LENGTH = 10 # Maximum sentence length to consider
    print('Filter sentence pairs with MAX_LENGTH: {}'.format(MAX_LENGTH))
    filtered_pairs = filterPairs(pairs, MAX_LENGTH)
    print('Filtered ratio (short sentense) {} / {} : {:.4f}'.format(
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

    # Print some pairs to validate
    print("\nchecking some trimmed pairs:")
    for pair in trimmed_pairs[:10]:
        print(pair)  

    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc,  [random.choice(trimmed_pairs) for _ in range(small_batch_size)])

    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("\nchecking training data:")
    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    # Configure models
    print("\nconfiguring the model:")
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    #loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

        # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, config.hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
    decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, voc.num_words, config.decoder_n_layers, config.dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    print("\nmodel training..!")
    # Run training iterations
    model_train(config.model_name, voc, trimmed_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, config.encoder_n_layers, config.decoder_n_layers, save_dir, config.n_iteration, config.batch_size,
                config.print_every, config.save_every, config.hidden_size, config.clip, corpus_name, loadFilename, device, config.teacher_forcing_ratio)

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    
    # Load model
    checkpoint = torch.load('data/save/cb_model/movie-corpus/2-2_500/4000_checkpoint.tar', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
  
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    print("model evaluation..!")
    # Initialize search module
    searcher = GreedySearchDecoderEvaluation(encoder, decoder)
    print("start chatting!")
    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, voc, config.device)

if __name__ == "__main__":
    main()