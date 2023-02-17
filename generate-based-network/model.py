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

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layer= 1, dropout = 0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layer
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, 
        dropout=(0 if n_layer == 1 else dropout), bidirectional = True)
    
    def forward(self, input_seq, input_lengths, hidden = None):
        """   
        return shape:
        - outputs: (max_length, batch_size, hidden states)
        - hidden: (n_layer, num_directions, batch_size, hidden_size)
        """
        # Convert word indexes to embeddings
        embedding = self.embedding(input_seq)
         # Pack padded batch of sequences for RNN module, to improve computational efficiency
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
         # Return output and final hidden state
        return outputs, hidden

# attention layer: 
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        # encoder output = (max_length, batch_size, hidden states)
        # hidden = (1, batch_size, hidden_state)
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        
    def dot_score(self, hidden, encoder_output):
        # return [max_length, batch size] weight for each encoder output
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        # return [max_length, batch size]
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        # return [max_length, batch size]
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        """   
        return shape:
        - [batch size, 1, max_length]
        """
        # Calculate the attention weights (energies) based on the given method
        # [max_length, batch size]
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        # [batch size, 1, max_length]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    """
    class for decoder
        
    Input:
    - attn_model
    - embedding: (batch_size, hidden_size)
    - hidden_size: int
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """   
        return shape:
        - input_step: (1, batch_size)
        - last_hidden: (n_layer, num_directions, batch_size, hidden_size)
        - encoder_outputs: (max_length, batch_size, hidden_size)
        - output: (batch_size, voc.num_words)
        - hidden: (n_layer, num_directions, batch_size, hidden_size)
        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded) # (1, batch_size, hidden_state)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden) # rnn_output: (1, batch_size, hidden_state)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs) # rnn_output: (1, batch_size, hidden_state), (max_length, batch_size, hidden_size)
        # attn_weights [batch size, 1, max_length]
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context (batch_size, 1, hidden state)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0) # rnn_output: (batch_size, hidden_state)
        context = context.squeeze(1) # context: (batch_size, hidden_state)
        concat_input = torch.cat((rnn_output, context), 1) # concat_input: (batch_size, 2* hidden_state)
        concat_output = torch.tanh(self.concat(concat_input)) # concat_input : (batch_size, hidden_state)
        # Predict next word using Luong eq. 6
        output = self.out(concat_output) 
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden