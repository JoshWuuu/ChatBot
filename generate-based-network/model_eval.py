from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from text_to_matrix import indexesFromSentence

from data_cleaning import normalizeString
from model_build import LuongAttnDecoderRNN, EncoderRNN

import torch
import torch.nn as nn

class GreedySearchDecoderEvaluation(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(GreedySearchDecoderEvaluation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_seq, input_length, max_length, device):
        # Forwad input through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * 1
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(searcher, voc, sentence, device, max_length = 10):
    """
    response pipeline of the chatbot
        
    Input:
    - searcher: obj, GreedySearchDecoderEvaluation
    - voc: obj, Vocabulary
    - sentence: str, input sentence
    - device: obj, torch.device
    - max_length: int, max length of the sentence

    Returns:
    - decoded_words: list, list of words
    """
    # format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, device)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(searcher, voc, device):
    """
    input sentence and get the response
        
    Input:
    - searcher: obj, GreedySearchDecoderEvaluation
    - voc: obj, Vocabulary
    - device: obj, torch.device

    Returns:
    - None
    """
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('User: ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, device)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        
        except KeyError:
            print("Error: Encountered unknown word.")

def main():

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    
    # Load model
    checkpoint = torch.load('model/trained_model.tar')
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoderEvaluation(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc)

if __name__ == '__main__':
    main()
    
