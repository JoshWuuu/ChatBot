from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from data_cleaning import *

import spacy

import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score

def generate_response(model, sentence, vocab, device, max_length=50):
    """
    generate response for chat bot
    Input:
    - model: model object
    - sentence: input sentence 
    - vocab: vocabulary object
    - device: device
    - max_length: max length of sentence

    Returns:
    - response: str, response sentence
    """
    # normalize the string
    sentence = normalizeString(sentence)
    # tokenize the string
    tokens = ['<SOS>'] + tokenize_eng(sentence) + ['<EOS>'] 
    # convert to tensor
    indexes = textToIndex(tokens, vocab)
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(indexes).unsqueeze(0).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [vocab.stoi["<EOS>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            cur_response = output.argmax(1).item()

        outputs.append(cur_response)

        # Model predicts it's the end of the sentence
        if cur_response == vocab.stoi["<eos>"]:
            break

    translated_sentence = indexToText(outputs, vocab)

    # remove start token and end token
    return translated_sentence[1:-1]


def bleu(pairs, model, vocab, device):
    """
    calculate bleu score
    Input:
    - pairs: list, list of str pairs
    - model: model object
    - vocab: vocabulary object
    - device: device

    Returns:
    - bleu score: float
    """
    targets = []
    outputs = []

    for input, target in pairs:
        prediction = generate_response(model, input, vocab, device)

        targets.append([target])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def evaluateInput(model, vocab, device):
    """
    input sentence and get the response
        
    Input:
    - model: model object
    - vocab: vocabulary object
    - device: device

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
            # Evaluate sentence
            output_words = generate_response(model, input_sentence, vocab, device)
            print('Bot:', ' '.join(output_words))
        
        except KeyError:
            print("Error: Encountered unknown word.")

def main():
    pass

if __name__ == '__main__':
    main()
    
