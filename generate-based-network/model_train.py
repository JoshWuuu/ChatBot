from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from model_build import LuongAttnDecoderRNN, EncoderRNN

from data_cleaning import Voc

from text_to_matrix import batch2TrainData
from config.config import config

import torch
import torch.nn as nn
from torch import optim
import random
import os

def maskNLLLoss(inp, target, mask, device):
    """   
    define the loss function for the model, the loss is calculated by the cross entropy loss with the padding mask
    
    Input
    - inp: (max_length, batch_size)
    - target: (max_length, batch_size)
    - mask: (max_length, batch_size)
    - device: str, device to run the model

    Returns
    - loss: tensor, loss of the model
    - nTotal: int, number of total words in the batch
    """
    nTotal_mask = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))    
    loss = crossEntropy.masked_select(mask).mean() 
    loss = loss.to(device)
    return loss, nTotal_mask.item()

def model_train(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, 
                print_every, save_every, hidden_size, clip, corpus_name, loadFilename, device, teacher_forcing_ratio=1.0):
    """   
    model train pipeline
    
    Input
    - model_name: str, name of the model
    - voc: obj, vocabulary
    - pairs: list, list of sentence pairs
    - encoder: obj, encoder model
    - decoder: obj, decoder model
    - encoder_optimizer: obj, encoder optimizer
    - decoder_optimizer: obj, decoder optimizer
    - embedding: obj, embedding model
    - encoder_n_layers: int, number of layers of encoder
    - decoder_n_layers: int, number of layers of decoder
    - save_dir: str, directory to save the model
    - n_iteration: int, number of iteration
    - batch_size: int, batch size
    - print_every: int, print loss every n iteration
    - save_every: int, save model every n iteration
    - hidden_size: int, hidden size of the model
    - clip: float, gradient clipping
    - corpus_name: str, name of the corpus
    - loadFilename: str, name of the model to load
    - device: str, device to run the model
    - teacher_forcing_ratio: float, ratio of teacher forcing

    Returns
    """

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

    print("Initializing ...")
    start_iter = 1
    acc_loss = 0
    if loadFilename:
        start_iter = checkpoint['iteration'] + 1
    
    print("Training...")
    for iteration in range(start_iter, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_tensor, input_length, output_tensor, output_mask, output_max_length = training_batch

        # Run the train function
        # loss = train(input_tensor, input_length, output_tensor, output_mask, output_max_length, encoder, decoder,
        #                 embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
        mask = output_mask.to(device)
        input_length = input_length.to("cpu")

        temp_loss = 0
        print_losses = []
        n_totals = 0

        # forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_length)

        # create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[0 for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # determine if we are using teacher forcing this iteration
        teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # forward batch of sequences through decoder one time step at a time
        if teacher_forcing:
            for t in range(output_max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Teacher forcing: next input is current target
                decoder_input = output_tensor[t].view(1, -1)
                # Calculate and accumulate loss per word in the batch
                mask_loss, nTotal = maskNLLLoss(decoder_output, output_tensor[t], mask[t], device)
                temp_loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(output_max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss per word in the batch
                mask_loss, nTotal = maskNLLLoss(decoder_output, output_tensor[t], mask[t], device)
                temp_loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        
        loss = temp_loss
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of loss
        loss = temp_loss / n_totals
        
        acc_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = acc_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            acc_loss = 0
        
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

            print("Model is saved!")
def main():
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
    embedding = nn.Embedding(Voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 1
    save_every = 500

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
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

    # Run training iterations
    model_train(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, loadFilename, device, teacher_forcing_ratio)


if __name__ == '__main__':
    main()