from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import random
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('TkAgg')
# from nltk.translate.bleu_score import corpus_bleu
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter

from utils import *
import config

def model_train(model_name, model, vocab, learning_rate, num_epochs, train_iterator, save_dir, 
                corpus_name, hidden_size, loadFilename, device, teacher_forcing_ratio, 
                clip = 1, load_model = False, save_model = False, max_length=config.MAX_LENGTH):
    """   
    model train pipeline
    
    Input
    - model_name: model name
    - model: model object
    - vocab: Voc object


    Returns
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    pad_idx = vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    writer = SummaryWriter("runs/loss_plot")
    loss_list = []

    for epoch in tqdm(range(num_epochs)):
        print(f"[Epoch {epoch} / {num_epochs}]")

        losses = []

        for input, target in train_iterator:
            # Get input and targets and get to cuda
            inp_data = input.to(device)
            target = target.to(device)

            # Forward prop
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        
        # score = bleu(test_data[1:100], model, german, english, device)
        mean_loss = sum(losses) / len(losses)
        loss_list.append(mean_loss)
        print('Train Loss: {:.4f}, Bleu Score: {:.4f}'.format(
                     mean_loss, 0))
        
        scheduler.step(mean_loss)
        
        # Save checkpoint
        if save_checkpoint:
            directory = os.path.join(save_dir, corpus_name, '{}_{}'.format(model_name, hidden_size))
            os.makedirs(directory, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

    print('save the result!')

    algorithm = '{}_{}'.format(model_name, hidden_size)
    os.makedirs(config.result_dir, exist_ok=True)
    with open(f'{config.result_dir}/{algorithm}_loss.csv', 'w') as f:
        for loss in loss_list:
            csv.writer(f).writerow([loss.item()])
    
    # with open(f'{config.result_dir}/{algorithm}_bleu_score.csv', 'w') as f:
    #     for bleu_s in bleu_score_list:
    #         csv.writer(f).writerow([bleu_s.item()])

    plot_result()
    print('plot the result!')

def indexToWord(index_list, voc, type='predict'):
    """
    convert the index to word
    Inputs:
    - index_list: list, the list of index
    """
    res = []
    for ind_lis in index_list:
        temp = []
        for index in ind_lis:
            if index not in [0, 1, 2]:
                temp.append(voc.index2word[index])
        if type == 'predict':
            res.append(temp)
        else:
            res.append([temp])
    return res

def plot_result():
    """
    plot the loss of each algorithm
    """
    loss_list = []
    bleu_score_list = []
    for filename in os.listdir(config.result_dir):
        if filename.endswith('loss.csv'):
            algorithm = filename.split('.')[0]
            with open(os.path.join(config.result_dir, filename), 'r') as f:
                loss_list.append(
                    (algorithm, np.array(list(csv.reader(f))).astype('float64').squeeze())
                )
        
        if filename.endswith('bleu_score.csv'):
            algorithm = filename.split('.')[0]
            with open(os.path.join(config.result_dir, filename), 'r') as f:
                bleu_score_list.append(
                    (algorithm, np.array(list(csv.reader(f))).astype('float64').squeeze())
                )
    plt.xlabel("Iterations(conversation)")
    plt.ylabel("Average loss")
    legend = []
    for name, values in loss_list:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join('results', 'loss_plot.png'))
    plt.close()

    plt.xlabel("Iterations(conversation)")
    plt.ylabel("Blue score")
    legend = []
    for name, values in bleu_score_list:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join('results', 'bleu_plot.png'))

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def test1():
    predicted_words = [['My', 'full', 'pytorch', 'test'], ['My', 'full', 'pytorch', 'test']]
    # target sentence need to be a list of list
    target_words = [[['My', 'full', 'pytorch', 'test']], [['My', 'full', 'pytorch', 'test']]]
    print(bleu_score(predicted_words, target_words))
    assert bleu_score(predicted_words, target_words) == 1.0, "test1 failed"
    print("test1 passed!")

def test2():
    print('save the result!')
    algorithm = '{}-{}_{}'.format(1, 1, 20)
    os.makedirs('results', exist_ok=True)
    with open(f'results/{algorithm}.csv', 'w') as f:
        for i in [0.1,1,0.5,0.2,0.4,0.5,0.6,0.2,0.3]:
            csv.writer(f).writerow([i])
    
    plot_result()

def main():
    pass

if __name__ == '__main__':
    test1()