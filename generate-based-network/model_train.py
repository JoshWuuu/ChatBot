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

import config

def model_train(model_name, model, vocab, learning_rate, num_epochs, train_iterator, save_dir, 
                corpus_name, hidden_size, device, clip = 1, load_model = False, 
                save_checkpoint = False):
    """   
    model train pipeline
    
    Input
    - model_name: str, model name
    - model: obj, model object
    - vocab: obj, vocab object
    - learning_rate: float, learning rate
    - num_epochs: int, number of epochs
    - train_iterator: obj, train iterator
    - save_dir: str, save directory
    - corpus_name: str, corpus name
    - hidden_size: int, hidden size
    - device: obj, device object
    - clip: int, clip, default 1
    - load_model: bool, load model, default False
    - save_checkpoint: bool, save checkpoint, default False
    
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
    loss_list, bleu_score_list = [], []
    step = 0
    response_list, target_list = [], []

    loop = tqdm(range(num_epochs))
    for epoch in loop:
        print(f"[Epoch {epoch} / {num_epochs}]")

        losses = []

        for input, target in train_iterator:
            # Get input and targets and get to cuda
            inp_data = input.to(device)
            target = target.to(device)

            # Forward prop
            output = model(inp_data, target)

            # for bleu score
            cur_response = output.argmax(2)
            response_list += cur_response.reshape(output.shape[1], -1).tolist()
            target_list += target.reshape(output.shape[1], -1).tolist()
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
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
            break
        
        # score = bleu(test_data[1:100], model, german, english, device)
        mean_loss = sum(losses) / len(losses)
        bleu_s = calculate_bleu_score(response_list, target_list, vocab)
        loss_list.append(mean_loss)
        bleu_score_list.append(bleu_s)
        loop.set_postfix(loss=mean_loss, bleu_score=bleu_s)
        
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
            csv.writer(f).writerow([loss])
    
    with open(f'{config.result_dir}/{algorithm}_bleu_score.csv', 'w') as f:
        for bleu_s in bleu_score_list:
            csv.writer(f).writerow([bleu_s])

    plot_result()
    print('plot the result!')

def calculate_bleu_score(output, target, vocab):
    """
    calculate bleu score

    Input
    - output: list, list of index
    - target: list, list of index
    - vocab: obj, vocab object

    Returns
    - bleu_score: float, bleu score
    """
    targets = []
    outputs = []
    
    for i in range(len(output)):
        temp_target, temp_output = [], []
        for j in range(len(output[i])):
            if output[i][j] not in [0, 1, 2, 3]:
                indexToWord = vocab.itos[output[i][j]]
                temp_output.append(indexToWord)
        for j in range(len(target[i])):
            if target[i][j] not in [0, 1, 2, 3]:
                indexToWord = vocab.itos[target[i][j]]
                temp_target.append(indexToWord)
        outputs.append(temp_output)
        targets.append([temp_target])
    return bleu_score(outputs, targets)

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

def test3():
    print('test bleu score')
    target = [[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]
    output = [[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]
    assert calculate_bleu_score(output, target) == 0.0, "test3 failed"
    print("test3 passed!")

def main():
    pass

if __name__ == '__main__':
    test3()