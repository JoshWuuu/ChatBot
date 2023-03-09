import torch
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import save_checkpoint, bleu

def train_fn(model, optimizer, num_epochs, train_iterator, valid_iterator, criterion, clip, save_model, device):
    """
    train function

    Inputs:
    - model: model
    - optimizer: optimizer
    - num_epochs: int, number of epochs
    - train_iterator: train iterator
    - valid_iterator: valid iterator
    - criterion: criterion
    - clip: float, gradient clipping
    - save_model: bool, whether to save model
    - device: device
    """
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        model.train()

        writer = SummaryWriter(f"runs/loss_plot")
        step = 0
        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

    # running on entire test data takes a while
    score = bleu(test_data[1:100], model, german, english, device)
    print(f"Bleu score {score * 100:.2f}")