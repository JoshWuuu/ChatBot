from datasets import data_preprocessing
from model_build import *
from model_train import *
from utils import *
import torch.optim as optim

def model_eval(model, sentence, german, english, device, max_length=50):
    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

def main():
    # Load dataset 
    train_iterator, valid_iterator, test_iterator = data_preprocessing()

    ### We're ready to define everything we need for training our Seq2Seq model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # Training hyperparameters
    num_epochs = 100
    learning_rate = 3e-4
    batch_size = 32

    # Model hyperparameters
    input_size_encoder = len(german.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 1
    enc_dropout = 0.0
    dec_dropout = 0.0

    encoder_net = Encoder(
        input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
    ).to(device)

    decoder_net = Decoder(
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_fn(model, train_iterator, optimizer, criterion, device, num_epochs, save_model)

    model.eval()

    sentence = (
    "ein boot mit mehreren männern darauf wird von einem großen"
    "pferdegespann ans ufer gezogen."
    )

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")




