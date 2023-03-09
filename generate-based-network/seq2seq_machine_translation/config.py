import spacy
import torch

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0