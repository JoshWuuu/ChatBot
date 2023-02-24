import torch

class config:
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    corpus_name = "movie-corpus"
    MAX_LENGTH = 10 # Maximum sentence length to consider
    MIN_COUNT = 3 # Minimum word count threshold for trimming

    # model config
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 1
    save_every = 500