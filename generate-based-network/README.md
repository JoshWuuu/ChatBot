# generation_based_network
## Introduction
It is a side project in 2023. The project is a Natural Language Processing topic. The languages and relevent packages are **Python - Pytorch**. The project builds a chatbot using generation based network. The network is seq2seq model. The encoder-decoder network is built with lstm with att layers and transfomer. The embedding layers is learned with the network. 
## Dataset
dataset: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
## Preprocess
The words are lemmatized and preprocessed into matrix with the index of the word in the corpus. 
## Model
The network is seq2seq model. The encoder-decoder network is built with lstm with att layers and transfomer. The connection between LSTM encoder and decoder are connected using attention machanism to calculate the importance of each input correspond to the output. The model is trained in the autoregression style with the teacher forcing trick. In addition, the transfomer is included in the project. 
