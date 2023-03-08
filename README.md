# ChatBot
The project explores different methods of ChatBot build. 
## Chatter_bot
The Chattter Bot package is written by https://github.com/gunthercox/ChatterBot. The part imports ChatBot to establish the bot, and use a few conversations to train the bot. The bot will be re-trained after each conversation. It is a good start to build my first chatbot. \
Reference: https://github.com/gunthercox/ChatterBot, https://chatbotslife.com/how-to-create-an-intelligent-chatbot-in-python-c655eb39d6b1 

## Retrieval-based Network
This part is using DNN to classify the sentiment of the input and use the sentiment result to choose predefined response randomly in the test.json file. It is an easy way to train the model and have the decent answer. However, if the user asks for specific question, the model is unable to answer it. The model is built using tensorflow. \
Reference: https://www.projectpro.io/article/python-chatbot-project-learn-to-build-a-chatbot-from-scratch/429

## Generative-based Network
This part is using seq2seq model. The encoder-decoder network is built with att layers. The embedding layers is learned with the network. The model is built using pytorch. The result is still not good. The future direction includes:
* Pre-trained embedding layers
* Normalization: batch norm, layer norm
* Activation function: ReLu, GeLu 
* Attention layer: dot, general, concate
* Model: GRU, LSTM, Transformer
* Model tuning: lr or batch size exploration
Reference: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

## ChatGPT
The final part is using popular openai gpt model and streamlit to build the chat bot with gui. The bot use openai engine, text-davinci-003 as the response generator. The bot seems smarter than the previous three methods, since the model is larger and pre-trained on the larger dataset. \
Reference:https://medium.com/@avra42/build-your-own-chatbot-with-openai-gpt-3-and-streamlit-6f1330876846
