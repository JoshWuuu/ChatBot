import json
import string 
import random
import pickle

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from bot_train import data_loading_json, model_build

def tokenize_and_vectorice_response(text, words):
    """
    tokenize, lemmetaion, and vectorice the word  
    
    Input:
    - text: str, input
    - words: set, include all lower and lemmatized word with no punctuation

    Returns:
    - input_x: np array
    """
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    bow = [0] * len(words)

    for w in tokens:
        for idx, word in enumerate(words):
            if word == w:
                bow[idx] == 1
    return np.array(bow)

def pred_class(input_x, words, classes, model):
    """
    return prediction with probability >= 0.5
    
    Input:
    - input_x: np array
    - words: set, include all lower and lemmatized word with no punctuation
    - classes: set, labels
    - model

    Returns:
    - response_list: list
    """
    result = model.predict(input_x)
    thresh = 0.5
    y_pred = [[index, res] for index, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True) 
    response_list = []
    for r in y_pred:
        response_list.append(classes[r[0]])
    
    return response_list

def get_response(response_list, data):
    """
    get response in response list
    
    Input:
    - response_list: list
    - data

    Returns:
    - res: str
    """
    if len(response_list) == 0:
        res = "Sorry! I don't understand"
    else:
        tag = response_list[0]
        list_of_intents = data['intents']
        for i in list_of_intents:
            if i["tag"] == tag:
                res = random.choice(i['response'])
                break
    return res

def main():
    with open("vocab", "rb") as fp:   # Unpickling
       words = pickle.load(fp)
    
    with open("classes", "rb") as fp:   # Unpickling
       classes = pickle.load(fp)

    data = data_loading_json("")
    model = model_build(len(words), len(classes))
    model.load_weights('checkpoints')

    # get the tag and use tag to random choose response
    print('Press 0 to exit the conversation!')
    while True:
        message = input("User: ")
        if message == "0":
            break
        else:
            input_x = tokenize_and_vectorice_response(message, words)
            intents = pred_class(input_x, words, classes, model)
            response = get_response(intents, data)
            print('bot: {}'.format(response))

if __name__ == '__main__':
    main()