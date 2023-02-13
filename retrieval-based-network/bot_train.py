import json
import string 
import random
import pickle

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

def data_loading_json(data_root):
    """
    data loading json file
    
    Input:
    - data_root: str, folder path

    Returns:
    - data: dictionary
    """
    data_file = open('test.json').read()
    data = json.loads(data_file)
    return data

def tokenize_and_lemmetaion(data):
    """
    tokenize, lemmetaion, delete punctuation, lower the word  
    
    Input:
    - data: dict

    Returns:
    - words: set, include all lower and lemmatized word with no punctuation
    - classes: set, include tag
    - data_x: list, all the sentances
    - data_y: list, all the tag
    """
    words = []
    classes = []
    data_x = []
    data_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]: 
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            data_x.append(pattern)
            data_y.append(intent["tag"])
        
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    
    lemmatizer = WordNetLemmatizer()
    # lemmatize all the words and delete punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # no duplicate and in alphabetical order
    words = sorted(set(words))
    classes = sorted(set(classes))

    with open("vocab", "wb") as fp:   #Pickling
        pickle.dump(words, fp)

    with open("class", "wb") as fp:   #Pickling
        pickle.dump(classes, fp)

    return words, classes, data_x, data_y

def text_to_matrix(words, classes, data_x, data_y): 
    """
    convert text and label into matrix
    
    Input:
    - words: set, include all lower and lemmatized word with no punctuation
    - classes: set, include tag
    - data_x: list, all the sentances
    - data_y: list, all the tag

    Returns:
    - train_X: np.array 
    - train_Y: np.array
    """
    training = []
    out_empty = [0] * len(classes)
    lemmatizer = WordNetLemmatizer()

    for idx, doc in enumerate(data_x):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        # convert sentence into matrix
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        
        # convert tag into matrix
        output_row = list(out_empty)
        output_row[classes.index(data_y[idx])] = 1

        training.append([bow, output_row])
    
    random.shuffle(training)
    training = np.array(training, dtype = object)

    train_X = np.array(list(training[:, 0]))
    train_Y = np.array(list(training[:, 1]))

    return train_X, train_Y

def model_build(train_X_len, train_Y_len):
    """
    model build
    
    Input:
    - train_X_len: int
    - train_Y_len: int

    Returns:
    - model
    """
    model = Sequential()
    model.add(Dense(128, input_shape=(train_X_len,), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(train_Y_len, activation="softmax"))
    
    return model

def model_train(model, train_X, train_Y):
    """
    model train
    
    Input:
    - train_X: np.array 
    - train_Y: np.array
    - model

    Returns:
    """
    adam = tf.keras.optimizer.Adam(learning_rate= 0.01, decay = 1e-6)
    model.compile(loss = 'categorical_crossentropy',
                optimizer=adam,
                metrics= ["accuracy"])
    print(model.summary())
    model.fit(x=train_X, y=train_Y, epochs=150, verbose =1)
    model.save_weight('checkpoints')
    
def main():
    data = data_loading_json("")

    words, classes, data_x, data_y = tokenize_and_lemmetaion(data)

    train_X, train_Y = text_to_matrix(words, classes, data_x, data_y)

    model = model_build(len(train_X[0]), len(train_Y[0]))

    model_train(model, train_X, train_Y)
    
if __name__ == "__main__":
    main()