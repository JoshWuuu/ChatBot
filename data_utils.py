import json
import nltk
import numpy as numpy
from nltk.stem import WordNetLemmatizer

def data_loading_json(data_root):
    """
    data loading json file
    
    Input:
    - data_root: str, folder path

    Returns:
    - data: dictionary
    """
    data_file = open(data_root + '/test.json').read()
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
            tokens = nltk.work_tokenize(pattern)
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

    return words, classes, data_x, data_y

if __name__ == "__main__":
    pass