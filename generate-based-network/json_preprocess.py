from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

def printLines(file, n=10):
    """
    count converstational lines
    
    Input:
    - file: str, folder path
    - n: int, number of lines to print

    Returns:
    """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    
    for line in lines[:n]:
        print(line)

def linesAndConversations(file):
    """
    parsing the raw json files to create lines and conversations
    
    Input:
    - file: str, folder path

    Returns:
    - lines: dictionary
    - conversations: dictionary
    """
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            # Extract fields for line object
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj['lineID']] = lineObj

            # Extract fields for conversation object
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

def extractSentencePairs(conversations):
    """
    extract pairs of sentences from conversations
    
    Input:
    - conversations: dictionary

    Returns:
    - qa_pairs: list, question and response pairs
    """
    qa_pairs = []
    for conversation in conversations.values():
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation["lines"]) - 1):  
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def main():
    # take a look on the data 
    corpus_name = "movie-corpus"
    corpus = os.path.join("data", corpus_name)

    printLines(os.path.join(corpus, "utterances.jsonl"))

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict and conversations dict
    lines = {}
    conversations = {}
    # Load lines and conversations
    print("\nProcessing corpus into lines and conversations...")
    lines, conversations = linesAndConversations(os.path.join(corpus, "utterances.jsonl"))

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)

# Sample lines from file:
# b'They do to!\tThey do not!\n'
# b'She okay?\tI hope so.\n'
# b"Wow\tLet's go.\n"
# b'"I\'m kidding.  You know how sometimes you just become this ""persona""?  And you don\'t know how to quit?"\tNo\n'
# b"No\tOkay -- you're gonna need to learn how to lie.\n"
# b"I figured you'd get to the good stuff eventually.\tWhat good stuff?\n"
# b'What good stuff?\t"The ""real you""."\n'
# b'"The ""real you""."\tLike my fear of wearing pastels?\n'
# b'do you listen to this crap?\tWhat crap?\n'
# b"What crap?\tMe.  This endless ...blonde babble. I'm like, boring myself.\n"
if __name__ == "__main__":
    main()