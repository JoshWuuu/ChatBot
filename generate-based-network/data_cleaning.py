from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import unicodedata
from io import open

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    """
    class for vocabulary matrix
        
    Input:
    - name: str
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
    
    def addWord(self, word):
        """
        add word to word2index, word2count and index2word dictionary, 
        and also keep track of the word count of whole dictionary
        
        Input:
        - word: str

        Returns:
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def addSentence(self, sentence):
        """
        add word of sentence to the addword function 
        
        Input:
        - sentence: str

        Returns:
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def trim(self, min_count):
        """
        if word count below a certain count threshold, rm it
        
        Input:
        - min_count: int

        Returns:
        """
        if self.trimmed:
            return 
        self.trimmed = True

        keep_words= []

        for key, value in self.word2count.items():
            if value >= min_count:
                keep_words.append(key)
        
        print('keep words ration {} / {}  = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words)/ len(self.word2index)
        ))

        # readd the keep word to word2index, word2count and index2word dictionary
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def unicodeToAscii(s):
    """
    if word count below a certain count threshold, rm it
        
    Input:
    - min_count: int

    Returns:
    """
    return "".join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    lowercase, trim, and rm non-letter characters
        
    Input:
    - s

    Returns:
    """
    # strip remove all the leading and trailing space 
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    """
    lowercase, trim, and rm non-letter characters
        
    Input:
    - s

    Returns:
    - voc: obj
    - pairs: list, [[pair of conversation]]
    """
    print('Reading lines')
    # splits the file into lines
    lines = open(datafile, encoding ='utf-8').read().strip().split('\n')
    # splits the lines into the pairs
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPairs(pairs, MAX_LENGTH):
    """
    filter each pair if the length of both sentences in the pairs are shorter than MAX_LENGTH
        
    Input:
    - pairs: list[list]
    - MAX_LENGTH: int

    Returns:
    - res: list[list], the pairs pass the constraint
    """    
    res = []
    for pair in pairs:
        if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH:
            continue
        else:
            res.append(pair)

    return res

def trimRareWords(voc, pairs, MIN_COUNT):
    """
    trim the word that happen < min count
        
    Input:
    - voc: obj
    - pairs: list[list]
    - MIN_COUNT: int

    Returns:
    - keep_pairs
    """  
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)
    ))
    return keep_pairs


def main():
    save_dir = os.path.join("data", "save")
    corpus_name = "movie-corpus"
    MAX_LENGTH = 10  # Maximum sentence length to consider
    corpus = os.path.join("data", corpus_name)
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    print('Preparing training data')
    voc, pairs = readVocs(datafile, corpus_name)
    print('Filter sentence pairs with MAX_LENGTH: {}'.format(MAX_LENGTH))
    filtered_pairs = filterPairs(pairs)
    print('Filtered ratio (short sentense) {} / {} : {:.4f}'.format(
        len(filtered_pairs), len(pairs), len(filtered_pairs) / len(pairs)
    ))
    print('Put word into object')
    for pair in filtered_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Count words:', voc.num_words)

    MIN_COUNT = 3
    # Trim voc and pairs
    trimmed_pairs = trimRareWords(voc, filtered_pairs, MIN_COUNT)   
    print('Trimmed ratio (rare word) {} / {} : {:.4f}'.format(
        len(trimmed_pairs), len(filtered_pairs), len(trimmed_pairs) / len(filtered_pairs)
    ))

    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)  

if __name__ == '__main__':
    main()