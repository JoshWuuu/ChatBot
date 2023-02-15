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

def main():
    pass

if __name__ == "__main__":
    main()