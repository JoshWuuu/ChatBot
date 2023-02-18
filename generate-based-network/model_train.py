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

def model_train():
    """   
    train the model with the attention encoder and decoder layers for chatbot
    
    Input
    - input_variable: (max_length, batch_size)

    Returns
    - loss: (1)
    """
    