import urllib.request
import re
import torch
from torch.utils.data import TensorDataset
from collections import Counter
from bs4 import BeautifulSoup

#===========================================#
#                Description                #
#===========================================#

# This file contains the functions necessary for preprocessing, and is loaded by train.py
# 1. parse_and_clean
# 2. load_from_url
# 3. get_vocab
# 4. get_tensor_dataset




#===========================================#
#         Preprocessing Functions           #
#===========================================#

# cleans up text by removing extraneous characters
def parse_and_clean(input):
    text = input.lower()
    text = re.sub('mr\.', 'mr ', text)                                           # Removes changes Mr. to Mr to avoid period confusion

    text = re.sub('can.t', 'cant', text)                                       #
    text = re.sub('isn.t', 'isnt', text)                                       #
    text = re.sub('won.t', 'wont', text)                                       #
    text = re.sub('ain.t', 'aint', text)                                       #
    text = re.sub('aren.t', 'arent', text)                                       #
    text = re.sub('didn.t', 'didnt', text)                                       #
    text = re.sub('doesn.t', 'doesnt', text)                                       #
    text = re.sub('hadn.t', 'hadnt', text)                                       #
    text = re.sub('haven.t', 'havent', text)                                       #
    text = re.sub('mustn.t', 'mustnt', text)                                       #
    text = re.sub('shouldn.t', 'shouldnt', text)                                       #
    text = re.sub('we.ve', 'weve', text)                                       #

    text = re.sub('[\'\"‘]', '', text)                                          # Removes all single/double quotes
    text = re.sub(r"(\.|\,|\;|\:|\!)", lambda x: f' {x.group(1)} ', text)          # Adds space on both sides of punctuation
    text = re.sub(re.compile('[^\w,.!:;]+', re.UNICODE), ' ', text)            # Replaces all remaining non-alphanumeric/punctuation with space

    return text




# loads and parses html text for Fellowship of the Ring
def load_full_text():
    with open('data/fotr.txt', 'r') as file:
        fotr = parse_and_clean(file.read())

    with open('data/tt.txt', 'r') as file:
        tt = parse_and_clean(file.read())

    with open('data/rotk.txt', 'r') as file:
        rotk = parse_and_clean(file.read())

    lotr_full = fotr + tt + rotk
    return lotr_full.split()




# assigns each word that occurs at least min_occurence times an integer id
def get_vocab(corpus, min_occurences):
    vocab = Counter()
    for word in corpus:
        vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= min_occurences})
    vocab_tuples = vocab_top.most_common(len(vocab_top))

    word_to_id = Counter({word: i+1 for i,(word, c) in enumerate(vocab_tuples)})
    id_to_word = ["_"] + [word for word, index in word_to_id.items()]

    return word_to_id, id_to_word




# coverts the tokenized text to training data: label is a word in the text, and feature is the n preceding words
def get_tensor_dataset(list_of_ids, n):
    features = []
    labels = []
    for i in range(n, len(list_of_ids)):
        labels.append(list_of_ids[i])
        features.append(list_of_ids[i-n:i])

    return TensorDataset(torch.tensor(features), torch.tensor(labels))
