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
    text = input.getText()
    text = re.sub('Mr.', 'Mr ', text)                                           # Removes changes Mr. to Mr to avoid period confusion
    text = re.sub('[\'\"‘]', '', text)                                          # Removes all single/double quotes
    text = re.sub(r"(\.|\,|\(|\)|\;|\:)", lambda x: f' {x.group(1)} ', text)    # Adds space on both sides of punctuation
    text = re.sub('[^0-9a-zA-Z.,:;()]+', ' ', text)                             # Replaces all remaining non-alphanumeric/punctuation with space
    text = text.lower()                                                         # Sets to lowercase
    return text




# loads and parses html text for Fellowship of the Ring
def load_from_url(url):
    fp = urllib.request.urlopen("http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm")
    mybytes = fp.read()
    mystr = mybytes.decode("latin-1")

    soup = BeautifulSoup(mystr, 'html.parser')
    p_tags = soup.find_all('p')
    p_tags_processed_text = list(map(parse_and_clean, p_tags))[109:3955]
    full_text = "".join(p_tags_processed_text)
    #Fellowship Start: 109, End: 3954 (inclusive)
    return full_text.split()




# assigns each word that occurs at least min_occurence times an integer id
def get_vocab(corpus, min_occurences):
    vocab = Counter()
    for word in corpus:
        vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= min_occurences})
    vocab_tuples = vocab_top.most_common(len(vocab_top))

    word_to_id = Counter({word: i+1 for i,(word, c) in enumerate(vocab_tuples)})

    id_to_word = [word for word, index in word_to_id.items()]
    id_to_word.insert(0, "_")

    return word_to_id, id_to_word




# coverts the tokenized text to training data: label is a word in the text, and feature is the n preceding words
def get_tensor_dataset(list_of_ids, n):
    features = []
    labels = []
    for i in range(n, len(list_of_ids)):
        labels.append(list_of_ids[i])
        features.append(list_of_ids[i-n:i])

    return TensorDataset(torch.tensor(features), torch.tensor(labels))
