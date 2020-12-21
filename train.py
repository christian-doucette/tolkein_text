import json
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import preprocess
import lstm_class



#=======================================#
#        Preprocessing Parameters       #
#=======================================#

n = 9 #2                    # Number of words used in prediction
min_occurences = 10 #1       # Minimum number of occurences of a word for it to occur in vocabulary
batch_size = 32 #1




#=======================================#
#             Preprocessing             #
#=======================================#

lotr_full_text = preprocess.load_full_text()

word_to_id, id_to_word = preprocess.get_vocab(lotr_full_text, min_occurences)

lotr_full_ids =  [word_to_id[word] for word in lotr_full_text]


training_dataset = preprocess.get_tensor_dataset(lotr_full_ids, n)
training_loader = DataLoader(training_dataset, batch_size=batch_size, drop_last=True, shuffle=True)




#=======================================#
#           Network Parameters          #
#=======================================#

# Size parameters
vocab_size = len(word_to_id) + 1
embedding_dim = 256     # size of the word embeddings
hidden_dim = 256        # size of the hidden state
n_layers = 2            # number of LSTM layers

# Training parameters
epochs = 14 #10
learning_rate = 0.001
clip = 1



#=======================================#
#       Initialize/Train Network        #
#=======================================#

net = lstm_class.LSTM(vocab_size, embedding_dim, hidden_dim, n_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

net.train()
for e in range(epochs):
    print(f'Epoch {e}')
    hidden = net.init_hidden(batch_size)

    # loops through each batch
    for features, labels in training_loader:


        # resets training history
        hidden = tuple([each.data for each in hidden])
        net.zero_grad()

        # computes gradient of loss from backprop
        output, hidden = net.forward(features, hidden)
        loss = loss_func(output, labels)
        loss.backward()

        # using clipping to avoid exploding gradient
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()




#=======================================#
#         Saves Trained Network         #
#=======================================#

net.eval()
torch.save(net, 'trained_model/trained_model.pt')

with open('trained_model/word_to_id.json', 'w') as fp:
    json.dump(word_to_id, fp, indent=4)
