import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader


import preprocess
import lstm_class

n = 2 #10
min_occurences = 1 #5
batch_size = 1 #32


#lotr1_text = preprocess.load_from_url("http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm")

lotr1_text = "and the best thing is . that the greatest thing the best thing the . and the best thing is that . the best".split()
#Counter({'greatest': 8, 'that': 7, 'is': 6, 'and': 5, '.': 4, 'thing': 3, 'best': 2, 'the': 1})

word_to_id, id_to_word = preprocess.get_vocab(lotr1_text, min_occurences)
lotr1_ids =  [word_to_id[word] for word in lotr1_text]



training_dataset = preprocess.get_tensor_dataset(lotr1_ids, n)
training_loader = DataLoader(training_dataset, batch_size=batch_size, drop_last=True, shuffle=True)



#=======================================#
#             LSTM Parameters           #
#=======================================#

# Size parameters
vocab_size = len(word_to_id) + 1
embedding_dim = 256     # size of the word embeddings
hidden_dim = 256        # size of the hidden state
n_layers = 2            # number of LSTM layers

# Training parameters
epochs = 10 #10
learning_rate = 0.001
clip = 1



#=======================================#
#         Initialize/Train RNN          #
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

net.eval()


#=======================================#
#         Generate Sample Text          #
#=======================================#
softmax = nn.Softmax(dim=0)

def predict_from_ids(network, ids_list, batch_size=1):

    #batch size is 1 as it is a single input
    hidden = network.init_hidden(batch_size)

    input = torch.tensor([ids_list])

    output, hidden = network.forward(input, hidden)

    last_word_logits = output[0]

    predicted_probabilities = softmax(last_word_logits).detach().numpy()

    # Picks a probability-weighted random choice of the words
    prediction = np.random.choice(len(last_word_logits), p=predicted_probabilities)
    return prediction



for j in range(0, 15):
    print(f'\n\nPrediction {j+1}')
    num_words = 300

    #First ID in generation is a period, so it begins with how it thinks a new sentence will start
    ids = [word_to_id["."]]
    for _ in range(num_words):
        last_n_ids = ids[-n:]
        prediction = predict_from_ids(net, last_n_ids)
        ids.append(prediction)

    predicted_string = ' '.join([id_to_word[id] for id in ids[1:]])
    print(predicted_string)





exit(0)
