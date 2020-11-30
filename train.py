import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader


import preprocess_data
import process_data
import vocab
import lstm_class

n = 20 #20
k = 20 #20
min_occurences = 5 #1
batch_size = 32 #1


lotr1_text = preprocess_data.load_from_url("http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm")

#lotr1_text = "and the best thing is . that the greatest thing the best thing the . and the best thing is that . the best".split()

word_to_id, id_to_word = vocab.get_vocab(lotr1_text, min_occurences)
lotr1_ids =  [word_to_id[word] for word in lotr1_text]



training_dataset = process_data.get_tensor_dataset(lotr1_ids, word_to_id["."], n, k)
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
epochs = 10
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
    hidden = net.init_hidden(batch_size)
    input = torch.tensor(ids_list)

    output, hidden = network.forward(input, hidden)

    last_word_logits = output[0]

    predicted_probabilities = softmax(last_word_logits).detach().numpy()

    # Picks a probability-weighted random choice of the words
    prediction = np.random.choice(len(last_word_logits), p=predicted_probabilities)
    return prediction



for j in range(0, 10):
    print(f'\n\nPrediction {j+1}')
    num_words = 300

    ids = [0] * (n-1)
    for i in range(num_words):
        last_n_ids = ids[-n:]
        prediction = predict_from_ids(net, [last_n_ids])
        ids.append(prediction)

    predicted_string = ' '.join([id_to_word[id] for id in ids[n:]])
    print(predicted_string)





exit(0)
