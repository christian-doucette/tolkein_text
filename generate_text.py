from  collections import Counter
import json
import torch
import torch.nn as nn
import numpy as np
import lstm_class


#===========================================#
#        Loads Model and word_to_id         #
#===========================================#

with open('trained_model/word_to_id.json') as json_file:
    word_to_id = Counter(json.load(json_file))

id_to_word = ["_"] + [word for word, index in word_to_id.items()]

net = torch.load('trained_model/trained_model.pt')
net.eval()

softmax     = nn.Softmax(dim=0)
log_softmax = nn.LogSoftmax(dim=0)




#===========================================#
#        Predict Next ID Function           #
#===========================================#

def predict_next_id(network, ids_list, batch_size=1):

    #batch size is 1 as it is a single input
    hidden = network.init_hidden(batch_size)

    input = torch.tensor([ids_list])


    output, hidden = network.forward(input, hidden)

    last_word_logits = output.squeeze()
    predicted_probabilities = softmax(last_word_logits).detach().numpy()

    # Sets probability of generating the <Unknown> token to 0, then adjusts other probabilities so they still sum to 1
    predicted_probabilities[0] = 0
    predicted_probabilities[ids_list[-1]] = 0
    predicted_probabilities = predicted_probabilities / np.sum(predicted_probabilities)

    # Picks a probability-weighted random choice of the words
    prediction = np.random.choice(len(last_word_logits), p=predicted_probabilities)
    return prediction





#===========================================#
#       Text Generation Parameters          #
#===========================================#

n = 9
num_sentences = 20
num_paragraphs = 3
seed_text = ""




#===========================================#
#             Generates Text                #
#===========================================#

for j in range(0, num_paragraphs):
    print(f'\n\nPrediction {j+1}')

    #First ID in generation is a period, so it begins with how it thinks a new sentence will start
    ids = [word_to_id["."]] + [word_to_id[word] for word in seed_text.split()]


    num_periods = 0
    while (num_periods < num_sentences):
        last_n_ids = ids[-n:]
        prediction = predict_next_id(net, last_n_ids)
        ids.append(prediction)

        if prediction == word_to_id["."]:
            num_periods += 1


    predicted_string = ' '.join([id_to_word[id] for id in ids[1:]])
    print(predicted_string)
