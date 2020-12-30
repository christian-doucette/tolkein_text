from  collections import Counter
import json
import re
import torch
import torch.nn as nn
import numpy as np
import lstm_class
import preprocess

#=======================================#
#           Reformat Function           #
#=======================================#

def add_formatting(list_of_ids, id2word, should_capitalize):
    words = [id2word[id] for id in list_of_ids]
    for i in range(1, len(words)):
        prev = words[i-1]
        cur  = words[i]


        if (prev in [".", "!", "?"]) or (cur in should_capitalize and should_capitalize[cur]):
            if words[i] != "<Unknown>":
                words[i] = cur.capitalize()

    string = ' '.join(words[1:])
    string = re.sub('Mr ', 'Mr. ', string)             # Changes Mr back to Mr.
    string = re.sub(r'\s([,.!?:;])', r'\1', string)    # Removes sapce before punctuation

    return string





#===========================================#
#        Predict Next ID Function           #
#===========================================#

def predict_next_id(network, ids_list, batch_size=1, temp=0.8):
    softmax = nn.Softmax(dim=0)

    #batch size is 1 as it is a single input
    hidden = network.init_hidden(batch_size)

    input = torch.tensor([ids_list])


    output, hidden = network.forward(input, hidden)

    last_word_logits = output.squeeze()

    predicted_probabilities = softmax(last_word_logits / temp).detach().numpy()

    # Sets probability of generating the <Unknown> token to 0,
    predicted_probabilities[0] = 0

    # Sets probability of repeating last token to 0
    predicted_probabilities[ids_list[-1]] = 0


    # Adjusts so probabilities still sum to 1
    predicted_probabilities = predicted_probabilities / np.sum(predicted_probabilities)

    # Picks a probability-weighted random choice of the words
    prediction = np.random.choice(len(last_word_logits), p=predicted_probabilities)
    return prediction



#===========================================#
#        Full Predictions Function          #
#===========================================#

def prediction(network, word2id, id2word, should_capitalize, user_input, n, num_sentences):
        seed_text = preprocess.parse_and_clean(user_input)
        ids = [word2id["."]] + [word2id[word] for word in seed_text.split()]

        finished_sentences = 0
        while (finished_sentences < num_sentences):
            last_n_ids = ids[-n:]
            prediction = predict_next_id(network, last_n_ids)
            ids.append(prediction)
            if prediction in [word2id["."], word2id["!"], word2id["?"]]:
                finished_sentences += 1

        predicted_string = add_formatting(ids, id2word, should_capitalize)
        return predicted_string
