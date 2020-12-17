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
            words[i] = cur.capitalize()

    string = ' '.join(words[1:])
    string = re.sub('Mr ', 'Mr. ', string)             # Changes Mr back to Mr.
    string = re.sub(r'\s([,.!?:;])', r'\1', string)    # Removes sapce before punctuation
    #string = re.sub(r"( \.| \,| \;| \:| \!| \?)", lambda x: f'{x.group(1)}', string)   # Removes extra space in front of punctuation

    return string





#===========================================#
#        Predict Next ID Function           #
#===========================================#

def predict_next_id(network, ids_list, batch_size=1):
    softmax = nn.Softmax(dim=0)

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
#        Full Predictions Function          #
#===========================================#

def prediction(network, word2id, id2word, should_capitalize, user_input, n, num_sentences):
        seed_text = preprocess.parse_and_clean(user_input)
        ids = [word_to_id["."]] + [word_to_id[word] for word in seed_text.split()]

        finished_sentences = 0
        while (finished_sentences < num_sentences):
            last_n_ids = ids[-n:]
            prediction = predict_next_id(network, last_n_ids)
            ids.append(prediction)
            if prediction in [word_to_id[punc] for punc in [".", "!", "?"]]:
                finished_sentences += 1

        predicted_string = add_formatting(ids, id2word, should_capitalize)
        return predicted_string






#===========================================#
#        Loads Model and word_to_id         #
#===========================================#

with open('trained_model/word_to_id.json') as json_file:
    word_to_id = Counter(json.load(json_file))

with open('trained_model/always_capitalized.json') as json_file:
    always_capitalized = json.load(json_file)

id_to_word = ["_"] + [word for word, index in word_to_id.items()]

net = torch.load('trained_model/trained_model.pt')
net.eval()







#===========================================#
#       Text Generation Parameters          #
#===========================================#

n = 9
num_sentences = 10
num_paragraphs = 0
user_input = ""




#===========================================#
#             Generates Text                #
#===========================================#

for j in range(0, num_paragraphs):
    print(f'Prediction {j+1}')

    predicted_string = prediction(net, word_to_id, id_to_word, always_capitalized, user_input, n, num_sentences)
    print(predicted_string)
    print('\n\n')
