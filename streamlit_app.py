import streamlit as st
import json
import torch
from collections import Counter

import generate_text

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
#        Loads Model and word_to_id         #
#===========================================#

#print('1: creating seed\n\n')
user_input = st.text_input('Seed Text')
#print(f'2: {user_input}')

generated_text = generate_text.prediction(net, word_to_id, id_to_word, always_capitalized, user_input, 9, 5)
#print(f'3: {generated_text}')
st.write(generated_text)
