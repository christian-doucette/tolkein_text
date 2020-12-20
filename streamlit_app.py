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

id_to_word = ["<Unknown>"] + [word for word, index in word_to_id.items()]

net = torch.load('trained_model/trained_model.pt')
net.eval()




#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Uses an LSTM neural network trained on *The Lord of the Rings*. Check out the code [here](https://github.com/christian-doucette/tolkein_text)!"

st.title('Lord of the Rings Text Generator')
st.write(desc)

num_sentences = st.number_input('Number of Sentences', min_value=1, max_value=20, value=5)
user_input = st.text_input('Seed Text (can leave blank)')


if st.button('Generate Text'):
    generated_text = generate_text.prediction(net, word_to_id, id_to_word, always_capitalized, user_input, 9, num_sentences)
    st.write(generated_text)
