import torch
import torch.nn as nn

#===========================================#
#                Description                #
#===========================================#

# This is the LSTM Neural Network Class. The architecture of the network is this:
# input -> embedding layer -> LSTM layers -> dropout layer -> fully connected linear layer -> output

# input is a list of n=9 words represented by their ids in the vocabulary, label is the following word

# output is a list of logits over the set of words, where higher means a higher chance that that word follows
# applying softmax to the output turns it into a pprobability distribution over all the words




#===========================================#
#    LSTM Recurrent Neural Network Class    #
#===========================================#

class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(LSTM, self).__init__()

        # network size parameters
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim


        # the layers of the network
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)



    def forward(self, input, hidden):
        # Performs a forward pass of the model on some input and hidden state.
        batch_size = input.size(0)

        # pass through embeddings layer
        embeddings_out = self.embedding(input)

        # pass through LSTM layers
        lstm_out, hidden = self.lstm(embeddings_out, hidden)

        # slice lstm_out to just get output of last element of the input sequence
        lstm_out = lstm_out[:, -1]

        # pass through dropout layer
        dropout_out = self.dropout(lstm_out)

        #pass through fully connected layer - don't need to use Softmax activation func as CrossEntropyLoss applies it
        fc_out = self.fc(dropout_out)

        # return final output and hidden state
        return fc_out, hidden


    def init_hidden(self, batch_size):
        #Initializes hidden state
        # Create two new tensors `with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM


        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden
