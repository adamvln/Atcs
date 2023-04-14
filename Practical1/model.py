import torch.nn as nn
import torch

class Average_Encoder(nn.Module):
    '''
    Encoder that gives a sentence representation based on the mean of all tokens of the sentence.
    '''
    def __init__(self, vocab_size, embedding_size, embedding_table, device):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        #trainable lookup table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_size).to(self.device)

        # copy pre-trained word vectors into embeddings table
        self.embed.weight.data.copy_(torch.from_numpy(embedding_table))
        # disable training the pre-trained embeddings
        self.embed.weight.requires_grad = False

    def forward(self, inputs):
        inputs[inputs > self.vocab_size] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #mean over all encoders
        logits = torch.mean(embeds, dim = 1)
        return logits
    
class Unidir_LSTM(nn.Module):
    '''
    Encoder that gives a sentence representation based on the mean of all tokens of the sentence.
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding_table, device):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        #trainable lookup table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_size).to(self.device)

        # copy pre-trained word vectors into embeddings table
        self.embed.weight.data.copy_(torch.from_numpy(embedding_table))
        # disable training the pre-trained embeddings
        self.embed.weight.requires_grad = False

        #LSTM cell
        self.LSTM_layer = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, inputs):
        inputs[inputs > self.vocab_size] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #hidden and cell states initialization
        hidden_state = torch.zeros((len(embeds), self.hidden_size))
        cell_state = torch.zeros((len(embeds), self.hidden_size))

        #only the last hidden state matters
        for i in range(0, len(embeds[0])):
            hidden_state, _ = self.LSTM_layer(embeds[:,i], (hidden_state, cell_state))

        return hidden_state
    
class Bidirect_LSTM(nn.Module):
    '''
    Encoder that gives a sentence representation based on the mean of all tokens of the sentence.
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding_table, device):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        #trainable lookup table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_size).to(self.device)

        # copy pre-trained word vectors into embeddings table
        self.embed.weight.data.copy_(torch.from_numpy(embedding_table))
        # disable training the pre-trained embeddings
        self.embed.weight.requires_grad = False

        #LSTM cell
        self.forward_LSTM = nn.LSTMCell(embedding_size, hidden_size)
        self.backward_LSTM = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, inputs):
        inputs[inputs > self.vocab_size] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #hidden and cell states initialization
        hidden_state_forward = torch.zeros((len(embeds), self.hidden_size))
        cell_state_forward = torch.zeros((len(embeds), self.hidden_size))

        hidden_state_backward = torch.zeros((len(embeds), self.hidden_size))
        cell_state_backward = torch.zeros((len(embeds), self.hidden_size))
        #compute the last hidden state for the forward LSTM
        for i in range(0, len(embeds[0])):
            hidden_state_forward, _ = self.forward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))

        #compute the last hidden state for the backward LSTM
        for i in range(len(embeds[0]) -1, -1, -1):
            hidden_state_backward, _ = self.backward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))

        #return the concatenation of both hidden_states
        return torch.cat((hidden_state_forward, hidden_state_backward), dim = 1)
    
class Classifier(nn.Module):
    '''
    Classifier taking an encoder as input to create a relation vector, passed then in a multi-layer perceptron before a softmax layer.
    '''
    def __init__(self, encoder, embedding_dim, device):
        super().__init__()
        self.device = device
        #initialize the encoder
        self.encoder = encoder

        #set up the multilayer perceptron and the softmax layer
        self.linear = nn.Linear(4 * embedding_dim, 3).to(self.device)
        self.softmax = nn.Softmax(dim = 1).to(self.device)

    def forward(self, input_p, input_h):   
        #create the relational vector that is feed to the mutli-layer perceptron
        embed_p = self.encoder.forward(input_p)
        embed_h = self.encoder.forward(input_h)
        self.relation_vector = torch.cat((embed_p, embed_h, torch.abs(torch.sub(embed_p,embed_h)), torch.mul(embed_p, embed_h)), dim=1).to(self.device)

        #input of size (1x1200) fed to the multi layer perceptron
        logits = self.linear(self.relation_vector)
        logits = self.softmax(logits)

        return logits



