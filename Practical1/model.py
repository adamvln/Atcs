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

        #dimension output
        self.output_dim = 300

    def forward(self, inputs):
        # inputs[inputs > self.vocab_size - 1] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #mean over all encoders
        logits = torch.mean(embeds, dim = 1)
        return logits
    
class Unidir_LSTM(nn.Module):
    '''
    Encoder that gives a sentence representation based on the last hidden_states of the last LSTM cell.
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
        self.LSTM_layer = nn.LSTMCell(embedding_size, hidden_size).to(self.device)

        #dimension output
        self.output_dim = hidden_size

    def forward(self, inputs):
        # inputs[inputs > self.vocab_size - 1] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #hidden and cell states initialization
        hidden_state = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        cell_state = torch.zeros((len(embeds), self.hidden_size)).to(self.device)

        #only the last hidden state matters
        for i in range(0, len(embeds[0])):
            hidden_state, _ = self.LSTM_layer(embeds[:,i], (hidden_state, cell_state))

        return hidden_state
    
class Bidirect_LSTM(nn.Module):
    '''
    Encoder that gives a sentence representation based on the concatenation of the last hidden states for two forward and backward LSTM.
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
        self.forward_LSTM = nn.LSTMCell(embedding_size, hidden_size).to(self.device)
        self.backward_LSTM = nn.LSTMCell(embedding_size, hidden_size).to(self.device)

        #dimension output
        self.output_dim = hidden_size * 2

    def forward(self, inputs):
        # inputs[inputs > self.vocab_size - 1] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #hidden and cell states initialization
        hidden_state_forward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        cell_state_forward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)

        hidden_state_backward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        cell_state_backward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        #compute the last hidden state for the forward LSTM
        for i in range(0, len(embeds[0])):
            hidden_state_forward, _ = self.forward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))

        #compute the last hidden state for the backward LSTM
        for i in range(len(embeds[0]) -1, -1, -1):
            hidden_state_backward, _ = self.backward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))

        #return the concatenation of both hidden_states
        return torch.cat((hidden_state_forward, hidden_state_backward), dim = 1)
    
class Bidirect_LSTM_Max_Pooling(nn.Module):
    '''
    Encoder that gives a sentence representation based on the max pooling over the concatenation of the hidden states for two forward and backward LSTM.
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
        self.forward_LSTM = nn.LSTMCell(embedding_size, hidden_size).to(self.device)
        self.backward_LSTM = nn.LSTMCell(embedding_size, hidden_size).to(self.device)

        #dimension output
        self.output_dim = hidden_size * 2



    def forward(self, inputs):
        # inputs[inputs > self.vocab_size - 1] = 0
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #hidden and cell states initialization
        hidden_state_forward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        cell_state_forward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)

        hidden_state_backward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)
        cell_state_backward = torch.zeros((len(embeds), self.hidden_size)).to(self.device)

        hidden_states_f = []
        hidden_states_b = []

        #compute the last hidden state for the forward LSTM
        for i in range(0, len(embeds[0])):
            hidden_state_forward, _ = self.forward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))
            hidden_states_f.append(hidden_state_forward)
        #compute the last hidden state for the backward LSTM
        for i in range(len(embeds[0]) -1, -1, -1):
            hidden_state_backward, _ = self.backward_LSTM(embeds[:,i], (hidden_state_backward, cell_state_backward))
            hidden_states_b.append(hidden_state_backward)

        #reverse the backward list to prepare concatenation
        hidden_states_b = hidden_states_b[::-1]

        #list to store concatenated tensor
        concatenated_h_states = []

        # Iterate through the lists and concatenate the tensors element-wise
        for i in range(len(hidden_states_f)):
            concatenated_h_state = torch.cat((hidden_states_f[i], hidden_states_b[i]), dim=1)
            concatenated_h_states.append(concatenated_h_state)

        #prepare max pooling
        concatenate_stack = torch.stack(concatenated_h_states, dim = 0)
        max_pooled, _ = torch.max(concatenate_stack, dim = 0)

        return max_pooled
    
class Classifier(nn.Module):
    '''
    Classifier taking an encoder as input to create a relation vector, passed then in a multi-layer perceptron before a softmax layer.
    '''
    def __init__(self, encoder, hidden_dim, n_classes, device):
        super().__init__()
        self.device = device
        #initialize the encoder
        self.encoder = encoder

        #set up the multilayer perceptron
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.encoder.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_classes)).to(self.device)

    def forward(self, input_p, input_h):   
        #create the relational vector that is feed to the mutli-layer perceptron
        embed_p = self.encoder(input_p)
        embed_h = self.encoder(input_h)
        self.relation_vector = torch.cat((embed_p, embed_h, torch.abs(torch.sub(embed_p,embed_h)), torch.mul(embed_p, embed_h)), dim=1).to(self.device)

        logits = self.classifier(self.relation_vector)

        return logits



