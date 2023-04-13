import torch.nn as nn
import torch

class Average_Encoder:
    '''
    Encoder that gives a sentence representation based on the mean of all tokens of the sentence.
    '''
    def __init__(self, vocab_size, embedding_size, embedding_table, device):
        self.device = device
        #trainable lookup table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_size).to(self.device)

        # copy pre-trained word vectors into embeddings table
        self.embed.weight.data.copy_(torch.from_numpy(embedding_table))
        # disable training the pre-trained embeddings
        self.embed.weight.requires_grad = False

    def forward(self, inputs):
        #this should output a (L x embedding_size) matrix
        embeds = self.embed(inputs)
        #mean over all encoders
        logits = torch.mean(embeds, dim = 1)
        return logits


class Classifier:
    '''
    Classifier taking an encoder as input to create a relation vector, passed then in a multi-layer perceptron before a softmax layer.
    '''
    def __init__(self, encoder, embedding_dim, device):
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



