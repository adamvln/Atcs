from datasets import load_dataset
from tqdm import tqdm

import nltk
nltk.download('punkt')

import numpy as np
import json
import pickle
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed manually to make runs reproducible
# You need to set this again if you do multiple runs of the same model
torch.manual_seed(42)

# When running on the CuDNN backend two further options must be set for reproducibility
if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def load_data():
    '''
    Loads the SNLI dataset and apply a preprocessing.

    Return : 
        snli_dataset : dataset preprocessed
    '''
    snli_dataset = load_dataset("snli")

    for split in ['train', 'validation', 'test']:
        snli_dataset[split] = snli_dataset[split].map(preprocess)
        snli_dataset[split] = snli_dataset[split].filter(filter_examples)

    return snli_dataset

def preprocess(example):
    '''
    Apply preprocessing, currently tokenization and lowercasing.
    Args :
        example : iteration of the split of the dictionnary containing the SNLI dataset, of the form
            {'premise' : sentence1, 'hypothesis' : sentence2, label}
    
    Return : 
        example object preprocessed
    '''
    
    example["premise"] = nltk.tokenize.word_tokenize(example["premise"])
    example["hypothesis"] = nltk.tokenize.word_tokenize(example["hypothesis"])

    lowercase_premise = [words.lower() for words in example["premise"]]
    lowercase_hypothesis = [words.lower() for words in example["hypothesis"]]

    example["premise"] = lowercase_premise
    example["hypothesis"] = lowercase_hypothesis

    return example

def filter_examples(example):
    '''
    Filters out examples with label -1.
    Args :
        example : iteration of the split of the dictionnary containing the SNLI dataset, of the form
            {'premise' : sentence1, 'hypothesis' : sentence2, label}
    
    Return : 
        True if example should be kept, False if it should be discarded
    '''
    return example["label"] != -1

def load_data_debug():
    snli_dataset = load_dataset("snli")
    
    snli_dataset['test'] = snli_dataset['test'].map(preprocess)
    snli_dataset['train'] = snli_dataset['test']
    snli_dataset['validation'] = snli_dataset['test']

    return snli_dataset

class Vocabulary:
    '''
    Vocabulary class that gives an index to every word that composes the dataset
    '''
    def __init__(self, dataset):
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]

        self.w2i = {}
        self.i2w = {}

    def build(self):
        '''
        build self.w2i that is a dictionnary with word as keys and index as value
        '''
        self.w2i["<unk>"] = 0
        self.w2i["<pad>"] = 1
        index_token = 2

        for i in tqdm(range(len(self.train_dataset))):
            for token in [*self.train_dataset[i]["premise"], *self.train_dataset[i]["hypothesis"]]:
                if token not in self.w2i:
                    self.w2i[token] = index_token
                    index_token += 1

        self.i2w = {value: key for key, value in self.w2i.items()}


def filereader(path): 
    '''
    Read line by line the glove pretrained vector file
    Args:
        path : path of the .txt with the glove embeddings
    '''
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\","")

# def data_to_embeddings(vocab, path_to_pretrained):

def embedding_dict(path):
    '''
    Create the embedding matrix, with unknown and padding token at line 1 and 2
    Args:
        path : path of the .txt with the glove embeddings
    '''

    #build the vocabulary based on the load_data function
    print("Construction of the vocabulary")
    vocab = Vocabulary(load_data_debug())
    vocab.build()    
    print("Vocabulary built")

    #initialize the embedding matrix
    # embedding_matrix = []
    # embedding_matrix.append(list(np.random.uniform(-1, 1, (300,))))
    # embedding_matrix.append(list(np.random.uniform(-1, 1, (300,))))

    # print("Creation of embedding matrix")
    # for line in tqdm(filereader(path)):
    #     if line.split(" ",1)[0] in vocab.w2i:
    #         embedding_matrix.append(np.float32(line.split(" ",1)[1].split()))

    # embedding_matrix = np.stack(embedding_matrix, axis = 0)

    embedding_matrix = np.zeros((len(vocab.w2i), 300))
    embedding_matrix[0] = list(np.random.uniform(-1, 1, (300,)))
    embedding_matrix[1] = list(np.random.uniform(-1, 1, (300,)))

    print("Creation of embedding matrix")
    for line in tqdm(filereader(path)):
        if line.split(" ",1)[0] in vocab.w2i:
            # embedding_matrix.append(np.float32(line.split(" ",1)[1].split()))
            embedding_matrix[vocab.w2i[line.split(" ",1)[0]]] = np.float32(line.split(" ",1)[1].split())

    zero_rows = np.all(embedding_matrix == 0, axis = 1)
    embedding_matrix[zero_rows] = embedding_matrix[0]

    with open("data\embedding_matrix.pickle", 'wb') as f:
        pickle.dump(embedding_matrix, f)

    print("Embedding_matrix created")
    return embedding_matrix

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen_premise = max([len(ex["premise"]) for ex in mb])
    maxlen_hypo = max([len(ex["hypothesis"]) for ex in mb])
    maxlen = max(maxlen_hypo, maxlen_premise)
    
    x_premise = [pad([vocab.get(t, 0) for t in ex["premise"]], maxlen) for ex in mb]
    x_hypo = [pad([vocab.get(t, 0) for t in ex["hypothesis"]], maxlen) for ex in mb]

    x_premise = torch.LongTensor(x_premise)
    x_premise = x_premise.to(device)

    x_hypo = torch.LongTensor(x_hypo)
    x_hypo = x_hypo.to(device)

    y = [ex["label"] for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x_premise, x_hypo, y

def prepare_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """
    
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example]
    
    x = torch.LongTensor([x])
    x = x.to(device)

    return x

    

if __name__ == "__main__":
# path = "data\glove.840B.300d.txt"     
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load_data()
    embedding_dict("data/glove.840B.300d.txt")
    # # Replace 'file_name.pkl' with the name of your pickle file
    # file_name = 'data\embedding_matrix.pickle'

    # # Open the file in read-binary ('rb') mode
    # with open(file_name, 'rb') as file:
    #     # Load the data from the file
    #     data = pickle.load(file)

    # example_1 = dataset["train"]["premise"][0]
    # a = prepare_example(example_1, vocab)
    # from model import Average_Encoder, Classifier
    # encoder = Average_Encoder(len(data), 300, data, device) 
    # classifier = Classifier(encoder, 300, device)

    # print(encoder)
    # dataset = load_data_debug()
    # vocab = Vocabulary(dataset)
    # vocab.build() 

    # from torch.utils.data import DataLoader
    # from functools import partial
    # my_collate_fn = partial(prepare_minibatch, vocab = vocab)
    # train_loader = DataLoader(dataset["train"], batch_size = 64, shuffle = True, collate_fn = my_collate_fn)

    # for batch in train_loader:
    #     x,y,z = batch[0], batch[1], batch[2]
    #     print(x)
    # print(train_loader)

    #PROBLEME AVEC LA DATA 
    # LE PROBLEME EST QUE LES INPUTS TENSORS QUI ONT DES MOTS REMPLACES PAR DES INDEX ONT UN INPUT PLUS GRAND QUE LA DATA MATRIX
    # C NORMAL CELA VEUT DIRE QUE CERTAINS MOTS NONT PAS DEMBEDDINGS DANS LE GLOVE PRETRAINED VECTOR SET 
    # CES MOTS DANS LA DAAT MATRIX SONT REFERES A DES UNKNOWN WORD IE DES INDEX 0 
    # JUSTE REMPLACONS LES PAR 0