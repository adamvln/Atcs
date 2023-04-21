# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import argparse

import torch
import json 
import pickle
import os 

from model import Average_Encoder, Unidir_LSTM, Bidirect_LSTM, Bidirect_LSTM_Max_Pooling, Classifier

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = '../SentEval/'
# path to the NLP datasets 

PATH_TO_DATA = '../SentEval/data'
# path to glove embeddings
PATH_TO_VEC = 'pretrained/glove.840B.300d.txt'


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import data

# import SentEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed manually to make runs reproducible
# You need to set this again if you do multiple runs of the same model
torch.manual_seed(42)

# When running on the CuDNN backend two further options must be set for reproducibility
if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    # _, params.word2id = data.create_dictionary(samples)
    # # load glove/word2vec format 
    # params.word_vec = data.get_wordvec(PATH_TO_VEC, params.word2id)
    # # dimensionality of glove embeddings
    # params.wvec_dim = 300
    return

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab):
    """
    Map tokens to their IDs for a single example
    """
    batch_size = len(mb)
    max_len = max([len(ex) for ex in mb])
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [pad([vocab.get(t, 0) for t in ex], max_len) for ex in mb]
    
    x = torch.LongTensor(x)
    x = x.to(device)

    return x

def batcher(params, batch, encoder, vocab):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """



    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]

    batch = [word.lower() for word in batch]

    batch = prepare_minibatch(batch, vocab_w2i)
    logits = encoder(batch).cpu()


    return logits.detach().numpy()



    # embeddings = []

    # for sent in batch:
    #     sentvec = []
    #     # the format of a sentence is a lists of words (tokenized and lowercased)
    #     for word in sent:
    #         if word in params.word_vec:
    #             # [number of words, embedding dimensionality]
    #             sentvec.append(params.word_vec[word])
    #     if not sentvec:
    #         vec = np.zeros(params.wvec_dim)
    #         # [number of words, embedding dimensionality]
    #         sentvec.append(vec)
    #     # average of word embeddings for sentence representation
    #     # [embedding dimansionality]
    #     sentvec = np.mean(sentvec, 0)
    #     embeddings.append(sentvec)
    # # [batch size, embedding dimensionality]
    # embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_model', type = str)
    args = parser.parse_args()

    print("Loading of Vocabulary")
    with open('../Practical1/data/data.json', 'r') as infile:
        vocab_w2i = json.load(infile)
    print("Vocabulary loaded")

    print("Loading of Embedding Matrix")
    file_name = 'data/embedding_matrix.pickle'
    # Open the file in read-binary ('rb') mode
    with open(file_name, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
    

    model = torch.load(args.path_to_model)

    from functools import partial
    my_batcher = partial(batcher, encoder = model.encoder, vocab = vocab_w2i)
    se = senteval.engine.SE(params_senteval, my_batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    # senteval prints the results and returns a dictionary with the scores
    print(os.getcwd())
    results = se.eval(transfer_tasks)
    print(results)