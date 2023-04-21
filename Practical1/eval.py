import torch
import torch.nn as nn
import torch.optim as optim
import os 
import argparse
import pickle
import json

from data import load_data, load_data_debug, Vocabulary, prepare_minibatch
from torch.utils.data import DataLoader, Dataset
from model import Classifier, Average_Encoder, Unidir_LSTM, Bidirect_LSTM, Bidirect_LSTM_Max_Pooling
from tqdm import tqdm

def evaluate(model, test_loader):
    criterion = nn.CrossEntropyLoss() # specify loss function
    model.eval() # set model to evaluation mode
    total_loss = 0
    valid_preds = 0
    with torch.no_grad():
        for (x_premise, x_hypo, y_label) in test_loader:

            y_pred = model(x_premise, x_hypo)

            loss = criterion(y_pred, y_label)

            total_loss += loss.item() 

            _, predicted = torch.max(y_pred, 1)
            valid_preds += (predicted == y_label).sum().item()

    avg_loss = total_loss / len(test_loader.dataset) # calculate average loss
    accuracy = valid_preds / len(test_loader.dataset) # calculate accuracy
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, valid_preds, len(test_loader.dataset),
        100. * accuracy))
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default = 'average', type = str, help = "encoders to use")
    parser.add_argument('--path_to_model', type = str)
    # parser.add_argument('--model_name', default = 'test', type = str, help = "model name to save in files")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_data()
    # vocab = Vocabulary(dataset)
    # vocab.build() 

    print("Loading of Vocabulary")
    with open('data/data.json', 'r') as infile:
        vocab_w2i = json.load(infile)
    print("Vocabulary loaded")

    file_name = 'data/embedding_matrix.pickle'
    # Open the file in read-binary ('rb') mode
    with open(file_name, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)

    model = torch.load(args.path_to_model)

    from functools import partial
    my_collate_fn = partial(prepare_minibatch, vocab = vocab_w2i)
    test_loader = DataLoader(dataset["test"], batch_size = 64, shuffle = True, collate_fn = my_collate_fn)

    evaluate(model, test_loader)

if __name__ == "__main__":
    main()