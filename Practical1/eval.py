import torch
import torch.nn as nn
import torch.optim as optim
import os 
import argparse
import pickle

from data import load_data, load_data_debug, Vocabulary, prepare_minibatch
from torch.utils.data import DataLoader, Dataset
from model import Classifier, Average_Encoder, Unidir_LSTM, Bidirect_LSTM, Bidirect_LSTM_Max_Pooling
from tqdm import tqdm

def evaluate(model, test_loader):
    criterion = nn.CrossEntropyLoss() # specify loss function
    model.eval() # set model to evaluation mode
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for (x_premise, x_hypo, y_label) in test_loader:

            y_pred = model(x_premise, x_hypo)

            loss = criterion(y_pred, y_label)

            total_loss += loss.item() 

            pred = y_pred.argmax(dim=1, keepdim=True) # get index of predicted class

            total_correct += pred.eq(y_label.view_as(pred)).sum().item() # accumulate correct predictions

    avg_loss = total_loss / len(test_loader.dataset) # calculate average loss
    accuracy = total_correct / len(test_loader.dataset) # calculate accuracy
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, total_correct, len(test_loader.dataset),
        100. * accuracy))
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default = 'average', type = str, help = "encoders to use")
    # parser.add_argument('--model_name', default = 'test', type = str, help = "model name to save in files")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_data_debug()
    vocab = Vocabulary(dataset)
    vocab.build() 

    file_name = 'Practical1\data\embedding_matrix.pickle'
    # Open the file in read-binary ('rb') mode
    with open(file_name, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
    
    if args.encoder == 'Average':
        encoder = Average_Encoder(len(data), 300, data, device)
    elif args.encoder == 'UniLSTM':
        encoder = Unidir_LSTM(len(data), 300, 2048, data, device)
    elif args.encoder == 'BiLSTM':
        encoder = Bidirect_LSTM(len(data), 300, 2048, data, device)
    elif args.encoder == 'BiLSTMConcat':
        encoder = Bidirect_LSTM_Max_Pooling(len(data), 300, 2048, data, device)

    model = Classifier(encoder, 300, device)

    from functools import partial
    my_collate_fn = partial(prepare_minibatch, vocab = vocab)
    test_loader = DataLoader(dataset["test"], batch_size = 4, shuffle = True, collate_fn = my_collate_fn)

    evaluate(model, test_loader)

if __name__ == "__main__":
    main()