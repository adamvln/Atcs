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

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, training_name, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        valid_loss = 0.0
        valid_preds = 0
        model.train()
        for (x_premise, x_hypo, y_label) in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_premise, x_hypo)
            loss = criterion(y_pred, torch.abs(y_label))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

        train_avg_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for (x_premise, x_hypo, y_label) in val_loader:

                y_pred = model(x_premise, x_hypo)
                loss = criterion(y_pred, torch.abs(y_label))
                valid_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)
                valid_preds += (predicted == y_label).sum().item()

            valid_avg_loss = valid_loss / len(val_loader)

        checkpoint_path = os.path.join("Practical1/model_checkpoint", f'{training_name}_checkpoint_epoch{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_avg_loss,
            'valid_loss' : valid_avg_loss,
            'correct_preds' : valid_preds
        }, checkpoint_path)

        print(f'Epoch {epoch}: Train loss = {train_avg_loss:.4f}, Valid loss = {valid_avg_loss:.4f}, Valid accuracy = {valid_preds/len(val_loader.dataset):.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.1, type = float, help = "learning_rate")
    parser.add_argument('--epochs', default = 10, type = int, help = "epochs")
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--encoder', default = 'Average', type = str, help = "encoders to use")
    parser.add_argument('--model_name', default = 'test', type = str, help = "model name to save in files")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading of dataset")
    dataset = load_data()
    # vocab = Vocabulary(dataset)
    # vocab.build()
    print("Dataset loaded")

    # with open('data.json', 'w') as infile:
    #     json.dump(vocab.w2i, infile)

    print("Loading of Vocabulary")
    with open('Practical1\data.json', 'r') as infile:
        vocab_w2i = json.load(infile)
    print("Vocabulary loaded")

    file_name = 'Practical1\data/embedding_matrix.pickle'
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
    my_collate_fn = partial(prepare_minibatch, vocab = vocab_w2i)
    train_loader = DataLoader(dataset["train"], batch_size = args.batch_size, shuffle = True, collate_fn = my_collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size = args.batch_size, shuffle = True, collate_fn = my_collate_fn)

    print("Training begins")
    train_model(model, train_loader, val_loader, args.epochs,args.lr, args.model_name, device)

if __name__ == "__main__":
    main()
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
#     # dataset = load_data_debug()
#     # vocab = Vocabulary(dataset)
#     # vocab.build() 

#     # import pickle
#     # # Replace 'file_name.pkl' with the name of your pickle file
#     # file_name = 'Practical1\data\embedding_matrix.pickle'

#     # # Open the file in read-binary ('rb') mode
#     # with open(file_name, 'rb') as file:
#     #     # Load the data from the file
#     #     data = pickle.load(file)

#     # model = Classifier(Average_Encoder(len(data), 300, data, device), 300, device)

#     # from functools import partial
#     # my_collate_fn = partial(prepare_minibatch, vocab = vocab)
#     # train_loader = DataLoader(dataset["train"], batch_size = 8, shuffle = True, collate_fn = my_collate_fn)
#     # val_loader = DataLoader(dataset["validation"], batch_size = 8, shuffle = True, collate_fn = my_collate_fn)

#     # train_model(model, train_loader, val_loader, 1, 0.01, "test", device)












#     from data import load_data_debug, Vocabulary, prepare_minibatch
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataset = load_data_debug()
#     vocab = Vocabulary(dataset)
#     vocab.build() 

#     from torch.utils.data import DataLoader
#     from functools import partial
#     my_collate_fn = partial(prepare_minibatch, vocab = vocab)
#     train_loader = DataLoader(dataset["train"], batch_size = 64, shuffle = True, collate_fn = my_collate_fn)
#     val_loader = DataLoader(dataset["validation"], batch_size = 64, shuffle = True, collate_fn = my_collate_fn)
#     import pickle
#     # Replace 'file_name.pkl' with the name of your pickle file
#     file_name = 'Practical1\data\embedding_matrix.pickle'

#     # Open the file in read-binary ('rb') mode
#     with open(file_name, 'rb') as file:
#         # Load the data from the file
#         data = pickle.load(file)

#     from model import Average_Encoder, Classifier, Unidir_LSTM, Bidirect_LSTM,Bidirect_LSTM_Max_Pooling
#     encoder_1 = Unidir_LSTM(len(data), 300, 2048, data, device) 
#     encoder_2 = Average_Encoder(len(data), 300, data, device)
#     encoder_3 = Bidirect_LSTM(len(data), 300, 2048, data, device)
#     encoder_4 = Bidirect_LSTM_Max_Pooling(len(data), 300, 2048, data, device)

#     classifier = Classifier(encoder, 300, device)

#     train_model(classifier, train_loader, val_loader, 1, 0.01, "test", device)