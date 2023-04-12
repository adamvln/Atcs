from data import load_data, load_data_debug
from tqdm import tqdm
import numpy as np
import json
import pickle

class Vocabulary:
    def __init__(self, dataset):
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]

        self.w2i = {}
        self.i2w = {}

        self.embedding_table = []

    def embedding_build(self, embedding_dict):

        for word in tqdm(self.w2i):
                self.embedding_table.append(embedding_dict[word])

    def build(self):
        self.w2i["<unk>"] = 0
        self.w2i["<pad>"] = 1
        index_token = 2

        splits = [self.train_dataset, self.val_dataset, self.test_dataset]
  
        for split in splits:
            for i in tqdm(range(len(split))):
                for token in [*split[i]["premise"], *split[i]["hypothesis"]]:
                    if token not in self.w2i:
                        self.w2i[token] = index_token
                        index_token += 1

        self.i2w = {value: key for key, value in self.w2i.items()}


def filereader(path): 
  with open(path, mode="r", encoding="utf-8") as f:
    for line in f:
      yield line.strip().replace("\\","")

# def data_to_embeddings(vocab, path_to_pretrained):

def embedding_dict(path):

    i = 0
    embedding_table = {}
    for line in tqdm(filereader(path)):
        embedding_table[line.split(" ",1)[0]] = line.split(" ",1)[1].split()


    vocab = Vocabulary(load_data())
    vocab.build()

    embedding_matrix = np.zeros((len(vocab.w2i), 300))
    for i,word in tqdm(enumerate(vocab.w2i)):
        if word in embedding_table:
            embedding_matrix[i] = [np.float32(embedding) for embedding in embedding_table[word]]

    with open("data\embedding_matrix.pickle", 'wb') as f:
        pickle.dump(embedding_matrix, f)


    return embedding_table


if __name__ == "__main__":
    # word_2_index, index_2_word = vocabulary(load_data_debug())
    # print(index_2_word)
    path = "data\glove.840B.300d.txt"
    # lower = False
    # i = 0
    # for line in filereader(path):
    #     print(line.split(" ",1)[0])
    #     print(line.split(" ",1)[1].split())
    #     i += 1
    #     if i >4:
    #         break
    embedding_dict(path)
