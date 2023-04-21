# ATCS : Reproduction of "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"

## Download link of pre-trained models 

https://drive.google.com/drive/folders/1-sZ6OJudssRCEkijydCYbY5Zn2QY-Yub?usp=sharing

## Package and Dependencies

The environment that I used is the one provided during DeepLearning 1. It is a course provided by Uva which has a github repository that can be found here : 
https://github.com/uvadlc/uvadlc_practicals_2022.git

This environment was not enough for the purpose of this work. Two libraries needed to be added with the following command lines : 

```
pip install datasets
conda install -c anaconda nltk
```

## Structure of the code

Be sure to be inside the Practical1 folder to run the python files in command lines.

The main code is located in 5 python files.
- data.py : this file loads the SNLI dataset that will be further used for training. It comprises preprocessing steps that tokenize and lowercase 
            the sentences of dataset. In addition, it creates the embedding matrix that is loaded in the data folder (data/embedding_matrix.pickle).
            This embedding matrix includes all pre-trained Glove embeddings of the SNLI dataset vocabulary. The words that do not have an embedding by
            the Glove embeddings are all given the same random embedding. The data.py file also saves a data.json file in the data folder which is the 
            vocabulary of the full SNLI dataset as a dictionnary. 
- model.py : this file contains all models used for the training:
            * Average_Encoder : Encoder that gives a sentence representation based on the mean of all embeddings of the sentence.
            * Unidir_LSTM : Encoder that gives a sentence representation based on the last hidden_state of a forward
            LSTM
            * Bidirect_LSTM : Encoder that gives a sentence representation based on the concatenation of the last hidden states of a bidirectional LSTM.
            * Bidirect_LSTM_Max_Pooling : Encoder that gives a sentence representation based on the max pooling over the concatenation of the hidden states of a bidirectional 
            LSTM.
            * Classifier : Classifier taking an encoder as input to create a relation vector, passed then in a multi-layer perceptron. 
- train.py : This file is the training file taking different flags as
            arguments such as : 
            * --lr : learning rate
            * --epochs : epochs
            * --batch_size : batch size
            * --encoder : name of the Encoder, choosing between Average, UniLSTM, BiLSTM, BiLSTMConcat
            * --model_name : name of the model to be used in the checkpoint path

- eval.py  : evaluate the model on the test SNLI dataset.
            Takes one flag as an argument : 
            * --path_to_model : path to checkpoint

- senteval_test.py : evaluate the model on the Senteval framework.
                     Takes one flag as an argument : 
                     * --path_to_model : path to checkpoint

## How to train and evaluate a model ?

Following these steps : 
- Download the pretrained word embeddings from Glove.
  Choose which embeddings to download on the official website : https://nlp.stanford.edu/projects/glove/ . Unzip the file and move it to Practical1/data folder.

- Run the data.py file. it will create the embedding matrix and store it in the data folder 
   by the name "data/glove.840B.300d.txt"

- Choose a model to train among the 4 different types and run the train.py file with the correct flags.

- Evaluation on the SNLI dataset by running eval.py with the appropriate flag leading to the model checkpoint
  located in the model_checkpoint folder.

- Evaluation on the SentEval Framework using senteval_eval.py using the path to the checkpoint model as flag. The SentEval repository is added as a submodule to this repository. 
  If working from the zip file, do not git clone SentEval but just follow their README to download the data. 

## Results

##On SNLI Dataset

                     | Validation | Test |
| Average Encoder    | 0.6027     | 0.6156 | EPOCH 13
| LSTM               | 0.8141     | 0.8072 | EPOCH 8
| BiLSTM             | 0.8095     | 0.8093 | EPOCH 6
| BiLSTM Max Pooling | 40         | Male   |