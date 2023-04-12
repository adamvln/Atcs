from datasets import load_dataset
import nltk

def load_data():
    '''
    Loads the SNLI dataset and apply a preprocessing.

    Return : 
        snli_dataset : dataset preprocessed
    '''
    snli_dataset = load_dataset("snli")

    for split in ['train', 'validation', 'test']:
        snli_dataset[split] = snli_dataset[split].map(preprocess)

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

def load_data_debug():
    snli_dataset = load_dataset("snli")
    
    snli_dataset['test'] = snli_dataset['test'].map(preprocess)
    snli_dataset['train'] = snli_dataset['test']
    snli_dataset['validation'] = snli_dataset['test']

    return snli_dataset



