from datasets import load_dataset
import re
import string
import unicodedata
import nltk
from collections import Counter
import math

from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
nltk.download("punkt_tab")
punct_set = set(string.punctuation)
from nltk.corpus import stopwords
nltk.download('stopwords')

import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim

from load_dataset import get_dataset
from preprocessing import processing_batch, build_vocab, vocab_map
from data_classes import SentiDataset, sentiment_collate_fn
from models.rnn import RNNScratch
from trainer import training_loop, testing_loop



KEEP_WORDS = {
    "not", "no", "nor", "never",
    "but", "very", "too", "so"
}

CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "don't": "do not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "they've": "they have",
    "we've": "we have",
}


train_ds, val_ds, test_ds = get_dataset()


stop_words = set(stopwords.words("english")) - KEEP_WORDS

stop_words = set(stopwords.words("english")) - KEEP_WORDS

tokenized_train = train_ds.map(
    processing_batch,
    batched=True,
    batch_size=1000,
    fn_kwargs={
        "contractions": CONTRACTIONS,
        "stop_words": stop_words
    }
)

tokenized_val = val_ds.map(
    processing_batch,
    batched=True,
    batch_size=1000,
    fn_kwargs={
        "contractions": CONTRACTIONS,
        "stop_words": stop_words
    }
)

tokenized_test = test_ds.map(
    processing_batch,
    batched=True,
    batch_size=1000,
    fn_kwargs={
        "contractions": CONTRACTIONS,
        "stop_words": stop_words
    }
)

tokenized_train = tokenized_train.filter(lambda x: len(x["tokens"]) > 0)
tokenized_val = tokenized_val.filter(lambda x: len(x["tokens"]) > 0)
tokenized_test = tokenized_test.filter(lambda x: len(x["tokens"]) > 0)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

vocab, id_to_word, counter = build_vocab(tokenized_train[:1000], 5,PAD_TOKEN,UNK_TOKEN)

conv_data_train, labels_train = vocab_map(tokenized_train[:1000],vocab,UNK_TOKEN)
conv_data_val, labels_val = vocab_map(tokenized_val[:1000],vocab,UNK_TOKEN)
conv_data_test, labels_test = vocab_map(tokenized_test[:1000],vocab,UNK_TOKEN)


train_dataset = SentiDataset(conv_data_train,labels_train)
val_dataset = SentiDataset(conv_data_val,labels_val)
test_dataset = SentiDataset(conv_data_test,labels_test)

train_loader = DataLoader(train_dataset,batch_size=32,collate_fn=sentiment_collate_fn,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,collate_fn=sentiment_collate_fn,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,collate_fn=sentiment_collate_fn,shuffle=False)

model = RNNScratch(
    vocab_size=len(vocab),
    embed_dim=100,
    num_hiddens=128,
    output_size=2,
    pad_idx=0
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_loop(model, train_loader, val_loader, 1, loss_fn, optimizer)
testing_loop(model, test_loader, loss_fn)