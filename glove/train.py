import os
from load_dataset import get_dataset
from preprocessing import processing_batch, build_vocab, vocab_map
from datasets_classes import GloveDataset
from models import GloveModel
from trainer import training_loop, testing_loop
from utils import weight_function, loss_fxn


import torch 
from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import functional as F
from torch import optim


dataset_length = 1000
num_epochs = 1
embedding_dim = 3

os.makedirs("save", exist_ok=True)

train_ds, val_ds, test_ds = get_dataset()

tokenized_train = train_ds.map(processing_batch, batched=True, batch_size=1000)
tokenized_train = tokenized_train.filter(lambda x: len(x["tokens"]) > 0)

tokenized_val = val_ds.map(processing_batch, batched=True, batch_size=1000)
tokenized_val = tokenized_train.filter(lambda x: len(x["tokens"]) > 0)

tokenized_test = test_ds.map(processing_batch, batched=True, batch_size=1000)
tokenized_test = tokenized_train.filter(lambda x: len(x["tokens"]) > 0)


vocab, id_to_word, counter = build_vocab(tokenized_train[:dataset_length], 30)

conv_data_train = vocab_map(tokenized_train[:dataset_length],vocab)
conv_data_val = vocab_map(tokenized_val[:dataset_length],vocab)
conv_data_test = vocab_map(tokenized_test[:dataset_length],vocab)


glove_dataset_train = GloveDataset(vocab,conv_data_train,2)
glove_dataloader_train = DataLoader(glove_dataset_train,batch_size=10,shuffle=True)

glove_dataset_val = GloveDataset(vocab,conv_data_val,2)
glove_dataloader_val = DataLoader(glove_dataset_val,batch_size=10,shuffle=True)

glove_dataset_test = GloveDataset(vocab,conv_data_test,2)
glove_dataloader_test = DataLoader(glove_dataset_test,batch_size=10,shuffle=True)

glove_model = GloveModel(len(vocab),embedding_dim)

optimizer = torch.optim.Adagrad(glove_model.parameters(),0.05)

training_loop(glove_model,glove_dataloader_train,glove_dataloader_val,2,loss_fxn,optimizer,weight_function)

testing_loop(glove_model,glove_dataloader_test,loss_fxn,weight_function)

torch.save(
            {
                "model_state_dict": glove_model.state_dict(),
                "word_to_id": vocab,
                "id_to_word": id_to_word,
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim,
            },
            "./save/model_checkpoint_glove_model.pt"
        )