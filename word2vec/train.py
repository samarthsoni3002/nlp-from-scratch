from load_dataset import get_dataset
from preprocessing import processing_batch, build_vocab, vocab_map
from datasets_classes import SkipgramDataset, CbowDataset, cbow_collate
from models import SkipgramModel, SkipgramModelNegativeSampling, CbowModel
from trainer import training_loop, training_loop_negative_sampling, testing_loop
from utils import negative_sampling_loss, build_noise_distribution


import torch 
from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import functional as F
from torch import optim


dataset_length = 1000
w2v_model = "cbow"
negative_sampling = False 
num_epochs = 1
neg_samples = 5

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

if(w2v_model == "skip-gram"):
    
    dataset_train = SkipgramDataset(vocab,id_to_word,conv_data_train,2)
    dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True)

    dataset_val = SkipgramDataset(vocab,id_to_word,conv_data_val,2)
    dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True)

    dataset_test = SkipgramDataset(vocab,id_to_word,conv_data_test,2)
    dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True)
    
    
    if(negative_sampling==False):
        model = SkipgramModel(len(vocab),3)
    
elif(w2v_model == "cbow"):
    
    print("--- Model Selection -> CBOW --- ")

    pad_id = len(vocab)

    dataset_train = CbowDataset(vocab,id_to_word,conv_data_train,2)
    dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))

    dataset_val = CbowDataset(vocab,id_to_word,conv_data_val,2)
    dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))

    dataset_test = CbowDataset(vocab,id_to_word,conv_data_test,2)
    dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))
    
    if(negative_sampling==False):
        model = CbowModel(len(vocab),3,len(vocab))
    
    
    
if(negative_sampling):
    
    model = SkipgramModelNegativeSampling(len(vocab),3)
    
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    
    print("--- Training With Negative Sampling ---")
    
    training_loop_negative_sampling(model,dataloader_train,dataloader_val,num_epochs,negative_sampling_loss,optimizer,neg_samples,build_noise_distribution,counter,vocab)
    
else:
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    
    print("--- Training ---")
    
    training_loop(model,dataloader_train,dataloader_val,num_epochs,loss_fn,optimizer)
    testing_loop(model,dataloader_test,loss_fn)


    
    
    
    
