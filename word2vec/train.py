import os
from load_dataset import get_dataset
from preprocessing import processing_batch, build_vocab, vocab_map
from datasets_classes import SkipgramDataset, CbowDataset, cbow_collate, cbow_collate_negative_sampling
from models import SkipgramModel, SkipgramModelNegativeSampling, CbowModel, CbowModelNegativeSampling
from trainer import training_loop, training_loop_negative_sampling_skipgram, training_loop_negative_sampling_cbow, testing_loop, testing_loop_negative_sampling_skipgram,testing_loop_negative_sampling_cbow
from utils import negative_sampling_loss, build_noise_distribution


import torch 
from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import functional as F
from torch import optim


dataset_length = 1000
w2v_model = "cbow"
negative_sampling = True 
num_epochs = 1
neg_samples = 5
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

if(w2v_model == "skip-gram"):
    
    print("--- Model Selection -> Skip-gram --- ")
    
    dataset_train = SkipgramDataset(vocab,id_to_word,conv_data_train,2)
    dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True)

    dataset_val = SkipgramDataset(vocab,id_to_word,conv_data_val,2)
    dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True)

    dataset_test = SkipgramDataset(vocab,id_to_word,conv_data_test,2)
    dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True)
    
    
    if(negative_sampling):
        
        model = SkipgramModelNegativeSampling(len(vocab),embedding_dim)

        optimizer = optim.Adam(model.parameters(),lr=0.01)

        print("--- Training With Negative Sampling ---")

        training_loop_negative_sampling_skipgram(model,dataloader_train,dataloader_val,num_epochs,negative_sampling_loss,optimizer,neg_samples,build_noise_distribution,counter,vocab)
        
        print("--- Testing With Negative Sampling ---")
        
        testing_loop_negative_sampling_skipgram(model, dataloader_test,negative_sampling_loss,neg_samples,build_noise_distribution,vocab,counter)
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "word_to_id": vocab,
                "id_to_word": id_to_word,
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim,
            },
            "./save/model_checkpoint_skipgram_ns.pt"
        )
        
    else: 
        
        model = SkipgramModel(len(vocab),embedding_dim)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.01)
    
        print("--- Training ---")
    
        training_loop(model,dataloader_train,dataloader_val,num_epochs,loss_fn,optimizer)
        
        print("--- Testing ---")
        
        testing_loop(model,dataloader_test,loss_fn)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "word_to_id": vocab,
                "id_to_word": id_to_word,
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim
            },
            "./save/model_checkpoint_skipgram.pt"
        )
        

    
elif(w2v_model == "cbow"):
    
    print("--- Model Selection -> CBOW --- ")

    pad_id = len(vocab)

    dataset_train = CbowDataset(vocab,id_to_word,conv_data_train,2)

    dataset_val = CbowDataset(vocab,id_to_word,conv_data_val,2)

    dataset_test = CbowDataset(vocab,id_to_word,conv_data_test,2)

    
    
    if(negative_sampling):
        
        dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate_negative_sampling(batch,counter, vocab))
        
        dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate_negative_sampling(batch,counter,vocab))
        
        dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate_negative_sampling(batch,counter,vocab))
        
        model = CbowModelNegativeSampling(len(vocab),embedding_dim,len(vocab))
        
        optimizer = optim.Adam(model.parameters(),lr=0.01)
        
        print("--- Training With Negative Sampling ---")
        
        training_loop_negative_sampling_cbow(model,dataloader_train, dataloader_val, num_epochs, negative_sampling_loss, optimizer)
        
        
        print("--- Testing With Negative Sampling ---")
        
        
        testing_loop_negative_sampling_cbow(model,dataloader_test,negative_sampling_loss)
 
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "word_to_id": vocab,
                "id_to_word": id_to_word,
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim,
                "pad_id": len(vocab),
            },
            "./save/model_checkpoint_cbow_ns.pt"
        )
        
        
    else:
        
        dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))
        
        dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))
        
        dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True,collate_fn=lambda batch: cbow_collate(batch,pad_id))
        
        model = CbowModel(len(vocab),embedding_dim,len(vocab))
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.01)
        
        
        print("--- Training ---")
        
        training_loop(model,dataloader_train,dataloader_val,num_epochs,loss_fn,optimizer)
        
        print("--- Testing ---")
        
        testing_loop(model,dataloader_test,loss_fn)
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "word_to_id": vocab,
                "id_to_word": id_to_word,
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim,
                "pad_id": len(vocab),
            },
            "./save/model_checkpoint_cbow.pt"
        )       
        
        
        
    
    

    
    
    
    
