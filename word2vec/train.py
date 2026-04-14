from load_dataset import get_dataset
from preprocessing import processing_batch, build_vocab, vocab_map
from datasets_classes import SkipgramDataset, CbowDataset, cbow_collate
from models import SkipgramModel, CbowModel

from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import functional as F
from torch import optim


dataset_length = 1000
w2v_model = "skip-gram"
num_epochs = 1

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
    
    model = SkipgramModel(len(vocab),3)
    
elif(w2v_model == "cbow"):

    dataset_train = CbowDataset(vocab,id_to_word,conv_data_train,2)
    dataloader_train = DataLoader(dataset_train,batch_size=10,shuffle=True,collate_fn=cbow_collate)

    dataset_val = CbowDataset(vocab,id_to_word,conv_data_val,2)
    dataloader_val = DataLoader(dataset_val,batch_size=10,shuffle=True,collate_fn=cbow_collate)

    dataset_test = CbowDataset(vocab,id_to_word,conv_data_test,2)
    dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True,collate_fn=cbow_collate)
    
    model = CbowModel(len(vocab),3,len(vocab))
    
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)


for epoch in range(num_epochs):

    train_loss = 0

    for pairs in dataloader_train:

      logits = model(pairs[0])
      loss = loss_fn(logits,pairs[1])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
    avg_train_loss = train_loss / len(dataloader_train)


    model.eval()
    val_loss = 0

    for pairs in dataloader_val:
      inputs, targets = pairs[0], pairs[1]
      logits = model(inputs)
      loss = loss_fn(logits,targets)
      val_loss += loss.item()

    avg_val_loss = val_loss/ len(dataloader_val)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    
    
    
    
    
    
