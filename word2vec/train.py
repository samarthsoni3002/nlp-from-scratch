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

train_ds = get_dataset()

tokenized_train = train_ds.map(processing_batch, batched=True, batch_size=1000)
tokenized_train = tokenized_train.filter(lambda x: len(x["tokens"]) > 0)


vocab, id_to_word, counter = build_vocab(tokenized_train[:dataset_length], 30)
conv_data = vocab_map(tokenized_train[:dataset_length],vocab)

if(w2v_model == "skip-gram"):
    
    dataset = SkipgramDataset(vocab,id_to_word,conv_data,2)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=True)
    model = SkipgramModel(len(vocab),3)
    
else:
    dataset = CbowDataset(vocab,id_to_word,conv_data,2)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=True,collate_fn=cbow_collate)
    model = CbowModel(len(vocab),3,len(vocab))
    
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)


for epoch in range(num_epochs):

    epoch_loss = 0

    for pairs in dataloader:

      logits = model(pairs[0])
      loss = loss_fn(logits,pairs[1])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
    epoch_loss_average = epoch_loss / len(dataloader)

    print(f"epoch :- {epoch+1} || loss :- {epoch_loss_average}")  
    
    
    
    
    
    
