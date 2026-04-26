import torch 
from torch.utils.data import Dataset 

class SentiDataset(Dataset):

  def __init__(self,conv_data,labels):
    self.conv_data = conv_data
    self.labels = labels

  def __len__(self):
    return len(self.conv_data)

  def __getitem__(self,idx):
    return self.conv_data[idx],self.labels[idx]


def sentiment_collate_fn(batch):

  input_ids = []
  labels = []
  lengths = []

  for ids, label in batch:
    input_ids.append(ids)
    labels.append(label)
    lengths.append(len(ids))

  max_len = max(lengths)

  padded_input_ids = []

  for ids in input_ids:
    padded = ids + [0] * (max_len - len(ids))
    padded_input_ids.append(padded)

  padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
  labels = torch.tensor(labels,dtype=torch.long)
  lengths = torch.tensor(lengths,dtype=torch.long)

  return padded_input_ids, labels, lengths