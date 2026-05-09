import torch 
from torch.utils.data import Dataset 


class TranslationDataset(Dataset):

  def __init__(self,X_conv,y_conv):
    self.X = X_conv
    self.y = y_conv

  def __len__(self):
    return len(self.X)

  def __getitem__(self,idx):
    return self.X[idx],self.y[idx]


def translation_collate_fn(batch, pad_id=0):
    input_ids = []
    output_ids = []
    ip_lengths = []
    op_lengths = []

    for ip_ids, op_ids in batch:
        input_ids.append(ip_ids)
        output_ids.append(op_ids)
        ip_lengths.append(len(ip_ids))
        op_lengths.append(len(op_ids))

    ip_max_len = max(ip_lengths)
    op_max_len = max(op_lengths)

    padded_input_ids = []
    padded_output_ids = []

    for ids in input_ids:
        padded = ids + [pad_id] * (ip_max_len - len(ids))
        padded_input_ids.append(padded)

    for ids in output_ids:
        padded = ids + [pad_id] * (op_max_len - len(ids))
        padded_output_ids.append(padded)

    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_output_ids = torch.tensor(padded_output_ids, dtype=torch.long)

    ip_lengths = torch.tensor(ip_lengths, dtype=torch.long)
    op_lengths = torch.tensor(op_lengths, dtype=torch.long)

    return padded_input_ids, padded_output_ids, ip_lengths, op_lengths