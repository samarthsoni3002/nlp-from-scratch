import torch 
from collections import Counter 
from torch.utils.data import Dataset

class GloveDataset(Dataset):
    def __init__(self, vocab, conv_data, window_size):
        self.vocab = vocab

        counts = Counter()
        for sent in conv_data:
            for sg_idx in range(len(sent)):
                center_word = sent[sg_idx]
                for i in range(sg_idx - window_size, sg_idx + window_size + 1):
                    if 0 <= i < len(sent) and i != sg_idx:

                        distance = abs(i - sg_idx)

                        counts[(center_word, sent[i])] += (1.0/distance)


        self.glove_list = list(counts.items())

    def __len__(self):
        return len(self.glove_list)

    def __getitem__(self, index):

        (word_i, word_j), x_ij = self.glove_list[index]

        return torch.tensor(word_i,dtype=torch.long), torch.tensor(word_j,dtype=torch.long), torch.tensor(x_ij, dtype=torch.float)