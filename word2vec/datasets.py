import torch
from torch.utils.data import Dataset


# Skip-gram Dataset 

class SkipgramDataset(Dataset):

    def __init__(self, vocab, id_to_word, conv_data, window_size):
        self.vocab = vocab
        self.id_to_word = id_to_word
        self.conv_data = conv_data
        self.window_size = window_size

        self.skipgram_pairs = []

        for sent in self.conv_data:
            for sg_idx in range(len(sent)):
                center_word = sent[sg_idx]

                for i in range(sg_idx-self.window_size,sg_idx+self.window_size+1):
                    if(i>=0 and i<len(sent) and i!=sg_idx):

                        self.skipgram_pairs.append(([center_word,sent[i]]))

    def __len__(self):
        return len(self.skipgram_pairs)

    def __getitem__(self,index):

        return self.skipgram_pairs[index]
    

# Cbow Dataset     

class CbowDataset(Dataset):

    def __init__(self, vocab, id_to_word, conv_data, window_size):
        self.vocab = vocab
        self.id_to_word = id_to_word
        self.conv_data = conv_data
        self.window_size = window_size

        self.cbow_pairs = []

        for sent in self.conv_data:

          for cw_idx in range(len(sent)):

            cbow_sent = []
            center_word = sent[cw_idx]
            cw_tup = ()

            for i in range(cw_idx-self.window_size,cw_idx+self.window_size+1):
                if(i>=0 and i<len(sent) and i!=cw_idx):
                    cbow_sent.append(sent[i])

            if(len(cbow_sent)>0):
              cw_tup+=(cbow_sent,center_word)


              self.cbow_pairs.append(cw_tup)


    def __len__(self):
        return len(self.cbow_pairs)

    def __getitem__(self,index):

        return self.cbow_pairs[index]