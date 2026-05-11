import torch 
import torch.nn as nn 


class Encoder(nn.Module):

  def __init__(self,embed_dim, hidden_dim, vocab_eng, pad_idx=0):
    super().__init__()

    self.embedding = nn.Embedding(
        len(vocab_eng),
        embed_dim,
        padding_idx=pad_idx
    )


    self.lstm = nn.LSTM(
        embed_dim,
        hidden_dim,
        batch_first=True  
    )


  def forward(self, input_ids, input_lengths):
    embedded = self.embedding(input_ids)
    outputs, (hidden,cell) = self.lstm(embedded)

    return outputs, hidden, cell 


class Decoder(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_fr, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            len(vocab_fr),
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            len(vocab_fr)
        )

    def forward(self, input_ids, hidden, cell):
        embedded = self.embedding(input_ids)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        predictions = self.fc(outputs)

        return predictions, hidden, cell
    

class Model(nn.Module):

  def __init__(self,encoder,decoder):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self,input_ids, output_ids, input_lengths=None):
    encoder_outputs, hidden, cell = self.encoder(input_ids, input_lengths)
    decoder_input = output_ids[:, :-1]

    predictions, hidden, cell = self.decoder(
            decoder_input,
            hidden,
            cell
        )

    return predictions     