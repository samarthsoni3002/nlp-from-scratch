import torch 
import torch.nn as nn 


class Encoder(nn.Module):

  def __init__(self,embed_dim, hidden_dim, vocab_size, pad_idx=0):
    super().__init__()

    self.embedding = nn.Embedding(
        vocab_size,
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
      packed = nn.utils.rnn.pack_padded_sequence(
      embedded,
      input_lengths.cpu(),
      batch_first=True,
      enforce_sorted=False
        )
    
      packed_outputs, (hidden, cell) = self.lstm(packed)

      outputs, _ = nn.utils.rnn.pad_packed_sequence(
          packed_outputs,
          batch_first=True
      )

      return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        pad_idx=0,
        dropout=0.3
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input_ids, hidden, cell):

        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        logits = self.fc_out(outputs)

        return logits, hidden, cell
    

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, output_ids, input_lengths=None):
        encoder_outputs, hidden, cell = self.encoder(input_ids, input_lengths)

        decoder_input = output_ids[:, :-1]

        predictions, hidden, cell = self.decoder(
            decoder_input,
            hidden,
            cell
        )

        return predictions