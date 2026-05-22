import torch
from torch import nn 


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
        batch_first=True,
        total_length=input_ids.size(1)
    )


    return outputs, hidden, cell


class BahdanauAttention(nn.Module):

  def __init__(self, hidden_dim, atten_dim):
    super().__init__()

    self.W_decoder = nn.Linear(hidden_dim, atten_dim)
    self.W_encoder = nn.Linear(hidden_dim, atten_dim)
    self.V = nn.Linear(atten_dim, 1, bias=False)

  def forward(self, decoder_hidden, encoder_outputs, mask=None):

    src_len = encoder_outputs.shape[1]

    decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

    energy = torch.tanh(
        self.W_decoder(decoder_hidden) + self.W_encoder(encoder_outputs)
    )

    scores = self.V(energy).squeeze(2)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    attention_weights = torch.softmax(scores, dim=1)

    context = torch.bmm(
        attention_weights.unsqueeze(1),
        encoder_outputs
    )

    return context, attention_weights


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        pad_idx=0,
        dropout=0.3,
        attn_dim=256
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.attention = BahdanauAttention(
            hidden_dim=hidden_dim,
            atten_dim=attn_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            embed_dim + hidden_dim + hidden_dim,
            vocab_size
        )

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):

        input_token = input_token.unsqueeze(1)


        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)


        decoder_hidden = hidden[-1]


        context, attention_weights = self.attention(
            decoder_hidden,
            encoder_outputs,
            mask
        )

        lstm_input = torch.cat((embedded, context), dim=2)


        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction_input = torch.cat(
            (
                output.squeeze(1),
                context.squeeze(1),
                embedded.squeeze(1)
            ),
            dim=1
        )


        logits = self.fc_out(prediction_input)


        return logits, hidden, cell, attention_weights
    
    
class Model(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_mask(self, input_ids):
        return input_ids != self.pad_idx


    def forward(self, input_ids, output_ids, input_lengths=None):

        encoder_outputs, hidden, cell = self.encoder(input_ids, input_lengths)

        mask = self.create_mask(input_ids)


        batch_size = output_ids.shape[0]
        tgt_len = output_ids.shape[1]
        vocab_size = self.decoder.vocab_size

        predictions = torch.zeros(
            batch_size,
            tgt_len - 1,
            vocab_size,
            device=output_ids.device
        )

        attentions = torch.zeros(
            batch_size,
            tgt_len - 1,
            input_ids.shape[1],
            device=output_ids.device
        )

        for t in range(tgt_len - 1):
            decoder_input = output_ids[:, t]

            logits, hidden, cell, attention_weights = self.decoder(
                decoder_input,
                hidden,
                cell,
                encoder_outputs,
                mask
            )

            predictions[:, t, :] = logits
            attentions[:, t, :] = attention_weights

        return predictions, attentions
