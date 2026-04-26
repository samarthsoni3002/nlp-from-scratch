import torch 
import torch.nn as nn 

class RNNScratch(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_hiddens, output_size, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.input_size = embed_dim
        self.hidden_size = num_hiddens
        self.output_size = output_size

        self.wx = nn.Parameter(torch.randn(self.input_size, self.hidden_size) * 0.01)
        self.wh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(self.hidden_size))

        self.wy = nn.Parameter(torch.randn(self.hidden_size, self.output_size) * 0.01)
        self.by = nn.Parameter(torch.zeros(self.output_size))

    def forward(self, x, lengths, h0=None):

        emb = self.embedding(x)

        batch_size, seq_len, _ = emb.shape

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        hidden_states = []

        for t in range(seq_len):
            
            x_t = emb[:, t, :]

            h_t = torch.tanh(
                x_t @ self.wx + h_t @ self.wh + self.bh
            )

            hidden_states.append(h_t.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)

      
        lengths = lengths.to(x.device)
        last_token_indices = lengths - 1

        batch_indices = torch.arange(batch_size, device=x.device)

        final_hidden = hidden_states[batch_indices, last_token_indices]

        logits = final_hidden @ self.wy + self.by

        return logits, hidden_states