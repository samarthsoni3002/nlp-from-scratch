import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, embed_dim, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.hidden_size = hidden_size
        self.output_size = output_size

        # reset gate
        self.wr = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.ur = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.br = nn.Parameter(torch.zeros(hidden_size))

        # update gate
        self.wz = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.uz = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bz = nn.Parameter(torch.zeros(hidden_size))

        # candidate hidden state
        self.wh = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.uh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        # hidden to output
        self.wy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.by = nn.Parameter(torch.zeros(output_size))

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

            # reset gate
            r_t = torch.sigmoid(
                x_t @ self.wr + h_t @ self.ur + self.br
            )

            # candidate hidden state
            h_tilde_t = torch.tanh(
                x_t @ self.wh + (r_t * h_t) @ self.uh + self.bh
            )

            # update gate
            z_t = torch.sigmoid(
                x_t @ self.wz + h_t @ self.uz + self.bz
            )

            # new hidden state
            h_t = (1 - z_t) * h_t + z_t * h_tilde_t

            hidden_states.append(h_t.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)

        lengths = lengths.to(x.device)
        last_token_indices = lengths - 1

        batch_indices = torch.arange(batch_size, device=x.device)

        final_hidden = hidden_states[batch_indices, last_token_indices]

        logits = final_hidden @ self.wy + self.by

        return logits, hidden_states