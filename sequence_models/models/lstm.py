import torch 
import torch.nn as nn 


class LSTM(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, embed_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim,padding_idx = pad_idx)
        self.hidden_size = hidden_size
        self.output_size = output_size

        # input gate
        self.wi = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.ui = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        # forget gate
        self.wf = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.uf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        # candidate cell
        self.wc = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.uc = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bc = nn.Parameter(torch.zeros(hidden_size))

        # output gate
        self.wo = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.uo = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        # hidden to output
        self.wy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, lengths, h0=None, c0=None):

        emb = self.embedding(x)
        batch_size, sq_len, _ = emb.shape

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0

        if c0 is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            c_t = c0

        hidden_states = []


        for t in range(sq_len):

          x_t = emb[:, t, :]

          f_t = torch.sigmoid(x_t @ self.wf + h_t @ self.uf + self.bf)
          i_t = torch.sigmoid(x_t @ self.wi + h_t @ self.ui + self.bi)
          c_tilde_t = torch.tanh(x_t @ self.wc + h_t @ self.uc + self.bc)
          c_t = (f_t * c_t) + (i_t * c_tilde_t)
          o_t = torch.sigmoid(x_t @ self.wo + h_t @ self.uo + self.bo)
          h_t = o_t * torch.tanh(c_t)

          hidden_states.append(h_t.unsqueeze(1))


        hidden_states = torch.cat(hidden_states, dim=1)
        
        lengths = lengths.to(x.device)
        last_token_indices = lengths - 1


        batch_indices = torch.arange(batch_size, device=x.device)

        final_hidden = hidden_states[batch_indices, last_token_indices]

        logits = final_hidden @ self.wy + self.by

        return logits, hidden_states