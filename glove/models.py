from torch import nn


class GloveModel(nn.Module):

  def __init__(self,vocab_size,embedding_dim):
    super().__init__()
    self.word_i_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.word_j_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.bias_i = nn.Embedding(vocab_size,1)
    self.bias_j = nn.Embedding(vocab_size,1)

  def forward(self,word_i,word_j):

    w_i = self.word_i_embedding(word_i)
    w_j = self.word_j_embedding(word_j)

    b_i = self.bias_i(word_i)
    b_j = self.bias_j(word_j)

    dot_product = (w_i*w_j).sum(dim=1)

    score = dot_product + b_i.squeeze(1) + b_j.squeeze(1)

    return score