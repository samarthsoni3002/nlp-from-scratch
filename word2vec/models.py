import torch
from torch import nn


# Skip-gram Model

class SkipgramModel(nn.Module):

  def __init__(self,vocab_size,embedding_dim):
    super().__init__()
    self.in_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.out_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

  def forward(self,center_ids):

    center_emb = self.in_embedding(center_ids)
    logits = center_emb @ self.out_embedding.weight.T

    return logits 


# Skip-gram Negative Sampling 

class SkipgramModelNegativeSampling(nn.Module):

  def __init__(self,vocab_size,embedding_dim):
    super().__init__()
    self.in_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.out_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

  def forward(self,center_ids,pos_ids,neg_ids):

    center_emb = self.in_embedding(center_ids)
    pos_emb = self.out_embedding(pos_ids)
    neg_emb = self.out_embedding(neg_ids)

    pos_scores = torch.sum(center_emb*pos_emb, dim=1)
    neg_scores = torch.sum(center_emb.unsqueeze(1)*neg_emb, dim=2)

    return pos_scores, neg_scores


# Cbow Model 

class CbowModel(nn.Module):

  def __init__(self,vocab_size,embedding_dim,pad_id):
    super().__init__()
    self.pad_id = pad_id
    self.in_embedding = nn.Embedding(num_embeddings=vocab_size+1,embedding_dim=embedding_dim,padding_idx=pad_id)
    self.out_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

  def forward(self,context_ids):

    context_emb = self.in_embedding(context_ids)

    mask = (context_ids != self.pad_id).unsqueeze(-1)
    masked_emb = context_emb * mask

    summed = masked_emb.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    avg_emb = summed / counts

    logits = avg_emb @ self.out_embedding.weight.T

    return logits
  
  

# CBOW Negative Sampling Model 

class CbowModel(nn.Module):

  def __init__(self,vocab_size,embedding_dim,pad_id):
    super().__init__()
    self.pad_id = pad_id
    self.in_embedding = nn.Embedding(num_embeddings=vocab_size+1,embedding_dim=embedding_dim,padding_idx=pad_id)
    self.out_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

  def forward(self,context_ids,target_ids,negative_ids):

    context_emb = self.in_embedding(context_ids)

    mask = (context_ids != self.pad_id).unsqueeze(-1)
    masked_emb = context_emb * mask

    summed = masked_emb.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    avg_emb = summed / counts

    pos_emb = self.out_embedding(target_ids)
    neg_emb = self.out_embedding(negative_ids)

    pos_score = torch.sum(avg_emb*pos_emb, dim=1)
    neg_score = torch.sum(avg_emb.unsqueeze(1)*neg_emb, dim=2)

    return pos_score, neg_score
