import torch
from torch.nn import functional as F 

def weight_function(pair_count):

  alpha = 0.75
  x_max = 100

  if(pair_count < x_max):
    return (pair_count/x_max)**alpha
  else:
    return 1


def loss_fxn(score, function, pair_counts):
    weights = torch.tensor([function(x.item()) for x in pair_counts], dtype=torch.float32)
    log_counts = torch.log(pair_counts)
    loss = (score - log_counts) ** 2
    return (weights * loss).mean()


def get_word_vector(word, model, word_to_id):
    if word not in word_to_id:
        raise ValueError(f"'{word}' not found in vocabulary.")

    embeddings = (
        model.word_i_embedding.weight.detach()
        + model.word_j_embedding.weight.detach()
    )

    word_id = word_to_id[word]
    return embeddings[word_id]


def most_similar(word, model, word_to_id, id_to_word, top_k=5, vocab_size=None):
    
    if word not in word_to_id:
        raise ValueError(f"'{word}' not found in vocabulary.")

    embeddings = (
        model.word_i_embedding.weight.detach()
        + model.word_j_embedding.weight.detach()
    )

    if vocab_size is not None:
        embeddings = embeddings[:vocab_size]

    word_id = word_to_id[word]
    query_vec = embeddings[word_id].unsqueeze(0)

    similarities = F.cosine_similarity(query_vec, embeddings, dim=1)
    similarities[word_id] = -1.0

    top_k = min(top_k, embeddings.size(0) - 1)
    top_ids = torch.topk(similarities, k=top_k).indices.tolist()

    return [(id_to_word[idx], similarities[idx].item()) for idx in top_ids]
