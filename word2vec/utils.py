import torch 
from torch.nn import functional as F

from models import (
    SkipgramModel,
    SkipgramModelNegativeSampling,
    CbowModel,
    CbowModelNegativeSampling,
)


def build_noise_distribution(counter, vocab):
    freqs = [0] * len(vocab)

    for word, idx in vocab.items():
        freqs[idx] = counter[word]

    freqs = torch.tensor(freqs, dtype=torch.float32)
    probs = freqs.pow(0.75)
    probs = probs / probs.sum()

    return probs


def sample_negative_words(pos_ids, num_neg_samples, noise_dist):
    if isinstance(pos_ids, int):
        pos_ids = [pos_ids]

    neg_samples = []

    for pos_id in pos_ids:
        if torch.is_tensor(pos_id):
            pos_id = pos_id.item()

        curr_negs = []

        while len(curr_negs) < num_neg_samples:
            sampled = torch.multinomial(noise_dist, num_samples=1, replacement=True).item()

            if sampled != pos_id:
                curr_negs.append(sampled)

        neg_samples.append(curr_negs)

    return neg_samples


def negative_sampling_loss(pos_scores, neg_scores):

  pos_term = F.logsigmoid(pos_scores)
  neg_term = F.logsigmoid(-neg_scores)

  return -(pos_term + neg_term.sum(dim=1)).mean()


def get_word_vector(word, model, word_to_id, vocab_size=None):
    if word not in word_to_id:
        raise ValueError(f"'{word}' not found in vocabulary.")

    embeddings = model.in_embedding.weight.detach()

    if vocab_size is not None:
        embeddings = embeddings[:vocab_size]

    word_id = word_to_id[word]
    return embeddings[word_id]


def most_similar(word, model, word_to_id, id_to_word, top_k=5, vocab_size=None):
    
    if word not in word_to_id:
        raise ValueError(f"'{word}' not found in vocabulary.")

    embeddings = model.in_embedding.weight.detach()

    if vocab_size is not None:
        embeddings = embeddings[:vocab_size]

    word_id = word_to_id[word]
    query_vec = embeddings[word_id].unsqueeze(0)

    similarities = F.cosine_similarity(query_vec, embeddings, dim=1)
    similarities[word_id] = -1.0

    top_k = min(top_k, embeddings.size(0) - 1)
    top_ids = torch.topk(similarities, k=top_k).indices.tolist()

    return [(id_to_word[idx], similarities[idx].item()) for idx in top_ids]


def load_trained_model(model_name, checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab_size = checkpoint["vocab_size"]
    embedding_dim = checkpoint["embedding_dim"]

    if model_name == "skip-gram":
        model = SkipgramModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

    elif model_name == "skip-gram-ns":
        model = SkipgramModelNegativeSampling(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

    elif model_name == "cbow":
        model = CbowModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=checkpoint["pad_id"],
        )

    elif model_name == "cbow-ns":
        model = CbowModelNegativeSampling(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=checkpoint["pad_id"],
        )

    else:
        raise ValueError(
            "Invalid model_name. Use one of: "
            "'skip-gram', 'skip-gram-negative-sampling', "
            "'cbow', 'cbow-negative-sampling'"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    word_to_id = checkpoint["word_to_id"]
    id_to_word = checkpoint["id_to_word"]

    return model, word_to_id, id_to_word, checkpoint