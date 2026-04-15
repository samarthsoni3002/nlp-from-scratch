import torch 
from torch.nn import functional as F


def build_noise_distribution(counter, vocab):
    freqs = [0] * len(vocab)

    for word, idx in vocab.items():
        freqs[idx] = counter[word]

    freqs = torch.tensor(freqs, dtype=torch.float32)
    probs = freqs.pow(0.75)
    probs = probs / probs.sum()

    return probs


def sample_negative_words(pos_ids, num_negatives, noise_dist):
    neg_ids = []

    for pos_id in pos_ids:
        curr_negs = []

        while len(curr_negs) < num_negatives:
            sampled = torch.multinomial(noise_dist, num_samples=1, replacement=True).item()

            if sampled != pos_id.item():
                curr_negs.append(sampled)

        neg_ids.append(curr_negs)

    return torch.tensor(neg_ids, dtype=torch.long)


def negative_sampling_loss(pos_scores, neg_scores):

  pos_term = F.logsigmoid(pos_scores)
  neg_term = F.logsigmoid(-neg_scores)

  return -(pos_term + neg_term.sum(dim=1)).mean()