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