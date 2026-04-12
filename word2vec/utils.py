def cbow_collate(batch):
    contexts = []
    targets = []
    lengths = []

    for context_ids, target_id in batch:
        contexts.append(context_ids)
        targets.append(target_id)
        lengths.append(len(context_ids))

    max_len = max(lengths)
    PAD_ID = len(vocab)

    padded_contexts = []
    for context_ids in contexts:
        padded = context_ids + [PAD_ID] * (max_len - len(context_ids))
        padded_contexts.append(padded)

    padded_contexts = torch.tensor(padded_contexts, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_contexts, targets, lengths