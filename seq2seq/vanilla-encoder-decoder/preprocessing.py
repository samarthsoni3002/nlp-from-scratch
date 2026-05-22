import re 
import unicodedata
from collections import Counter


def basic_preprocessing(data):
    token_pattern = re.compile(r"[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)
    all_tokens = []

    for text in data:
        if not text or not text.strip():
            all_tokens.append([])
            continue

        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        tokens = token_pattern.findall(text)

        tokens = ["<NUM>" if token.isdigit() else token for token in tokens]

        all_tokens.append(tokens)

    return {"tokens": all_tokens}
  
  
def filter_pairs(src_tokens, tgt_tokens, max_src_len=12, max_tgt_len=14):
    filtered_src = []
    filtered_tgt = []

    for src, tgt in zip(src_tokens, tgt_tokens):
        if len(src) == 0 or len(tgt) == 0:
            continue

        if len(src) > max_src_len or len(tgt) > max_tgt_len:
            continue

        ratio = max(len(src) / len(tgt), len(tgt) / len(src))
        if ratio > 2.5:
            continue

        filtered_src.append(src)
        filtered_tgt.append(tgt)

    return filtered_src, filtered_tgt


def build_vocab(tokenized_sentences, special_token, min_freq=2, max_size=None):
    counter = Counter()

    for sent in tokenized_sentences:
        counter.update(sent)

    vocab = {tok: idx for idx, tok in enumerate(special_token)}

    words = counter.most_common()

    if max_size is not None:
        words = words[:max_size]

    for word, freq in words:
        if freq < min_freq:
            continue

        if word not in vocab:
            vocab[word] = len(vocab)

    id_to_word = {idx: word for word, idx in vocab.items()}

    return vocab, id_to_word, counter


def vocab_map(tokens, vocab):
    unk_id = vocab["<UNK>"]

    converted = [
        [vocab.get(word, unk_id) for word in sent]
        for sent in tokens
    ]

    return converted