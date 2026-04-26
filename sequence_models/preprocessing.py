import unicodedata 
import re 
from utils import expand_contractions
from collections import Counter

def processing_batch(batch,contractions,stop_words):

    english_word_pattern = re.compile(r"[a-z]+")
    all_tokens = []

    for text in batch["text"]:
        if not text or not text.strip():
            all_tokens.append([])
            continue

        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = text.lower()
        text = expand_contractions(text,contractions)
        tokens = english_word_pattern.findall(text)

        tokens = [token for token in tokens if token not in stop_words and len(token)>1]

        all_tokens.append(tokens)


    return {"tokens": all_tokens}


def build_vocab(tokenized_train, min_freq,pad_token,unk_token):
    counter = Counter()

    for sent in tokenized_train["tokens"]:
        counter.update(sent)

    vocab = {
        pad_token: 0,
        unk_token: 1,
    }

    id_to_word = {
        0: pad_token,
        1: unk_token,
    }

    idx = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            id_to_word[idx] = word
            idx += 1

    return vocab, id_to_word, counter


def vocab_map(tokenized_data, vocab,unk_token):
    all_input_ids = []
    all_labels = []

    unk_id = vocab[unk_token]

    for tokens, label in zip(tokenized_data["tokens"], tokenized_data["label"]):

        input_ids = [
            vocab.get(token, unk_id)
            for token in tokens
        ]

        if len(input_ids) == 0:
            input_ids = [unk_id]

        all_input_ids.append(input_ids)
        all_labels.append(label)

    return all_input_ids, all_labels