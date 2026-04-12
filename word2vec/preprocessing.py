import re 
import unicodedata
from collections import Counter

def processing_batch(batch):

    english_word_pattern = re.compile(r"[a-z]+")
    all_tokens = []

    for text in batch["text"]:
        if not text or not text.strip():
            all_tokens.append([])
            continue

        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = text.lower()

        tokens = english_word_pattern.findall(text)
        all_tokens.append(tokens)


    return {"tokens": all_tokens}


def build_vocab(tokenized_train,min_freq):

  counter = Counter()

  for sent in tokenized_train["tokens"]:
    counter.update(sent)

  vocab = {}
  id_to_word = {}

  idx = 0

  for word,freq in counter.items():
    if freq>=min_freq:
      vocab[word] = idx
      id_to_word[idx] = word
      idx+=1

  return vocab, id_to_word, counter


def vocab_map(tokenized_train, vocab):

  all_sent_converted = []

  for sent in tokenized_train["tokens"]:
    sent_converted = []
    for word in sent:
      if word in vocab:
        sent_converted.append(vocab[word])

    if sent_converted:
      all_sent_converted.append(sent_converted)


  return all_sent_converted

