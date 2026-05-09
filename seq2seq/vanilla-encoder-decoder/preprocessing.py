import re 
import unicodedata
from collections import Counter


def basic_preprocessing(data):

    token_pattern = re.compile(r"[^\W\d_]+|[^\w\s]", re.UNICODE)
    all_tokens = []

    for text in data:
        if not text or not text.strip():
            all_tokens.append([])
            continue

        text = unicodedata.normalize("NFC", text)
        text = text.lower()

        tokens = token_pattern.findall(text)
        all_tokens.append(tokens)


    return {"tokens": all_tokens}


def build_vocab(X_tokens,y_tokens,special_tokens):

  

  counter_eng = Counter()
  counter_fr = Counter()

  for sent in X_tokens:
    counter_eng.update(sent)

  for sent in y_tokens:
    counter_fr.update(sent)


  vocab_eng = {token: idx for idx, token in enumerate(special_tokens)}
  vocab_fr = {token: idx for idx, token in enumerate(special_tokens)}
  id_to_word_eng = {}
  id_to_word_fr = {}


  idx = len(special_tokens)

  for word,_ in counter_eng.items():

    if word not in vocab_eng:
      vocab_eng[word] = idx
      id_to_word_eng[idx] = word
      idx+=1


  idx = len(special_tokens)

  for word,freq in counter_fr.items():

    if word not in vocab_fr:
      vocab_fr[word] = idx
      id_to_word_fr[idx] = word
      idx+=1

  return vocab_eng, vocab_fr, id_to_word_eng, id_to_word_fr, counter_eng, counter_fr


def vocab_map(tokens, vocab):
    unk_id = vocab["<UNK>"]

    converted = [
        [vocab.get(word, unk_id) for word in sent]
        for sent in tokens
    ]

    return converted