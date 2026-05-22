# Vanilla Encoder-Decoder from Scratch

This folder contains a from-scratch PyTorch implementation of a basic **sequence-to-sequence encoder-decoder model** for machine translation.

The goal is to understand the original encoder-decoder idea before adding attention.

Current task:

```text
English sentence -> French sentence
```

This is the baseline model for the `seq2seq/` section of the repository.

---

## What This Implements

This implementation contains:

- tab-separated English-French data loading
- basic Unicode-aware preprocessing
- source and target vocabulary creation
- special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- custom translation dataset
- custom collate function with dynamic padding
- LSTM encoder
- LSTM decoder
- teacher-forced decoder input shifting
- cross-entropy training objective
- padding ignored in the loss
- gradient clipping
- greedy decoding inference

---

## Folder Structure

```bash
seq2seq/vanilla-encoder-decoder/
├── README.md
├── eng-fra.txt
├── datasets_classes.py
├── preprocessing.py
├── models.py
├── trainer.py
├── utils.py
└── train.py
```

---

## File Overview

### `eng-fra.txt`

Tab-separated translation pairs.

Each row is expected to look like:

```text
english sentence<TAB>french sentence
```

---

### `utils.py`

Contains helper functions for:

- reading the translation file
- splitting English and French sentence pairs
- greedy decoding through `translate_sentence`

The inference helper encodes the source sentence, starts decoding from `<SOS>`, repeatedly predicts the next token, and stops when `<EOS>` is generated or `max_len` is reached.

---

### `preprocessing.py`

Handles:

- Unicode normalization
- lowercasing
- tokenization of words and punctuation
- vocabulary creation for English and French
- mapping tokens to integer ids

Special tokens used:

```python
["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
```

Current sequence formatting:

```text
English input: token_1 token_2 ... token_n <EOS>
French output: <SOS> token_1 token_2 ... token_n <EOS>
```

---

### `datasets_classes.py`

Contains:

- `TranslationDataset`
- `translation_collate_fn`

The collate function dynamically pads each batch and returns:

```python
padded_input_ids
padded_output_ids
input_lengths
output_lengths
```

The lengths are useful for future packed-sequence support and attention masks.

---

### `models.py`

Contains three classes:

- `Encoder`
- `Decoder`
- `Model`

#### Encoder

The encoder uses:

```text
source token ids -> embedding -> LSTM -> final hidden/cell state
```

It returns:

```python
encoder_outputs, hidden, cell
```

#### Decoder

The decoder uses:

```text
target token ids -> embedding -> LSTM -> Linear -> vocabulary logits
```

It receives the encoder's final hidden and cell states as its initial state.

#### Full Model

During training, the decoder input is shifted right:

```python
decoder_input = output_ids[:, :-1]
decoder_target = output_ids[:, 1:]
```

So the model learns:

```text
given <SOS>, predict first target token
given first target token, predict second target token
...
given previous target token, predict <EOS>
```

---

### `trainer.py`

Contains the training logic for one epoch.

The training loop:

1. sends batches through the model
2. creates shifted decoder targets
3. computes cross-entropy loss
4. ignores `<PAD>` tokens in the loss
5. backpropagates gradients
6. clips gradients
7. updates model parameters

---

### `train.py`

Main training script.

It currently:

1. loads `eng-fra.txt`
2. preprocesses English and French text
3. builds source and target vocabularies
4. converts tokens to ids
5. creates dataset and dataloader
6. initializes encoder and decoder
7. trains the model
8. runs a small translation example

---

## Architecture

```text
English sentence
      |
      v
English token ids
      |
      v
English embeddings
      |
      v
LSTM Encoder
      |
      v
final hidden state + final cell state
      |
      v
LSTM Decoder
      |
      v
French vocabulary logits
      |
      v
Predicted French tokens
```

---

## Training Objective

The model is trained using cross-entropy loss over the French vocabulary.

Padding tokens are ignored:

```python
nn.CrossEntropyLoss(ignore_index=0)
```

Since `<PAD>` has id `0`, padded positions do not contribute to the loss.

---

## How to Run

From the repository root:

```bash
python seq2seq/vanilla-encoder-decoder/train.py
```

The script will train the model and then run a small greedy-decoding translation example.

---

## Current Hyperparameters

The current script uses small values for fast learning experiments:

```python
num_epochs = 10
batch_size = 32
encoder_embedding_dim = 8
decoder_embedding_dim = 8
hidden_dim = 16
learning_rate = 0.01
```

These are intentionally small and should be increased for better translation quality.

---

## Current Inference Method

Inference is greedy decoding.

At each decoder step:

1. pass the previous token into the decoder
2. compute logits over the French vocabulary
3. choose the token with highest probability
4. feed that token back into the decoder
5. stop on `<EOS>` or after `max_len`

This is simple and useful for understanding, but it is not as strong as beam search.

---

## What This Model Does Not Include Yet

This is a vanilla encoder-decoder baseline.

It does **not** yet include:

- attention
- bidirectional encoder
- packed sequences
- beam search
- BLEU score
- checkpoint saving/loading
- teacher forcing ratio control
- validation split
- proper experiment configuration
- CLI arguments

---

## Why Start With Vanilla Encoder-Decoder?

Before implementing attention, it is important to understand the original bottleneck.

In a vanilla encoder-decoder model, the entire source sentence is compressed into the final hidden and cell states of the encoder. The decoder then has to generate the full output sentence from that compressed representation.

This works for short sentences, but it becomes weak for longer sentences because the decoder does not directly look back at every encoder state.

That weakness motivates attention.

---

## Future Work / To Do

Planned improvements:

- [ ] Add validation and test splits
- [ ] Add checkpoint saving and loading
- [ ] Add BLEU score evaluation
- [ ] Add teacher forcing ratio control
- [ ] Add packed sequence support
- [ ] Add bidirectional encoder
- [ ] Add Bahdanau attention
- [ ] Add Luong attention
- [ ] Add beam search decoding
- [ ] Add CLI-based experiment configuration
- [ ] Add training and validation loss plots
- [ ] Improve translation examples in inference

---


## Final Note

This folder is meant to be the simplest clean baseline for seq2seq learning.

