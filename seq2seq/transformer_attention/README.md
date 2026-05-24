# Transformer Attention from Scratch

This folder contains a learning-oriented PyTorch implementation of Transformer-style encoder-decoder attention for English-to-French translation.

The goal is to understand the core mechanics from **Attention Is All You Need** by implementing the important pieces directly: positional encodings, scaled dot-product attention, multi-head attention, source masks, target causal masks, cross-attention, residual connections, layer normalization, and feed-forward blocks.

---

## What This Implements

This implementation includes:

- sinusoidal positional encoding
- scaled dot-product attention
- multi-head attention
- encoder self-attention
- decoder masked self-attention
- encoder-decoder cross-attention
- source padding mask
- target padding + causal mask
- residual connections
- layer normalization
- position-wise feed-forward layers
- cross-entropy loss with `<PAD>` ignored
- gradient clipping

---

## Folder Structure

```bash
seq2seq/transformer_attention/
├── README.md
├── models.py
├── trainer.py
└── train.py
```

Shared seq2seq utilities live in:

```bash
seq2seq/
├── eng-fra.txt
├── datasets_classes.py
├── preprocessing.py
├── utils.py
└── inference.py
```

---

## Core Idea

Unlike RNN encoder-decoder models, the Transformer does not process tokens one step at a time with recurrence.

Instead, it uses attention to let each token build a representation by looking at other tokens.

```text
token embeddings + positional encodings
      |
      v
self-attention / cross-attention
      |
      v
feed-forward transformation
      |
      v
vocabulary logits
```

---

## Main Components

### 1. Positional Encoding

Since attention has no recurrence, the model needs position information.

The implementation adds sinusoidal positional encodings to token embeddings before attention.

### 2. Scaled Dot-Product Attention

The attention module computes:

```text
scores = QK^T / sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights V
```

Masks are applied before softmax so padded or future positions do not receive attention.

### 3. Multi-Head Attention

The model projects inputs into query, key, and value tensors, splits them across multiple heads, applies scaled dot-product attention per head, concatenates the results, and projects back to `d_model`.

### 4. Encoder Block

The encoder currently uses:

```text
embedding + positional encoding
      |
      v
multi-head self-attention
      |
      v
residual + layer norm
      |
      v
feed-forward network
      |
      v
residual + layer norm
```

### 5. Decoder Block

The decoder currently uses:

```text
target embedding + positional encoding
      |
      v
masked multi-head self-attention
      |
      v
residual + layer norm
      |
      v
cross-attention over encoder outputs
      |
      v
residual + layer norm
      |
      v
feed-forward network
      |
      v
residual + layer norm
```

### 6. Transformer Wrapper

The `Transformer` class creates the source padding mask, runs the encoder, runs the decoder, and projects decoder outputs to target vocabulary logits.

---

## File Overview

### `models.py`

Contains:

- `positional_encoding`
- `scaled_dot_product_attention`
- `MultiHeadAttention`
- `Encoder`
- `Decoder`
- `Transformer`

Important implementation details:

- attention tensors are shaped as `[batch_size, num_heads, seq_len, d_k]`
- source mask blocks attention to `<PAD>` tokens
- target mask combines padding mask and causal lower-triangular mask
- decoder has both self-attention and cross-attention

### `trainer.py`

Contains the training loop.

The loop:

1. splits target sequence into decoder input and labels
2. runs the Transformer
3. computes cross-entropy loss
4. ignores `<PAD>` tokens
5. clips gradients
6. updates parameters
7. prints average loss per epoch

### `train.py`

Main runnable script.

It loads data, preprocesses text, builds vocabularies, creates dataloaders, initializes the Transformer-style model, trains it, and runs a small translation example.

---

## How to Run

From the repository root:

```bash
python -m seq2seq.transformer_attention.train.py
```

---

## Current Hyperparameters

The script currently uses:

```python
d_model = 128
hidden_dim = 512
num_heads = 4
batch_size = 32
learning_rate = 1e-4
max_len = 100
num_epochs = 2
```

These are learning-oriented defaults.

---

## Transformer vs Recurrent Attention Models

| Model | Sequence Processing | Attention Role |
|---|---|---|
| Bahdanau / Luong | recurrent decoder generates one step at a time | attention helps decoder choose encoder states |
| Transformer | attention is the main sequence operation | self-attention and cross-attention replace recurrence |

---


## Future Work

- [ ] Add Transformer-specific greedy decoding
- [ ] Add checkpoint saving/loading
- [ ] Add validation and BLEU evaluation
- [ ] Add dropout
- [ ] Stack multiple encoder and decoder layers
- [ ] Add attention visualization
- [ ] Add CLI/config support
- [ ] Compare against vanilla, Bahdanau, and Luong models
