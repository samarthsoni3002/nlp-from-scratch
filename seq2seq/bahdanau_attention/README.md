# Bahdanau Attention from Scratch

This folder contains a from-scratch PyTorch implementation of an LSTM encoder-decoder model with **Bahdanau additive attention** for English-to-French translation.

Bahdanau attention improves the vanilla encoder-decoder model by allowing the decoder to look back at all encoder hidden states instead of relying only on the final encoder state.

---

## What This Implements

This implementation includes:

- LSTM encoder with packed sequences
- additive attention module
- LSTM decoder with attention context vector
- padding mask for source tokens
- teacher-forced decoding during training
- attention weight collection for each target step
- cross-entropy loss with `<PAD>` ignored
- gradient clipping
- greedy translation example after training

---

## Folder Structure

```bash
seq2seq/bahdanau_attention/
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

In the vanilla encoder-decoder model, the decoder receives only the final encoder hidden/cell states.

Bahdanau attention changes this by computing a context vector at every decoding step:

```text
decoder hidden state + all encoder outputs -> attention scores -> context vector
```

The decoder then uses this context vector while predicting the next target token.

---

## Architecture

```text
source tokens
   |
   v
LSTM encoder
   |
   v
encoder outputs for every source position
   |
   v
Bahdanau attention <--- decoder hidden state
   |
   v
context vector
   |
   v
LSTM decoder + output layer
   |
   v
target vocabulary logits
```

---

## Bahdanau Attention Flow

For each decoding step:

1. take the current decoder hidden state
2. compare it with every encoder output
3. compute additive attention scores using learned linear layers
4. mask padded source positions
5. apply softmax over source positions
6. compute a weighted sum of encoder outputs
7. concatenate context with the target embedding before passing into the decoder LSTM

The implementation stores attention weights so they can later be visualized.

---

## File Overview

### `models.py`

Contains:

- `Encoder`
- `BahdanauAttention`
- `Decoder`
- `Model`

Important implementation details:

- encoder uses `pack_padded_sequence`
- attention uses learned projections for decoder hidden state and encoder outputs
- decoder LSTM input is `embedding + context`
- output layer receives decoder output, context vector, and embedding

### `trainer.py`

Contains the training loop for one epoch.

The loop:

1. runs the model over a batch
2. receives predictions and attention weights
3. shifts target labels with `output_ids[:, 1:]`
4. computes cross-entropy loss
5. clips gradients
6. updates parameters

### `train.py`

Main runnable script.

It loads data, preprocesses text, builds vocabularies, creates dataloaders, initializes the Bahdanau attention model, trains it, and runs a small translation example.

---

## How to Run

From the repository root:

```bash
python -m seq2seq.bahdanau_attention.train.py
```

---

## Current Hyperparameters

The script currently uses:

```python
num_epochs = 10
embed_dim = 128
hidden_dim = 256
batch_size = 32
learning_rate = 0.01
attention_dim = 256
```

These are learning-oriented defaults and should be tuned for stronger translation quality.

---

## Bahdanau vs Vanilla Encoder-Decoder

| Model | Decoder Can Access | Main Limitation |
|---|---|---|
| Vanilla encoder-decoder | final encoder hidden/cell state | source sentence compressed into one recurrent state |
| Bahdanau attention | all encoder outputs through learned attention | slower autoregressive recurrent decoding |

Bahdanau attention is usually easier to understand first because attention is computed before the decoder produces the next recurrent output.

---

## Future Work

- [ ] Add attention heatmap visualization
- [ ] Add checkpoint saving/loading
- [ ] Add BLEU score evaluation
- [ ] Add validation and test split
- [ ] Add beam search decoding
- [ ] Add teacher forcing ratio control
- [ ] Compare against vanilla encoder-decoder and Luong attention
- [ ] Add CLI/config support
