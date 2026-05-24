# Luong Attention from Scratch

This folder contains a from-scratch PyTorch implementation of an LSTM encoder-decoder model with **Luong-style dot-product attention** for English-to-French translation.

Luong attention improves the vanilla encoder-decoder model by letting the decoder attend over all encoder outputs. In this implementation, attention scores are computed using dot products between encoder outputs and the current decoder state.

---

## What This Implements

This implementation includes:

- LSTM encoder with packed sequences
- Luong dot-product attention module
- LSTM decoder
- source padding mask
- teacher-forced decoding during training
- attention weight collection for each target step
- cross-entropy loss with `<PAD>` ignored
- gradient clipping
- greedy translation example after training

---

## Folder Structure

```bash
seq2seq/luong_attention/
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

Luong attention computes the decoder output first, then compares that decoder state with each encoder output.

In this implementation:

```text
score_i = encoder_output_i dot decoder_hidden
```

The scores are normalized with softmax and used to create a context vector from encoder outputs.

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
target token embedding
   |
   v
LSTM decoder output
   |
   v
Luong dot-product attention over encoder outputs
   |
   v
context vector + decoder output
   |
   v
target vocabulary logits
```

---

## Luong Attention Flow

For each decoding step:

1. embed the current target input token
2. run one decoder LSTM step
3. use the decoder output as the query
4. compute dot-product scores with every encoder output
5. mask padded source positions
6. apply softmax over source positions
7. compute the context vector as a weighted sum of encoder outputs
8. concatenate decoder output and context vector
9. project to target vocabulary logits

---

## File Overview

### `models.py`

Contains:

- `Encoder`
- `LuongAttention`
- `Decoder`
- `Model`

Important implementation details:

- encoder uses packed sequences
- attention uses `torch.bmm` for batched dot-product scoring
- decoder output is concatenated with the attention context vector before the final linear layer

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

It loads data, preprocesses text, builds vocabularies, creates dataloaders, initializes the Luong attention model, trains it, and runs a small translation example.

---

## How to Run

From the repository root:

```bash
python -m seq2seq.luong_attention.train.py
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
```

These are learning-oriented defaults and should be tuned for stronger translation quality.

---

## Luong vs Bahdanau Attention

| Attention Type | Score Function | When Attention Is Computed |
|---|---|---|
| Bahdanau | learned additive scoring | before decoder recurrent update |
| Luong | dot-product scoring | after decoder recurrent update |

Luong attention is simpler in this implementation because the score is directly computed with a dot product instead of learned additive projections.

---


## Future Work

- [ ] Add attention heatmap visualization
- [ ] Add checkpoint saving/loading
- [ ] Add BLEU score evaluation
- [ ] Add validation and test split
- [ ] Add beam search decoding
- [ ] Implement Luong general scoring
- [ ] Implement Luong concat scoring
- [ ] Compare against vanilla encoder-decoder and Bahdanau attention
- [ ] Add CLI/config support
