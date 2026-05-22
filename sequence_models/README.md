# Sequence Models from Scratch

This folder contains from-scratch PyTorch implementations of foundational recurrent sequence models:

- Vanilla RNN
- LSTM
- GRU

The current task is **binary sentiment classification** on the IMDB dataset.

The goal is to understand how recurrent neural networks process text one token at a time and how hidden states are used for sequence classification.

---

## What This Implements

This folder includes:

- IMDB dataset loading through Hugging Face Datasets
- text preprocessing for sentiment analysis
- contraction expansion
- stopword removal while preserving important sentiment words
- vocabulary construction with minimum-frequency filtering
- token-to-id conversion
- custom PyTorch dataset
- dynamic padding collate function
- manually implemented RNN cell
- manually implemented LSTM gates
- manually implemented GRU gates
- training, validation, and testing loops

The recurrent models use `nn.Parameter` tensors for their weights instead of calling PyTorch's built-in `nn.RNN`, `nn.LSTM`, or `nn.GRU`.

---

## Folder Structure

```bash
sequence_models/
├── README.md
├── load_dataset.py
├── preprocessing.py
├── data_classes.py
├── trainer.py
├── utils.py
├── train.py
└── models/
    ├── rnn.py
    ├── lstm.py
    └── gru.py
```

---

## File Overview

### `load_dataset.py`

Loads the IMDB dataset:

```python
load_dataset("imdb")
```

Current split logic:

- original train split is divided into train and validation
- original test split is used as test data

---

### `preprocessing.py`

Handles text preprocessing and vocabulary logic.

Current preprocessing pipeline:

1. Unicode normalization
2. ASCII conversion
3. lowercasing
4. contraction expansion
5. regex-based English word extraction
6. stopword removal
7. token filtering

The pipeline keeps important sentiment-bearing words such as:

```python
"not", "no", "nor", "never", "but", "very", "too", "so"
```

This matters because removing negation words can destroy sentiment meaning.

Example:

```text
"not good" != "good"
```

---

### `utils.py`

Contains helper utilities such as contraction expansion.

---

### `data_classes.py`

Contains:

- `SentiDataset`
- `sentiment_collate_fn`

The collate function dynamically pads each batch and returns:

```python
padded_input_ids, labels, lengths
```

The `lengths` tensor is important because the model should use the final real token, not the final padded token.

---

### `models/rnn.py`

Contains a manually implemented vanilla RNN.

Core recurrence:

```text
h_t = tanh(x_t W_x + h_{t-1} W_h + b_h)
```

After processing all tokens, the model selects the hidden state corresponding to the last non-padded token and maps it to class logits.

---

### `models/lstm.py`

Contains a manually implemented LSTM.

The LSTM uses:

- forget gate
- input gate
- candidate cell state
- output gate
- cell state
- hidden state

Core idea:

```text
the cell state carries long-term memory
the gates decide what to forget, write, and expose
```

This helps reduce the vanishing-gradient problem compared to a vanilla RNN.

---

### `models/gru.py`

Contains a manually implemented GRU.

The GRU uses:

- reset gate
- update gate
- candidate hidden state

Core idea:

```text
the update gate controls how much old hidden state is kept
the reset gate controls how much past information is used when creating the candidate state
```

GRU is simpler than LSTM because it does not maintain a separate cell state.

---

### `trainer.py`

Contains:

- `training_loop`
- `testing_loop`

The training loop currently reports:

- average training loss
- average validation loss

The testing loop reports final test loss.

---

### `train.py`

Main training script.

It currently:

1. loads IMDB
2. preprocesses text
3. builds vocabulary
4. maps tokens to ids
5. creates train/validation/test datasets
6. creates dataloaders
7. selects one model: RNN, LSTM, or GRU
8. trains the model
9. evaluates on the test set

Model selection is controlled by editing:

```python
model = "gru"
```

Supported values:

```python
model = "rnn"
model = "lstm"
model = "gru"
```

---

## Data Flow

```text
Raw IMDB review
      |
      v
cleaned tokens
      |
      v
token ids
      |
      v
padded batch + sequence lengths
      |
      v
RNN / LSTM / GRU
      |
      v
final non-padded hidden state
      |
      v
linear classifier
      |
      v
positive / negative logits
```

---


## How to Run

From the repository root:

```bash
python sequence_models/train.py
```

To switch models, edit `train.py`:

```python
model = "rnn"
```

or

```python
model = "lstm"
```

or

```python
model = "gru"
```

---

## Current Training Setup

The current script is configured for quick experimentation.

It uses:

```python
batch_size = 32
embed_dim = 100
hidden_size = 128
output_size = 2
learning_rate = 0.001
num_epochs = 1
```

It also currently trains on a small subset:

```python
tokenized_train[:1000]
tokenized_val[:1000]
tokenized_test[:1000]
```

This is useful for debugging and learning, but should be increased for meaningful performance.

---

## What These Models Teach

### RNN

A vanilla RNN shows the basic idea of recurrence:

```text
current input + previous hidden state -> new hidden state
```

It is simple, but it can struggle with long-range dependencies.

---

### LSTM

LSTM improves the RNN by adding a memory cell and gates.

It teaches:

- how memory is preserved
- how information is forgotten
- how new information is written
- how hidden state is exposed

---

### GRU

GRU simplifies the LSTM while keeping gating.

It teaches:

- how update gates control memory retention
- how reset gates control candidate state creation
- how fewer gates can still work well in practice

---

## Current Limitations

This folder is intentionally learning-oriented and still evolving.

Current limitations:

- model selection is hardcoded in `train.py`
- hyperparameters are hardcoded
- training currently uses a small subset of IMDB
- only loss is reported; accuracy is not reported yet
- no checkpoint saving/loading yet
- no CLI interface yet
- no dropout or regularization yet
- no pretrained embeddings yet
- no bidirectional models yet
- no packed-sequence optimization yet
- no plots for training/validation curves yet

---

## Future Work / To Do

Planned improvements:

- [ ] Add accuracy, precision, recall, and F1 score
- [ ] Add checkpoint saving and loading
- [ ] Add command-line arguments for model selection and hyperparameters
- [ ] Add device support for GPU training
- [ ] Add dropout
- [ ] Add gradient clipping
- [ ] Add packed sequence support
- [ ] Add bidirectional RNN/LSTM/GRU
- [ ] Add pretrained embedding option
- [ ] Add training and validation loss plots
- [ ] Add inference script for custom review text
- [ ] Compare RNN vs LSTM vs GRU results

---

## Final Note

This folder is the foundation for understanding sequence models before moving to encoder-decoder architectures, attention, and Transformers.

The purpose is not only to train a sentiment classifier, but to deeply understand how recurrent neural networks process language step by step.
