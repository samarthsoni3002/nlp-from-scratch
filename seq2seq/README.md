# Seq2Seq Models from Scratch

This folder contains a progression of sequence-to-sequence models for English-to-French translation.

The goal is to understand how neural machine translation evolved from a simple recurrent encoder-decoder architecture to attention-based models and then Transformer-style attention.

Current task:

```text
English sentence -> French sentence
```

---

## Implemented Models

| Folder | Model | Core Idea |
|---|---|---|
| `vanilla_encoder_decoder/` | LSTM encoder-decoder | Encode the full source sentence into final hidden/cell states, then decode target tokens autoregressively. |
| `bahdanau_attention/` | Bahdanau additive attention | Decoder attends over all encoder outputs before each decoding step. |
| `luong_attention/` | Luong dot-product attention | Decoder first produces a state, then scores encoder outputs using dot-product attention. |
| `transformer_attention/` | Transformer-style attention | Uses positional encodings, multi-head self-attention, encoder-decoder cross-attention, residuals, layer norm, and feed-forward blocks. |

---

## Shared Files

```bash
seq2seq/
├── README.md
├── eng-fra.txt
├── datasets_classes.py
├── preprocessing.py
├── utils.py
├── inference.py
├── vanilla_encoder_decoder/
├── bahdanau_attention/
├── luong_attention/
└── transformer_attention/
```

### `eng-fra.txt`

Tab-separated English-French translation pairs.

Each row is expected to look like:

```text
english sentence<TAB>french sentence
```

### `preprocessing.py`

Handles:

- Unicode normalization
- lowercasing
- tokenization of words and punctuation
- source and target vocabulary creation
- token-to-id conversion

Special tokens:

```python
["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
```

Current sequence formatting:

```text
English input: token_1 token_2 ... token_n <EOS>
French output: <SOS> token_1 token_2 ... token_n <EOS>
```

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

The length tensors are used by the LSTM-based models for packed sequences and are useful later for masking.

### `utils.py`

Contains helper functions for reading the translation file and creating source/target sentence lists.

### `inference.py`

Contains a greedy decoding helper. This is currently most directly aligned with the vanilla recurrent encoder-decoder interface and should be cleaned up as attention/Transformer inference becomes unified.

---

## Learning Progression

### 1. Vanilla Encoder-Decoder

The source sentence is compressed into the final encoder hidden/cell states.

This teaches the basic seq2seq idea, but it creates an information bottleneck for longer sentences.

### 2. Bahdanau Attention

Bahdanau attention removes some of that bottleneck by letting the decoder look at every encoder output before generating each target token.

The score function is learned through small feed-forward layers:

```text
score(decoder_hidden, encoder_output_i)
```

### 3. Luong Attention

Luong attention also lets the decoder attend over encoder outputs, but the implementation here uses dot-product scoring after the decoder state is produced.

This is simpler and more direct than additive attention.

### 4. Transformer Attention

The Transformer-style implementation removes recurrence and uses attention as the core sequence operation.

It includes:

- sinusoidal positional encoding
- scaled dot-product attention
- multi-head attention
- encoder self-attention
- decoder causal self-attention
- encoder-decoder cross-attention
- residual connections
- layer normalization
- feed-forward blocks

---

## How to Run

From the repository root:

```bash
python -m seq2seq.vanilla_encoder_decoder.train.py
python -m seq2seq.bahdanau_attention.train.py
python -m seq2seq.luong_attention.train.py
python -m seq2seq.transformer_attention.train.py
```

Each training script currently builds the dataset, creates vocabularies, initializes the selected model, trains it, and runs a small translation example.

---


## Roadmap

Planned improvements:

- [ ] Add checkpointing for every seq2seq variant
- [ ] Add validation and BLEU evaluation
- [ ] Add beam search decoding
- [ ] Add attention visualization
- [ ] Unify inference across vanilla, Bahdanau, Luong, and Transformer models
- [ ] Add CLI/config-based experiment control
- [ ] Add training/validation curves
- [ ] Expand Transformer toward the full paper architecture
- [ ] Add clearer comparison experiments between all seq2seq variants

---

## Final Note

This folder is meant to show the architectural transition from vanilla encoder-decoder models to attention and then Transformers.

The purpose is not only to train a translation model, but to understand exactly why attention became necessary and how it changes the tensor flow inside seq2seq models.
