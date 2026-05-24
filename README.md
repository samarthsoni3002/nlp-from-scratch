# NLP From Scratch

A learning-first repository where I implement core NLP models, papers, and sequence modeling ideas from scratch using PyTorch.

The goal of this repository is not to hide the important parts behind high-level libraries. The goal is to understand how NLP systems are actually built: preprocessing, vocabularies, datasets, batching, model equations, training loops, inference, and evaluation.

This repo is intentionally written as a **paper-to-code / concept-to-code learning repo**. The implementations favor explicit tensor flows and readable training logic over heavy abstraction.

---

## Repository Goal

This repo is my running collection of NLP implementations from first principles.

Each folder focuses on one model family or paper idea and tries to answer:

- What is the core intuition?
- What data format does the model need?
- What tensors flow through the model?
- What objective is being optimized?
- How does training actually happen?
- How can the learned model be inspected or used?

---

## Current Implementations

| Folder | Topic | Current Status |
|---|---|---|
| `word2vec/` | Word2Vec from scratch | Skip-gram, CBOW, negative sampling, checkpointing, inference |
| `glove/` | GloVe from scratch | Co-occurrence construction, weighted least-squares objective, inference |
| `sequence_models/` | RNN, LSTM, GRU from scratch | IMDB sentiment classification with manually implemented recurrent cells |
| `seq2seq/vanilla_encoder_decoder/` | Vanilla encoder-decoder | LSTM English-to-French translation baseline |
| `seq2seq/bahdanau_attention/` | Additive attention | Encoder-decoder with Bahdanau-style attention over encoder states |
| `seq2seq/luong_attention/` | Multiplicative attention | Encoder-decoder with Luong-style dot-product attention |
| `seq2seq/transformer_attention/` | Transformer attention | Single-layer Transformer-style encoder-decoder with multi-head attention |

---

## Repository Structure

```bash
nlp-from-scratch/
├── README.md
│
├── word2vec/
├── glove/
├── sequence_models/
└── seq2seq/
    ├── README.md
    ├── eng-fra.txt
    ├── datasets_classes.py
    ├── preprocessing.py
    ├── utils.py
    ├── inference.py
    ├── vanilla_encoder_decoder/
    │   ├── README.md
    │   ├── models.py
    │   ├── trainer.py
    │   └── train.py
    ├── bahdanau_attention/
    │   ├── README.md
    │   ├── models.py
    │   ├── trainer.py
    │   └── train.py
    ├── luong_attention/
    │   ├── README.md
    │   ├── models.py
    │   ├── trainer.py
    │   └── train.py
    └── transformer_attention/
        ├── README.md
        ├── models.py
        ├── trainer.py
        └── train.py
```

---

## Implemented Projects

### 1. Word2Vec

The `word2vec/` folder implements the core Word2Vec family in PyTorch.

Implemented variants:

- Skip-gram
- CBOW
- Skip-gram with negative sampling
- CBOW with negative sampling

The implementation includes preprocessing, vocabulary construction, custom dataset classes, training/testing loops, checkpoint saving, and inference utilities for vector lookup and nearest-neighbor similarity search.

---

### 2. GloVe

The `glove/` folder implements **GloVe: Global Vectors for Word Representation**.

The implementation includes corpus preprocessing, vocabulary construction, word-word co-occurrence generation, distance-weighted co-occurrence counting, separate word/context embeddings, separate bias terms, weighted least-squares loss, and nearest-neighbor inference.

---

### 3. Sequence Models

The `sequence_models/` folder implements foundational recurrent sequence models from scratch for sentiment classification.

Implemented models:

- Vanilla RNN
- LSTM
- GRU

The recurrent cells are manually implemented using `nn.Parameter` tensors instead of directly using `nn.RNN`, `nn.LSTM`, or `nn.GRU`. This keeps the gate equations and hidden-state updates visible.

Current task:

- IMDB binary sentiment classification

---

### 4. Seq2Seq Models

The `seq2seq/` folder builds a progression of English-to-French translation models:

1. **Vanilla encoder-decoder** — compresses the source sentence into final LSTM hidden/cell states.
2. **Bahdanau attention** — lets the decoder attend over all encoder outputs before generating each target token.
3. **Luong attention** — computes attention after the decoder state is produced using dot-product scoring.
4. **Transformer attention** — replaces recurrence with positional encodings, self-attention, cross-attention, residual connections, layer normalization, and feed-forward blocks.

This section is designed to show the historical path from recurrent encoder-decoder models to attention-based models and then Transformer-style architectures.

---

## Project Style

This repository is intentionally written in a **learning-first** style.

That means:

- core mechanics are kept explicit
- model equations are translated directly into code
- training loops are written manually
- preprocessing and batching are visible
- each implementation is kept mostly self-contained
- clarity is preferred over heavy abstraction

This is not meant to be a polished NLP library. It is meant to be a deep learning notebook in repository form.

---

## Current Limitations

The repository is still evolving. Some current limitations are:

- many hyperparameters are still hardcoded inside `train.py`
- not every implementation has checkpointing yet
- evaluation is simple in some folders
- no unified CLI interface yet
- no global experiment tracking yet
- no shared configuration system yet
- seq2seq inference is still being unified across vanilla, attention, and Transformer variants
- code is optimized for learning clarity more than speed

---

## Roadmap

Planned improvements:

- [ ] Add CLI/config support for experiments
- [ ] Add seed control and reproducibility utilities
- [ ] Add checkpointing consistently across folders
- [ ] Add training/validation plots
- [ ] Add better evaluation metrics such as accuracy, F1, BLEU, and embedding analogy tests
- [ ] Add cleaner experiment organization
- [ ] Add attention visualization for seq2seq models
- [ ] Expand Transformer implementation toward the full paper architecture
- [ ] Add stronger README examples after each implementation stabilizes

---

## Why This Repository Exists

I created this repository to build strong NLP intuition by implementing models manually.

Reading papers is useful, but implementing them forces you to understand:

- what each symbol means in code
- what the model input/output shapes are
- how batches are constructed
- how losses are computed
- how training and inference differ
- where engineering details matter

This repo is my way of going from **paper understanding** to **working implementation**.

---

## Final Note

This repository is less about building a benchmark-ready library and more about building a deep, practical, paper-level understanding of NLP models.

Over time, this will grow into a larger collection of NLP models implemented from scratch.
