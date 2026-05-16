# NLP From Scratch

A learning-first repository where I implement core NLP models, papers, and sequence modeling ideas from scratch using PyTorch.

The goal of this repository is not to hide the important parts behind high-level libraries. The goal is to understand how NLP systems are actually built: preprocessing, vocabularies, datasets, batching, model equations, training loops, inference, and evaluation.

---

## Repository Goal

This repo is my running collection of **paper-to-code** and **concept-to-code** NLP implementations.

Each folder focuses on one model family or paper idea and tries to answer:

- What is the core intuition?
- What data format does the model need?
- What tensors flow through the model?
- What objective is being optimized?
- How does training actually happen?
- How can the learned model be inspected or used?

This is written in a learning-oriented style, so the code is intentionally explicit and easy to trace.

---

## Current Implementations

| Folder | Topic | Status |
|---|---|---|
| `word2vec/` | Word2Vec from scratch | Skip-gram, CBOW, negative sampling, checkpointing, inference |
| `glove/` | GloVe from scratch | Co-occurrence construction, weighted least-squares objective, inference |
| `sequence_models/` | RNN, LSTM, GRU from scratch | Sentiment classification on IMDB with manually implemented recurrent cells |
| `seq2seq/vanilla-encoder-decoder/` | Vanilla encoder-decoder | LSTM-based English-to-French sequence-to-sequence translation baseline |

---

## Repository Structure

```bash
nlp-from-scratch/
├── README.md
│
├── word2vec/
│   ├── README.md
│   ├── load_dataset.py
│   ├── preprocessing.py
│   ├── datasets_classes.py
│   ├── models.py
│   ├── trainer.py
│   ├── utils.py
│   ├── train.py
│   └── inference.py
│
├── glove/
│   ├── README.md
│   ├── load_dataset.py
│   ├── preprocessing.py
│   ├── datasets_classes.py
│   ├── models.py
│   ├── trainer.py
│   ├── utils.py
│   ├── train.py
│   └── inference.py
│
├── sequence_models/
│   ├── README.md
│   ├── load_dataset.py
│   ├── preprocessing.py
│   ├── data_classes.py
│   ├── trainer.py
│   ├── utils.py
│   ├── train.py
│   └── models/
│       ├── rnn.py
│       ├── lstm.py
│       └── gru.py
│
└── seq2seq/
    └── vanilla-encoder-decoder/
        ├── README.md
        ├── datasets_classes.py
        ├── preprocessing.py
        ├── models.py
        ├── trainer.py
        ├── utils.py
        ├── train.py
        └── eng-fra.txt
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

The implementation includes:

- corpus preprocessing
- vocabulary construction
- word-word co-occurrence pair generation
- distance-weighted co-occurrence counting
- separate word and context embeddings
- separate bias terms
- weighted least-squares GloVe objective
- inference using combined word and context vectors

---

### 3. Sequence Models

The `sequence_models/` folder implements foundational recurrent sequence models from scratch for sentiment classification.

Implemented models:

- Vanilla RNN
- LSTM
- GRU

The recurrent cells are manually implemented using `nn.Parameter` tensors instead of directly using `nn.RNN`, `nn.LSTM`, or `nn.GRU`. This makes the gate mechanics and hidden-state updates visible.

Current task:

- IMDB binary sentiment classification

---

### 4. Vanilla Encoder-Decoder

The `seq2seq/vanilla-encoder-decoder/` folder implements a basic LSTM encoder-decoder model for machine translation.

Current task:

- English-to-French translation using tab-separated sentence pairs

The implementation includes:

- source and target preprocessing
- source and target vocabulary creation
- `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>` tokens
- LSTM encoder
- LSTM decoder
- teacher-forced decoder input shifting
- cross-entropy loss with padding ignored
- greedy decoding inference

This is the baseline before adding attention.

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
- evaluation is still simple in some folders
- no unified CLI interface yet
- no global experiment tracking yet
- no shared configuration system yet
- code is optimized for learning clarity more than speed

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
