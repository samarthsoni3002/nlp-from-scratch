# NLP Paper Implementations

This repository is where I implement core **NLP papers and foundational ideas from scratch** to understand both the theory and the engineering behind them.

The aim is to break papers down into:

- preprocessing
- dataset construction
- model implementation
- training logic
- evaluation
- inference
- practical engineering structure

---

## Current Status

At the moment, this repository contains the following implementations:

- **Word2Vec**
- **GloVe**

Each implementation is kept in its own dedicated folder and is written in a learning-oriented style, with the goal of understanding how the paper translates into working code rather than only reproducing results.

---

## Repository Structure

```bash
nlp-paper-implementations/
├── glove/
├── word2vec/
└── README.md
```

---

## Implemented Projects

### 1. Word2Vec

The `word2vec/` folder contains a from-scratch PyTorch implementation of core Word2Vec ideas, including multiple training variants and supporting utilities for preprocessing, dataset creation, training, evaluation, and inference.

### 2. GloVe

The `glove/` folder contains a from-scratch PyTorch implementation of **GloVe (Global Vectors for Word Representation)** using:

- co-occurrence based dataset construction
- distance-weighted context counting
- separate word and context embedding matrices
- separate bias terms
- weighted least-squares training objective
- similarity-based inference from learned embeddings

---

## Why This Repository Exists

I created this repository to learn papers more deeply by implementing them manually rather than only reading them or using high-level libraries.

The focus is on understanding:

- how research ideas translate into code
- how different objectives affect training
- how to structure NLP projects cleanly
- how to move from paper intuition to an actual working implementation

This repo is meant to grow over time as I add more paper implementations.

---

## Who This Repo Is For

This repository is especially useful for:

- students learning NLP fundamentals
- people trying to go from paper to code
- anyone who wants to understand NLP models more deeply
- me, as a running notebook of implementations and learning

---

## Project Style

This repository is intentionally written in a **learning-first** style.

That means the code is primarily designed to make the ideas clear and understandable, even when there are places where future refactoring, optimization, or better experiment tooling would improve the engineering quality.

Where possible, each paper implementation is kept relatively self-contained so the project can be understood folder by folder.

---

## Notes

This is an actively growing repository.

At the current stage:

- `word2vec/` and `glove/` are the main implemented projects
- project-specific details should be documented inside each folder's own `README.md`
- future implementations will be added incrementally as the repository grows

If you are looking for project-specific details, start with:

- `word2vec/`
- `glove/`

---


## Final Note

This repository is less about building a benchmark-ready library and more about building a strong, paper-level understanding of NLP models through implementation.

As more papers are added, the repository will gradually become a larger collection of **paper-to-code NLP implementations**.
