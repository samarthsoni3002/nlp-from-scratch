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

At the moment, the main implemented project in this repository is:

- **Word2Vec**

This repository currently contains a dedicated `word2vec/` folder with a from-scratch PyTorch implementation of multiple Word2Vec variants.

---

## Repository Structure

```bash
nlp-paper-implementations/
├── word2vec/
└── README.md
```

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

## Notes

This is an actively growing repository.  
At the current stage, **Word2Vec is the main implemented project**, and future paper implementations will be added incrementally.

If you are looking for project-specific details, check the README inside the relevant folder, starting with:

- [`word2vec/`](./word2vec/)
