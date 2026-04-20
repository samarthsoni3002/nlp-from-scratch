# GloVe from Scratch (PyTorch)

This folder contains a from-scratch PyTorch implementation of **GloVe (Global Vectors for Word Representation)** based on the Stanford paper by Pennington, Socher, and Manning.

The implementation is built to make the paper concrete:
- preprocess a raw corpus
- build a vocabulary
- construct a **word-word co-occurrence matrix** through dataset generation
- train separate **word** and **context** embeddings with bias terms
- optimize the **weighted least-squares GloVe objective**
- run simple inference with nearest-neighbor similarity

---

## What is implemented

### 1. Dataset loading
- Uses **WikiText-103** through Hugging Face Datasets.
- File: `load_dataset.py`

### 2. Preprocessing
- lowercasing
- ASCII normalization
- regex-based token extraction
- vocabulary building with minimum-frequency filtering
- mapping tokenized sentences to integer ids
- File: `preprocessing.py`

### 3. Co-occurrence dataset construction
- Builds **non-zero word-context co-occurrence pairs**
- Uses **distance-weighted counts**: a context word at distance `d` contributes `1/d`
- Stores training samples as `(word_i, word_j, x_ij)`
- File: `datasets_classes.py`

### 4. GloVe model
- separate target-word embedding matrix
- separate context-word embedding matrix
- separate bias vectors for both sides
- forward score:

\[
\hat{x}_{ij} = w_i^T \tilde{w}_j + b_i + \tilde{b}_j
\]

- File: `models.py`

### 5. GloVe weighting function and loss
- weighting function:

\[
f(x)=
\begin{cases}
(x/x_{max})^{\alpha} & x < x_{max} \\
1 & x \ge x_{max}
\end{cases}
\]

- weighted loss:

\[
J = \sum_{i,j} f(X_{ij})\left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
\]

- File: `utils.py`

### 6. Training / validation / testing loops
- standard PyTorch training loop
- separate validation and testing passes
- File: `trainer.py`

### 7. Inference
- builds final word vectors using:

\[
W + \tilde{W}
\]

- includes nearest-neighbor lookup with cosine similarity
- File: `inference.py`

---

## Folder structure

```bash
glove/
├── README.md
├── datasets_classes.py
├── inference.py
├── load_dataset.py
├── models.py
├── preprocessing.py
├── requirements.txt
├── train.py
├── trainer.py
└── utils.py
```

---


## How the implementation maps to the paper

This code follows the core GloVe formulation well:
- trains on **global co-occurrence counts** rather than local prediction examples
- uses **non-zero co-occurrence entries**
- uses **distance-based co-occurrence accumulation**
- has **two embedding matrices** and **two bias terms**
- optimizes the **weighted squared-error GloVe objective**
- uses **W + W_tilde** at inference time


---

## How to run

From inside the `glove/` folder:

```bash
python train.py
```

After training, a checkpoint is saved under:

```bash
save/model_checkpoint_glove_model.pt
```

Then run inference with:

```bash
python inference.py
```

---

## Current Limitations

This project is intentionally written in a learning-oriented style, so some design choices are currently simple and manual.

Current limitations include:

- training configuration is hardcoded in `train.py`
- inference configuration is hardcoded in `inference.py`
- there is no CLI-based experiment interface yet
- no visualization of embeddings yet
- no analogy evaluation yet
- hierarchical softmax is not implemented yet
- the code is more focused on understanding than optimization

---

## Future Work / To Do

Planned improvements:

- [ ] Convert training and inference scripts to a **command-line argument based interface**
- [ ] Add **embedding visualization** (for example, PCA / t-SNE)
- [ ] Add **analogy-based evaluation**
- [ ] Add **training and validation loss plotting**
- [ ] Add **seed control / reproducibility support**
- [ ] Improve checkpoint naming and experiment organization
- [ ] Add better documentation and examples

---
