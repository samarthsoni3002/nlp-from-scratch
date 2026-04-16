# Word2Vec from Scratch

This folder contains my from-scratch implementation of the **Word2Vec** family in PyTorch. The goal of this project was not just to use an existing library, but to understand how Word2Vec works end-to-end: preprocessing, dataset construction, model definition, training, negative sampling, checkpointing, and inference.

## Implemented Variants

This implementation currently supports four Word2Vec variants:

- **Skip-gram**
- **CBOW**
- **Skip-gram with Negative Sampling**
- **CBOW with Negative Sampling**

## What is Included

This folder includes:

- dataset loading
- preprocessing and vocabulary creation
- custom dataset classes for skip-gram and CBOW
- custom collate functions for CBOW batching and CBOW negative sampling
- PyTorch model implementations for all four variants
- separate training / validation / testing loops
- checkpoint saving
- inference utilities for:
  - loading a saved checkpoint
  - retrieving a word vector
  - finding the top-k most similar words using cosine similarity

---

## Folder Structure

```bash
word2vec/
├── load_dataset.py
├── preprocessing.py
├── datasets_classes.py
├── models.py
├── trainer.py
├── utils.py
├── train.py
├── inference.py
└── save/
```

### File Overview

#### `load_dataset.py`
Loads the dataset splits used for training, validation, and testing.

#### `preprocessing.py`
Handles preprocessing steps such as tokenization, vocabulary construction, and mapping tokens to ids.

#### `datasets_classes.py`
Contains:

- `SkipgramDataset`
- `CbowDataset`
- `cbow_collate`
- `cbow_collate_negative_sampling`

#### `models.py`
Contains the following model classes:

- `SkipgramModel`
- `SkipgramModelNegativeSampling`
- `CbowModel`
- `CbowModelNegativeSampling`

#### `trainer.py`
Contains training and testing loops for:

- standard Skip-gram / CBOW
- negative sampling versions of Skip-gram / CBOW

#### `utils.py`
Contains helper utilities such as:

- negative sampling loss
- noise distribution construction
- negative word sampling
- checkpoint loading
- vector lookup
- nearest-neighbor similarity search

#### `train.py`
Main training script. This is where model type, negative sampling flag, and other hyperparameters are currently selected.

#### `inference.py`
Loads a saved checkpoint and performs simple inference on learned embeddings.

---

## Dataset

The current implementation uses the **WikiText** dataset via Hugging Face Datasets, specifically:

- `Salesforce/wikitext`
- configuration: `wikitext-103-v1`

At the moment, training is done on a smaller subset for faster experimentation.

---

## Implemented Training Objectives

### 1. Skip-gram
Given a **center word**, predict surrounding **context words**.

### 2. CBOW
Given surrounding **context words**, predict the **center word**.

### 3. Skip-gram + Negative Sampling
Instead of computing scores over the entire vocabulary, the model learns to:

- assign a high score to the true context word
- assign low scores to sampled negative words

### 4. CBOW + Negative Sampling
The model averages context embeddings and learns to:

- assign a high score to the true target word
- assign low scores to sampled negative words

---

## How to Train

Training is currently controlled by editing configuration variables inside `train.py`.

Example variables:

```python
dataset_length = 1000
w2v_model = "cbow"
negative_sampling = True
num_epochs = 1
neg_samples = 5
embedding_dim = 3
```

### Run training

```bash
python train.py
```

Depending on the selected configuration, this will:

1. load and preprocess the data
2. create datasets and dataloaders
3. initialize the selected model
4. train and validate the model
5. evaluate on the test set
6. save the trained checkpoint in `./save/`

---

## Saved Checkpoints

The following checkpoint files are currently used:

- `./save/model_checkpoint_skipgram.pt`
- `./save/model_checkpoint_skipgram_ns.pt`
- `./save/model_checkpoint_cbow.pt`
- `./save/model_checkpoint_cbow_ns.pt`

These checkpoints store:

- `model_state_dict`
- `word_to_id`
- `id_to_word`
- `vocab_size`
- `embedding_dim`
- `pad_id` for CBOW-based models

---

## How to Run Inference

Inference is currently controlled by editing variables inside `inference.py`.

Example:

```python
# model_name = "skip-gram"
# model_name = "cbow"
# model_name = "skip-gram-ns"
model_name = "cbow-ns"

word = "human"
```

### Run inference

```bash
python inference.py
```

This will:

- load the selected checkpoint
- print the learned embedding vector for the chosen word
- print the top-k most similar words using cosine similarity

---

## Example Inference Tasks

The current inference utilities support:

- **word vector lookup**
- **top-k nearest neighbors**
- cosine similarity based semantic similarity search

This makes it easy to inspect whether the embeddings are learning useful word relationships.

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

- [ ] Implement **hierarchical softmax**
- [ ] Convert training and inference scripts to a **command-line argument based interface**
- [ ] Add **embedding visualization** (for example, PCA / t-SNE)
- [ ] Add **analogy-based evaluation**
- [ ] Add **training and validation loss plotting**
- [ ] Add **seed control / reproducibility support**
- [ ] Improve checkpoint naming and experiment organization
- [ ] Refactor training code to reduce repetition across model variants
- [ ] Improve the negative sampling pipeline for cleaner reuse and better efficiency
- [ ] Add better documentation and examples for each variant

---

## Why This Project

The purpose of this implementation is to deeply understand the mechanics behind Word2Vec rather than treat it as a black box.

Through this project, I wanted to understand:

- how Skip-gram and CBOW differ
- how negative sampling changes training
- how custom NLP datasets and collate functions are built
- how learned embeddings can be saved and reused for inference
- how paper ideas translate into actual training code

This folder is part of a larger repository where I implement NLP papers and core ideas from scratch to build stronger intuition and practical understanding.

---

## Notes

This implementation is still evolving. The current version is a working from-scratch Word2Vec pipeline with training, evaluation, checkpointing, and inference. Hierarchical softmax and cleaner experiment management are planned as future improvements.
