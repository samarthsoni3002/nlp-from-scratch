import torch
import torch.nn.functional as F
from utils import load_trained_model, get_word_vector, most_similar


#model_name = "skip-gram"
#model_name = "cbow"
#model_name = "skip-gram-ns"
model_name = "cbow-ns"



word = "human"


checkpoint_paths = {
    "skip-gram": "./save/model_checkpoint_skipgram.pt",
    "skip-gram-ns": "./save/model_checkpoint_skipgram_ns.pt",
    "cbow": "./save/model_checkpoint_cbow.pt",
    "cbow-ns": "./save/model_checkpoint_cbow_ns.pt",
 
}

checkpoint_path = checkpoint_paths[model_name]

model, word_to_id, id_to_word, checkpoint = load_trained_model(
    model_name=model_name,
    checkpoint_path=checkpoint_path,
)

vector = get_word_vector(
    word=word,
    model=model,
    word_to_id=word_to_id,
    vocab_size=checkpoint["vocab_size"],
)

print(vector)

top_k_words = most_similar(
    word=word,
    model=model,
    word_to_id=word_to_id,
    id_to_word=id_to_word,
    top_k=5,
    vocab_size=checkpoint["vocab_size"],
)

print(top_k_words)