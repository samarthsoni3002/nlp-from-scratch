import torch
import torch.nn.functional as F
from utils import  get_word_vector, most_similar
from models import GloveModel



word = "human"

checkpoint_path = "save/model_checkpoint_glove_model.pt"

checkpoint = torch.load(checkpoint_path)

vocab_size = checkpoint["vocab_size"]
embedding_dim = checkpoint["embedding_dim"]

word_to_id = checkpoint["word_to_id"]
id_to_word = checkpoint["id_to_word"]

model = GloveModel(vocab_size=vocab_size, embedding_dim=embedding_dim)


vector = get_word_vector(
    word=word,
    model=model,
    word_to_id=word_to_id
)

print(vector)

top_k_words = most_similar(
    word=word,
    model=model,
    word_to_id=word_to_id,
    id_to_word=id_to_word,
    top_k=5
)

print(top_k_words)