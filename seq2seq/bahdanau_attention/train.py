import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim

from seq2seq.datasets_classes import TranslationDataset, translation_collate_fn
from seq2seq.bahdanau_attention.models import Encoder, Decoder, BahdanauAttention, Model
from seq2seq.utils import create_data, translate_sentence
from seq2seq.preprocessing import basic_preprocessing, filter_pairs, build_vocab, vocab_map
from seq2seq.bahdanau_attention.trainer import train
from seq2seq.inference import translate_sentence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path = "seq2seq/eng-fra.txt"

num_epochs = 10
embed_dim = 128
hidden_dim = 256

X,y = create_data(file_path)

print("Data Created Successfully!")

X_tokens = basic_preprocessing(X)["tokens"]
y_tokens = basic_preprocessing(y)["tokens"]

X_tokens, y_tokens = filter_pairs(X_tokens, y_tokens)

X_tokens = [tokens + ["<EOS>"] for tokens in X_tokens]

y_tokens = [["<SOS>"] + tokens + ["<EOS>"] for tokens in y_tokens]

print("Basic preprocessing done successfully!")

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

vocab_eng, id_to_word_eng, counter_eng = build_vocab(X_tokens, SPECIAL_TOKENS,min_freq=2)
vocab_fr, id_to_word_fr, counter_fr = build_vocab(y_tokens, SPECIAL_TOKENS,min_freq=2)

X_conv = vocab_map(X_tokens, vocab_eng)
y_conv = vocab_map(y_tokens, vocab_fr)

print("Data converted to integers!")

translation_dataset = TranslationDataset(X_conv,y_conv)
translation_dataloader = DataLoader(translation_dataset,batch_size=32,collate_fn=translation_collate_fn,shuffle=True)

print("Dataloader created!")



model = Model(
    Encoder(embed_dim, hidden_dim, len(vocab_eng), pad_idx=vocab_eng["<PAD>"]),
    Decoder(embed_dim, hidden_dim, len(vocab_fr), pad_idx=vocab_fr["<PAD>"])
)

criterion = nn.CrossEntropyLoss(ignore_index=vocab_fr["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


print("Starting Training!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_id = vocab_fr["<PAD>"]


for epoch in range(num_epochs):
    train_loss = train(
        model=model.to(device),
        dataloader=translation_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}")
    
print("Training Finished")

translation = translate_sentence(
    sentence="I Love You",
    model=model,
    vocab_eng=vocab_eng,
    vocab_fr=vocab_fr,
    id_to_word_fr=id_to_word_fr,
    preprocess_fn=basic_preprocessing,
    device=device,
    max_len=30
)

print(translation)