from datasets import load_dataset

def get_dataset():
    ds = load_dataset("imdb")
    split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    valid_ds = split_ds["test"]
    test_ds = ds["test"]
    return train_ds, valid_ds, test_ds