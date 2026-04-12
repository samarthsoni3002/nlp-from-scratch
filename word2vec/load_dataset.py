from datasets import load_dataset

def get_dataset():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    train_ds = ds["train"]
    return train_ds