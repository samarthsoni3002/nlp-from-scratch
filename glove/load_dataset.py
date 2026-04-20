from datasets import load_dataset

def get_dataset():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    train_ds = ds["train"]
    valid_ds = ds["validation"]
    test_ds = ds["test"]
    return train_ds, valid_ds, test_ds