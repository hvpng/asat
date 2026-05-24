# data/load_data.py
# Unified data loading for all datasets: SST-2, IMDB, Yelp Polarity.

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_CONFIGS = {
    "sst2": {
        "hf_name": "glue",
        "hf_config": "sst2",
        "text_col": "sentence",
        "label_col": "label",
        "splits": {"train": "train", "val": "validation", "test": "validation"},
    },
    "imdb": {
        "hf_name": "imdb",
        "hf_config": None,
        "text_col": "text",
        "label_col": "label",
        "splits": {"train": "train", "val": "test", "test": "test"},
    },
    "yelp": {
        "hf_name": "yelp_polarity",
        "hf_config": None,
        "text_col": "text",
        "label_col": "label",
        "splits": {"train": "train", "val": "test", "test": "test"},
    },
}


def load_and_tokenize(
    dataset_name,
    tokenizer_name="bert-base-uncased",
    max_length=128,
    train_subset=None,
    val_subset=None,
):
    """
    Load and tokenize a dataset.

    Args:
        dataset_name : "sst2", "imdb", or "yelp"
        tokenizer_name: HuggingFace model name
        max_length    : max sequence length
        train_subset  : if set, only use first N training samples (pilot mode)
        val_subset    : same for validation

    Returns:
        splits (dict): keys "train", "val", "test" -> tokenized HF Dataset
        tokenizer    : the loaded tokenizer
    """
    assert dataset_name in DATASET_CONFIGS, \
        f"dataset_name must be one of: {list(DATASET_CONFIGS.keys())}"

    cfg = DATASET_CONFIGS[dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load raw dataset
    if cfg["hf_config"]:
        raw = load_dataset(cfg["hf_name"], cfg["hf_config"])
    else:
        raw = load_dataset(cfg["hf_name"])

    # Tokenize using "text" column (after rename)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    splits = {}
    for split_key, hf_split in cfg["splits"].items():
        ds = raw[hf_split]

        # Optional subset for pilot experiments
        if split_key == "train" and train_subset is not None:
            ds = ds.select(range(min(train_subset, len(ds))))
        if split_key == "val" and val_subset is not None:
            ds = ds.select(range(min(val_subset, len(ds))))

        # Rename columns to standard names before tokenizing
        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        if cfg["label_col"] != "label":
            ds = ds.rename_column(cfg["label_col"], "label")

        # Tokenize
        ds = ds.map(tokenize_fn, batched=True, desc=f"Tokenizing {split_key}")

        # Keep only columns the model needs
        keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in keep]
        )
        ds.set_format("torch")
        splits[split_key] = ds

    print(f"\nLoaded {dataset_name}:")
    for k, v in splits.items():
        print(f"  {k}: {len(v):,} samples")

    return splits, tokenizer


if __name__ == "__main__":
    splits, tok = load_and_tokenize("sst2", train_subset=100, val_subset=50)
    sample = splits["train"][0]
    print(f"\nSample keys  : {list(sample.keys())}")
    print(f"input_ids    : {sample['input_ids'].shape}")
    print(f"label        : {sample['label']}")