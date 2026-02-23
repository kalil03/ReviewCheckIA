"""
Step 3: Preparação do dataset para treinamento.
Tokeniza com BERTimbau, cria splits train/val/test, computa class weights.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_WEIGHTS_FILE, MAX_LENGTH, MODEL_NAME, NUM_LABELS,
    PROCESSED_FILE, RANDOM_SEED, TEST_RATIO, TOKENIZED_DIR,
    TRAIN_RATIO, VAL_RATIO,
)


def main():
    print("\n--- STEP 3: DATASET PREPARATION ---")

    if not PROCESSED_FILE.exists():
        print(f"Error: {PROCESSED_FILE} not found.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_FILE, encoding="utf-8")
    print(f"Loaded: {len(df):,} reviews")

    print(f"\nCreating splits ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["sentiment"], random_state=RANDOM_SEED,
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["sentiment"], random_state=RANDOM_SEED,
    )

    print(f"   Train:      {len(train_df):>7,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df):>7,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:       {len(test_df):>7,} ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\nSplit distributions:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["sentiment"].value_counts(normalize=True).sort_index()
        parts = " | ".join(f"{v*100:.1f}%" for v in dist)
        print(f"   {name:>5s}: {parts}")

    print(f"\nComputing class weights...")
    classes = np.array(sorted(df["sentiment"].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=train_df["sentiment"].values)
    weights_dict = {str(int(c)): float(w) for c, w in zip(classes, weights)}

    print(f"   Weights: {weights_dict}")
    with open(CLASS_WEIGHTS_FILE, "w") as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\nTokenizing with {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(
            train_df[["text", "sentiment"]].rename(columns={"sentiment": "label"}).reset_index(drop=True)
        ),
        "validation": Dataset.from_pandas(
            val_df[["text", "sentiment"]].rename(columns={"sentiment": "label"}).reset_index(drop=True)
        ),
        "test": Dataset.from_pandas(
            test_df[["text", "sentiment"]].rename(columns={"sentiment": "label"}).reset_index(drop=True)
        ),
    })

    print(f"   Tokenizing train ({len(dataset['train']):,})...")
    dataset = dataset.map(tokenize, batched=True, batch_size=1000)

    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    dataset.save_to_disk(str(TOKENIZED_DIR))
    
    print(f"\nSaved tokenized dataset to: {TOKENIZED_DIR}")
    for split in dataset:
        print(f"   {split:>12s}: {len(dataset[split]):,} samples")


if __name__ == "__main__":
    main()
