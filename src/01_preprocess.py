"""
Step 1: Pré-processamento dos dados brutos do Mercado Livre.
Carrega JSONs, limpa texto, cria labels de sentimento, salva CSV processado.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MIN_TEXT_LENGTH, PROCESSED_FILE, RAW_FILES,
    RANDOM_SEED, SENTIMENT_LABELS, SENTIMENT_MAP,
)


def load_reviews(files: list[Path]) -> pd.DataFrame:
    """Carrega reviews de múltiplos arquivos JSON."""
    all_reviews = []
    for filepath in files:
        print(f"Loading {filepath.name}...")
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        print(f" -> {len(data):,} reviews")
        all_reviews.extend(data)
    return pd.DataFrame(all_reviews)


def clean_text(text: str) -> str:
    """Limpa e normaliza texto de review."""
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"[\U00010000-\U0010ffff]"
        r"|[\u2600-\u27BF]"
        r"|[\uFE00-\uFE0F]"
        r"|[\u200d]",
        " ", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return text


def extract_product_id(url: str) -> str:
    """Extrai o ID do produto da URL do Mercado Livre."""
    if not isinstance(url, str):
        return ""
    match = re.search(r"(MLB-?\d+)", url)
    return match.group(1) if match else ""


def main():
    print("\n--- STEP 1: PREPROCESSING ---")

    if not RAW_FILES:
        print("No JSON files found in data/mercadolivre_com_br/")
        sys.exit(1)

    print(f"{len(RAW_FILES)} file(s) found.")

    df = load_reviews(RAW_FILES)
    print(f"\nTotal items: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nCleaning data...")

    n_before = len(df)
    df = df.drop_duplicates(subset=["content", "product_url"], keep="first")
    print(f"Duplicates removed: {n_before - len(df):,}")

    df["text"] = df["content"].apply(clean_text)

    n_before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH].copy()
    print(f"Short reviews removed: {n_before - len(df):,}")

    n_before = len(df)
    df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()
    if (n_invalid := n_before - len(df)) > 0:
        print(f"Invalid ratings removed: {n_invalid:,}")

    print("\nCreating features...")

    df["sentiment"] = df["rating"].map(SENTIMENT_MAP)
    df["sentiment_label"] = df["sentiment"].map(SENTIMENT_LABELS)
    df["product_id"] = df["product_url"].apply(extract_product_id)
    df["text_length"] = df["text"].str.len()

    print(f"\nProcessed dataset: {len(df):,} reviews")
    print("\nRatings distribution:")
    for rating in sorted(df["rating"].unique()):
        count = (df["rating"] == rating).sum()
        pct = count / len(df) * 100
        print(f"  {rating}-star: {count:>7,} ({pct:5.1f}%)")

    print("\nSentiment distribution:")
    for sent_id in sorted(df["sentiment"].unique()):
        label = SENTIMENT_LABELS[sent_id]
        count = (df["sentiment"] == sent_id).sum()
        pct = count / len(df) * 100
        print(f"  {label:>10s}: {count:>7,} ({pct:5.1f}%)")

    print("\nText length statistics:")
    print(f"  Min/Max: {df['text_length'].min()} / {df['text_length'].max()}")
    print(f"  Median:  {df['text_length'].median():.0f}")
    print(f"  Mean:    {df['text_length'].mean():.0f}")

    print(f"\nUnique products: {df['product_id'].nunique():,}")

    print("\nContent Examples:")
    for sent_id in sorted(df["sentiment"].unique()):
        label = SENTIMENT_LABELS[sent_id]
        examples = df[df["sentiment"] == sent_id].sample(
            n=min(3, (df["sentiment"] == sent_id).sum()),
            random_state=RANDOM_SEED,
        )
        print(f"\n   [{label}]")
        for _, row in examples.iterrows():
            print(f"   {row['rating']}-star -> \"{row['text'][:100]}...\"")

    cols_to_save = ["text", "rating", "sentiment", "sentiment_label", "product_id", "date", "text_length"]
    df[cols_to_save].to_csv(PROCESSED_FILE, index=False, encoding="utf-8")

    size_mb = PROCESSED_FILE.stat().st_size / 1024 / 1024
    print(f"\nSaved to: {PROCESSED_FILE}")
    print(f"Size: {size_mb:.1f} MB, Records: {len(df):,}")


if __name__ == "__main__":
    main()
