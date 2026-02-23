"""
Step 6: Insights de mercado e comportamento do consumidor.
Análises avançadas sobre os dados de reviews.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_FILE, RESULTS_DIR, SENTIMENT_LABELS

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

INSIGHTS_DIR = RESULTS_DIR / "insights"
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"Negativo": "#e74c3c", "Neutro": "#f39c12", "Positivo": "#2ecc71"}


def analyze_text_length_vs_sentiment(df: pd.DataFrame):
    """Reviews negativas são mais longas? Consumidores reclamam mais detalhadamente?"""
    fig, ax = plt.subplots(figsize=(10, 5))
    stats = df.groupby("sentiment_label")["text_length"].agg(["mean", "median", "std"])
    stats = stats.reindex(["Negativo", "Neutro", "Positivo"])

    x = range(len(stats))
    bars = ax.bar(x, stats["mean"], yerr=stats["std"], capsize=5,
                  color=[COLORS[c] for c in stats.index], edgecolor="white", linewidth=1.5)

    # Adicionar mediana como ponto
    ax.scatter(x, stats["median"], color="black", zorder=5, s=50, label="Mediana")

    for i, (bar, mean, med) in enumerate(zip(bars, stats["mean"], stats["median"])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stats["std"].iloc[i] + 3,
                f"μ={mean:.0f}\nMd={med:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(stats.index)
    ax.set_ylabel("Texto (chars)")
    ax.set_title("Comprimento do texto por satisfação", fontsize=13, fontweight="bold")
    ax.legend()
    plt.savefig(INSIGHTS_DIR / "text_length_insight.png")
    plt.close()
    print("  Created text_length_insight.png")


def analyze_top_negative_keywords(df: pd.DataFrame, top_n: int = 25):
    """Quais palavras aparecem SOMENTE em reviews negativas (vs positivas)?"""
    stopwords = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
        "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
        "como", "mas", "foi", "ao", "das", "tem", "seu", "sua", "ou", "ser",
        "muito", "nos", "já", "está", "eu", "também", "só", "pelo", "pela",
        "até", "isso", "ela", "era", "sem", "mesmo", "ter", "meu", "minha",
        "nem", "esse", "essa", "você", "foi", "pro", "pra", "tudo", "bem",
    }

    def get_words(texts):
        words = []
        for text in texts:
            tokens = re.findall(r"\b[a-záàâãéêíóôõúüç]{3,}\b", str(text).lower())
            words.extend([w for w in tokens if w not in stopwords])
        return Counter(words)

    neg_words = get_words(df[df["sentiment_label"] == "Negativo"]["text"])
    pos_words = get_words(df[df["sentiment_label"] == "Positivo"]["text"])

    # Calcular razão neg/pos (palavras mais exclusivas de reviews negativas)
    exclusive_neg = {}
    for word, count in neg_words.most_common(200):
        pos_count = pos_words.get(word, 1)
        # Normalizar pelo tamanho de cada grupo
        n_neg = len(df[df["sentiment_label"] == "Negativo"])
        n_pos = len(df[df["sentiment_label"] == "Positivo"])
        ratio = (count / n_neg) / (pos_count / n_pos)
        if count >= 10:  # mínimo de ocorrências
            exclusive_neg[word] = ratio

    top = sorted(exclusive_neg.items(), key=lambda x: x[1], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    words, ratios = zip(*top)
    y_pos = range(len(words))
    bars = ax.barh(y_pos, ratios, color="#e74c3c", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Razão Negativo/Positivo")
    ax.set_title("Palavras em Reviews Negativas", fontsize=13, fontweight="bold")
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5, label="Neutro (1:1)")
    ax.legend()
    plt.savefig(INSIGHTS_DIR / "negative_keywords.png")
    plt.close()
    print("  Created negative_keywords.png")


def analyze_rating_evolution(df: pd.DataFrame):
    """Evolução do sentimento ao longo do tempo."""
    # Parsear datas
    month_map = {
        "jan": "01", "fev": "02", "mar": "03", "abr": "04",
        "mai": "05", "jun": "06", "jul": "07", "ago": "08",
        "set": "09", "out": "10", "nov": "11", "dez": "12",
    }

    def parse_date(d):
        if not isinstance(d, str):
            return None
        match = re.match(r"(\d{2})\s+(\w{3})\.\s+(\d{4})", d)
        if match:
            day, month_str, year = match.groups()
            month = month_map.get(month_str.lower())
            if month:
                return f"{year}-{month}"
        return None

    df_temp = df.copy()
    df_temp["year_month"] = df_temp["date"].apply(parse_date)
    df_temp = df_temp.dropna(subset=["year_month"])

    if len(df_temp) == 0:
        print("  ⚠ Não foi possível parsear datas — pulando análise temporal")
        return

    # Agrupar por mês
    monthly = df_temp.groupby(["year_month", "sentiment_label"]).size().unstack(fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Pegar últimos 24 meses
    monthly_pct = monthly_pct.sort_index().tail(24)

    if len(monthly_pct) < 3:
        print("  ⚠ Poucos meses de dados — pulando análise temporal")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    for label in ["Positivo", "Neutro", "Negativo"]:
        if label in monthly_pct.columns:
            ax.plot(range(len(monthly_pct)), monthly_pct[label],
                    marker="o", label=label, color=COLORS[label], linewidth=2)

    ax.set_xticks(range(len(monthly_pct)))
    ax.set_xticklabels(monthly_pct.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Evolução do Sentimento", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.savefig(INSIGHTS_DIR / "sentiment_evolution.png")
    plt.close()
    print("  Created sentiment_evolution.png")


def analyze_product_satisfaction(df: pd.DataFrame, min_reviews: int = 20, top_n: int = 15):
    """Produtos com maior e menor satisfação."""
    product_stats = df.groupby("product_id").agg(
        total_reviews=("rating", "count"),
        avg_rating=("rating", "mean"),
        pct_negative=("sentiment", lambda x: (x == 0).mean() * 100),
        pct_positive=("sentiment", lambda x: (x == 2).mean() * 100),
    ).reset_index()

    # Filtrar produtos com mínimo de reviews
    product_stats = product_stats[product_stats["total_reviews"] >= min_reviews]

    if len(product_stats) == 0:
        print(f"  ⚠ Nenhum produto com >= {min_reviews} reviews")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top piores
    worst = product_stats.nlargest(top_n, "pct_negative")
    bars = axes[0].barh(range(len(worst)), worst["pct_negative"],
                        color="#e74c3c", edgecolor="white")
    axes[0].set_yticks(range(len(worst)))
    axes[0].set_yticklabels([f"{pid[:15]}... ({n} rev)"
                             for pid, n in zip(worst["product_id"], worst["total_reviews"])],
                            fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("")
    axes[0].set_xlabel("% Negative")
    axes[0].set_title(f"Top {top_n} Mismatches", fontsize=12, fontweight="bold")

    # Top melhores
    best = product_stats.nlargest(top_n, "pct_positive")
    bars = axes[1].barh(range(len(best)), best["pct_positive"],
                        color="#2ecc71", edgecolor="white")
    axes[1].set_yticks(range(len(best)))
    axes[1].set_yticklabels([f"{pid[:15]}... ({n} rev)"
                             for pid, n in zip(best["product_id"], best["total_reviews"])],
                            fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("")
    axes[1].set_xlabel("% Positive")
    axes[1].set_title(f"Top {top_n} Satisfying", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(INSIGHTS_DIR / "product_satisfaction.png")
    plt.close()
    print("  Created product_satisfaction.png")


def generate_summary_stats(df: pd.DataFrame):
    """Resumo estatístico geral."""
    print(f"\n--- DATASET SUMMARY ---")
    print(f"  Total reviews: {len(df):,}")
    print(f"  Unique products: {df['product_id'].nunique():,}")
    print(f"  Average rating: {df['rating'].mean():.2f}")

    print(f"\n   Sentimento:")
    for label in ["Positivo", "Neutro", "Negativo"]:
        count = (df["sentiment_label"] == label).sum()
        pct = count / len(df) * 100
        print(f"   {label:>10s}: {count:>7,} ({pct:.1f}%)")

    print(f"\n   Texto:")
    print(f"   Comprimento médio: {df['text_length'].mean():.0f} chars")
    neg_len = df[df["sentiment_label"] == "Negativo"]["text_length"].mean()
    pos_len = df[df["sentiment_label"] == "Positivo"]["text_length"].mean()
    print(f"   Neg médio: {neg_len:.0f} chars vs Pos médio: {pos_len:.0f} chars")
    print(f"   → Reviews negativas são {neg_len/pos_len:.1f}x mais longas")


def main():
    print("\n--- STEP 6: MARKET INSIGHTS ---")

    if not PROCESSED_FILE.exists():
        print(f"Error: {PROCESSED_FILE} not found.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_FILE, encoding="utf-8")
    print(f"Loaded: {len(df):,} reviews\n")

    print("Generating analyses...")
    analyze_text_length_vs_sentiment(df)
    analyze_top_negative_keywords(df)
    analyze_rating_evolution(df)
    analyze_product_satisfaction(df)
    generate_summary_stats(df)

    print(f"\nInsights saved to: {INSIGHTS_DIR}/")


if __name__ == "__main__":
    main()
