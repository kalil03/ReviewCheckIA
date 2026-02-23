"""
Step 2: Análise Exploratória dos Dados (EDA).
Gera visualizações sobre distribuição de sentimentos, texto e produtos.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_FILE, RESULTS_DIR, SENTIMENT_LABELS, RANDOM_SEED

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

EDA_DIR = RESULTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def plot_rating_distribution(df: pd.DataFrame):
    """Distribuição de ratings (1-5)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    counts = df["rating"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xlabel("Rating (estrelas)", fontsize=12)
    ax.set_ylabel("Quantidade", fontsize=12)
    ax.set_title("Distribuição de Ratings", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    
    plt.savefig(EDA_DIR / "rating_distribution.png")
    plt.close()
    print("  Created rating_distribution.png")


def plot_sentiment_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Negativo": "#e74c3c", "Neutro": "#f39c12", "Positivo": "#2ecc71"}
    counts = df["sentiment_label"].value_counts().reindex(["Negativo", "Neutro", "Positivo"])
    bars = axes[0].bar(counts.index, counts.values,
                       color=[colors[c] for c in counts.index],
                       edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                     f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Distribuição de Sentimento", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Quantidade")

    axes[1].pie(counts.values, labels=counts.index,
                colors=[colors[c] for c in counts.index],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Proporção de Sentimento", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "sentiment_distribution.png")
    plt.close()
    print("  Created sentiment_distribution.png")


def plot_text_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Negativo": "#e74c3c", "Neutro": "#f39c12", "Positivo": "#2ecc71"}

    for label in ["Negativo", "Neutro", "Positivo"]:
        subset = df[df["sentiment_label"] == label]["text_length"]
        axes[0].hist(subset.clip(upper=500), bins=50, alpha=0.6,
                     label=label, color=colors[label])
    axes[0].set_xlabel("Comprimento do texto (chars)")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição do Comprimento de Texto", fontsize=13, fontweight="bold")
    axes[0].legend()

    order = ["Negativo", "Neutro", "Positivo"]
    palette = [colors[c] for c in order]
    sns.boxplot(data=df[df["text_length"] <= 500], x="sentiment_label", y="text_length",
                order=order, palette=palette, ax=axes[1])
    axes[1].set_xlabel("Sentimento")
    axes[1].set_ylabel("Comprimento (chars)")
    axes[1].set_title("Comprimento por Sentimento", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "text_length_distribution.png")
    plt.close()
    print("  Created text_length_distribution.png")


def plot_top_words(df: pd.DataFrame, top_n: int = 20):
    """Top palavras mais frequentes por sentimento."""
    import re as _re
    stopwords_pt = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com",
        "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
        "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
        "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo",
        "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
        "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
        "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha",
        "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós",
        "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse",
        "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu",
        "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela",
        "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles",
        "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão",
        "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos",
        "estavam", "um", "uns", "mas", "que", "com", "pro", "pra",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sentiments = ["Negativo", "Neutro", "Positivo"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for ax, sent, color in zip(axes, sentiments, colors):
        texts = df[df["sentiment_label"] == sent]["text"].str.lower()
        words = []
        for text in texts:
            tokens = _re.findall(r"\b[a-záàâãéêíóôõúüç]{3,}\b", text)
            words.extend([w for w in tokens if w not in stopwords_pt])
        top = Counter(words).most_common(top_n)
        if top:
            words_list, counts_list = zip(*top)
            y_pos = range(len(words_list))
            ax.barh(y_pos, counts_list, color=color, edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words_list)
            ax.invert_yaxis()
        ax.set_title(f"Top {top_n} — {sent}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequência")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "top_words_by_sentiment.png")
    plt.close()
    print("  Created top_words_by_sentiment.png")


def plot_reviews_per_product(df: pd.DataFrame, top_n: int = 15):
    """Top produtos mais avaliados."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products = df["product_id"].value_counts().head(top_n)
    bars = ax.barh(range(len(top_products)), top_products.values,
                   color=sns.color_palette("viridis", top_n), edgecolor="white")
    ax.set_yticks(range(len(top_products)))
    ax.set_yticklabels(top_products.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Número de Reviews")
    ax.set_title(f"Top {top_n} Produtos Mais Avaliados", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, top_products.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    plt.savefig(EDA_DIR / "top_products.png")
    plt.close()
    print("  Created top_products.png")


def plot_sentiment_by_rating(df: pd.DataFrame):
    """Heatmap: sentimento vs rating."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cross = pd.crosstab(df["rating"], df["sentiment_label"],
                        normalize="index")[["Negativo", "Neutro", "Positivo"]]
    sns.heatmap(cross, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Proporção"})
    ax.set_title("Sentimento por Rating", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rating")
    ax.set_xlabel("Sentimento")
    plt.savefig(EDA_DIR / "sentiment_by_rating.png")
    plt.close()
    print("  Created sentiment_by_rating.png")


def main():
    print("\n--- STEP 2: EDA ---")

    if not PROCESSED_FILE.exists():
        print(f"Error: {PROCESSED_FILE} not found.")
        print("Run 01_preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_FILE, encoding="utf-8")
    print(f"Loaded: {len(df):,} reviews\n")

    print("Generating charts...")
    plot_rating_distribution(df)
    plot_sentiment_distribution(df)
    plot_text_length(df)
    plot_top_words(df)
    plot_reviews_per_product(df)
    plot_sentiment_by_rating(df)

    print(f"\nCharts saved to {EDA_DIR}/")


if __name__ == "__main__":
    main()
