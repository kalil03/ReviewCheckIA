"""
Step 5: Avaliação detalhada do modelo no test set.
Confusion matrix, classification report, análise de erros.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BATCH_SIZE, GPU_AVAILABLE, MODEL_OUTPUT_DIR, NUM_LABELS,
    RESULTS_DIR, SENTIMENT_LABELS, TOKENIZED_DIR,
)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                linewidths=0.5)
    axes[0].set_title("Confusion Matrix (Absolute)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                linewidths=0.5)
    axes[1].set_title("Confusion Matrix (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predito")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("\n--- STEP 5: MODEL EVALUATION ---")

    best_dir = MODEL_OUTPUT_DIR / "best"
    if not best_dir.exists():
        print(f"Error: Model not found at {best_dir}")
        sys.exit(1)

    print(f"Loading model from {best_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))

    if not TOKENIZED_DIR.exists():
        print(f"Error: Dataset not found at {TOKENIZED_DIR}")
        sys.exit(1)

    dataset = load_from_disk(str(TOKENIZED_DIR))
    test_dataset = dataset["test"]
    print(f"Test set: {len(test_dataset):,} samples")

    print(f"\nEvaluating...")
    trainer = Trainer(
        model=model,
        args=__import__("transformers").TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=BATCH_SIZE * 2,
            report_to="none",
            fp16=GPU_AVAILABLE,
            dataloader_pin_memory=False,
        ),
    )

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    label_names = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]

    report = classification_report(y_true, y_pred, target_names=label_names)
    print(f"\nClassification Report:")
    print(report)

    report_file = RESULTS_DIR / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"   Saved to: {report_file}")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "total_samples": int(len(y_true)),
    }

    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(label_names):
        metrics[f"f1_{label.lower()}"] = float(f1_per_class[i])

    metrics_file = RESULTS_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nOverall Metrics:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"   F1 Weighted: {metrics['f1_weighted']:.4f}")
    for i, label in enumerate(label_names):
        print(f"   F1 {label:>10s}: {f1_per_class[i]:.4f}")
    print(f"   Saved to: {metrics_file}")

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, label_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

    print(f"\nError Analysis:")

    # Carregar textos originais do processed.csv pra mostrar exemplos
    import pandas as pd
    from config import PROCESSED_FILE
    if PROCESSED_FILE.exists():
        df_full = pd.read_csv(PROCESSED_FILE, encoding="utf-8")
        # Pegar os textos correspondentes ao test set (mesma seed)
        from sklearn.model_selection import train_test_split
        from config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

        _, temp = train_test_split(
            df_full, test_size=(VAL_RATIO + TEST_RATIO),
            stratify=df_full["sentiment"], random_state=RANDOM_SEED,
        )
        _, test_df = train_test_split(
            temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=temp["sentiment"], random_state=RANDOM_SEED,
        )
        test_df = test_df.reset_index(drop=True)

        errors = np.where(y_pred != y_true)[0]
        print(f"   Total errors: {len(errors):,} / {len(y_true):,} ({len(errors)/len(y_true)*100:.1f}%)")

        if len(errors) > 0 and len(test_df) == len(y_true):
            print(f"\n   Error Examples:")
            for idx in errors[:10]:
                real = SENTIMENT_LABELS[y_true[idx]]
                pred = SENTIMENT_LABELS[y_pred[idx]]
                text = test_df.iloc[idx]["text"][:120]
                print(f"   True: {real:>10s} | Pred: {pred:>10s} | \"{text}...\"")

    print(f"\nEvaluation completed. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
