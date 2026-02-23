"""
Step 4: Treinamento do modelo BERTimbau para análise de sentimento.
Fine-tuning com HuggingFace Trainer + class weights para dataset desbalanceado.
3 classes: Negativo (0), Neutro (1), Positivo (2).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BATCH_SIZE, CLASS_WEIGHTS_FILE, DEVICE, GPU_AVAILABLE, GPU_NAME,
    LEARNING_RATE, MODEL_NAME, MODEL_OUTPUT_DIR, NUM_EPOCHS, NUM_LABELS,
    RANDOM_SEED, SENTIMENT_LABELS, TOKENIZED_DIR, WARMUP_RATIO, WEIGHT_DECAY,
)


def load_class_weights() -> torch.Tensor:
    if not CLASS_WEIGHTS_FILE.exists():
        print(f"Warning: {CLASS_WEIGHTS_FILE} not found, using equal weights")
        return torch.ones(NUM_LABELS)

    with open(CLASS_WEIGHTS_FILE) as f:
        weights = json.load(f)

    w = torch.tensor([weights[str(i)] for i in range(NUM_LABELS)], dtype=torch.float32)
    return w


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Treinamento do modelo")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    print("\n--- STEP 4: MODEL TRAINING ---")

    if GPU_AVAILABLE:
        print(f"GPU detected: {GPU_NAME}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Warning: GPU not detected, running on CPU")

    if not TOKENIZED_DIR.exists():
        print(f"Error: Dataset not found in {TOKENIZED_DIR}")
        sys.exit(1)

    print(f"Loading dataset from {TOKENIZED_DIR}")
    dataset = load_from_disk(str(TOKENIZED_DIR))

    if args.max_samples:
        for split in dataset:
            if len(dataset[split]) > args.max_samples:
                dataset[split] = dataset[split].select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples per split")

    for split in dataset:
        print(f"   {split:>12s}: {len(dataset[split]):,} samples")

    if not args.no_class_weights:
        class_weights = load_class_weights()
        print(f"\nClass weights:")
        for i, label in SENTIMENT_LABELS.items():
            print(f"   {label:>10s} ({i}): {class_weights[i]:.4f}")
    else:
        class_weights = None

    print(f"\nLoading: {MODEL_NAME}")
    id2label = {i: v for i, v in SENTIMENT_LABELS.items()}
    label2id = {v: i for i, v in SENTIMENT_LABELS.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} (trainable: {trainable:,})")

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        logging_steps=50,
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to="none",
        fp16=GPU_AVAILABLE,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if class_weights is not None:
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    print(f"\nTraining ({args.epochs} epochs)...")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Mixed precision: {GPU_AVAILABLE}")

    train_result = trainer.train()

    best_dir = MODEL_OUTPUT_DIR / "best"
    trainer.save_model(str(best_dir))
    print(f"\nModel saved to: {best_dir}")

    print(f"\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

    print(f"\nValidation metrics:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")


if __name__ == "__main__":
    main()
