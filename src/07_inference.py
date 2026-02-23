"""
Step 7: Script de Inferência para textos livres.
Carrega o modelo treinado e faz predições de sentimento com probabilidades.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEVICE, MODEL_OUTPUT_DIR, SENTIMENT_LABELS


def load_model_and_tokenizer():
    """Carrega o modelo e o tokenizer do diretório 'best'."""
    model_path = MODEL_OUTPUT_DIR / "best"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run main.py first.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    from config import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer


def predict(text, model, tokenizer):
    """Realiza a predição de um texto."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        
    prob_list = probs[0].tolist()
    pred_id = torch.argmax(probs, dim=-1).item()
    
    return pred_id, prob_list


def main():
    parser = argparse.ArgumentParser(description="Inferência de Sentimento com BERTimbau")
    parser.add_argument("--text", type=str, help="Texto para analisar")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if not args.text:
        print("\nTip: Use --text to pass a text via CLI.")
        examples = [
            "Produto excelente, superou minhas expectativas!",
            "O produto é bom, mas a entrega atrasou muito.",
            "Não gostei. Chegou quebrado e a caixa estava amassada.",
            "Achei mediano. Cumpre o que promete, nada demais.",
            "Péssima experiência, não recomendo a ninguém."
        ]
        texts_to_process = examples
    else:
        texts_to_process = [args.text]

    print("\n" + "-" * 80)
    print(f"{'TEXT':<50} | {'SENTIMENT':<12} | {'CONFIDENCE'}")
    print("-" * 80)

    for text in texts_to_process:
        pred_id, probs = predict(text, model, tokenizer)
        label = SENTIMENT_LABELS[pred_id]
        conf = probs[pred_id]
        
        display_text = (text[:47] + "..") if len(text) > 47 else text
        print(f"{display_text:<50} | {label:<12} | {conf:.2%}")

    print("-" * 80)


if __name__ == "__main__":
    main()
