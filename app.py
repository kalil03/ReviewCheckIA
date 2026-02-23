from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cpu"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MODEL_DIR = Path("models/bertimbau-sentiment/best")
SENTIMENT_LABELS = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model not found at {MODEL_DIR}.")

    print(f"Loading '{MODEL_NAME}' on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    yield

app = FastAPI(title="ReviewCheckIA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(req: AnalyzeRequest):
    inputs = tokenizer(
        req.text,
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
    
    label = SENTIMENT_LABELS[pred_id]
    emoji_map = {0: "😤", 1: "😐", 2: "😊"}
    
    return {
        "sentiment": label,
        "emoji": emoji_map.get(pred_id, "⚪"),
        "scores": {
            "neg": round(prob_list[0] * 100, 1),
            "neu": round(prob_list[1] * 100, 1),
            "pos": round(prob_list[2] * 100, 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
