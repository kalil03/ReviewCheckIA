"""
Configurações centralizadas do projeto de Análise de Sentimento.
Dataset: Reviews do Mercado Livre (PT-BR)
"""
from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "mercadolivre_com_br"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RESULTS_DIR / "eda", RESULTS_DIR / "insights"]:
    d.mkdir(parents=True, exist_ok=True)

RAW_FILES = sorted(RAW_DIR.glob("reviews_mercadolivre_com_br_*.json"))
PROCESSED_FILE = DATA_DIR / "processed.csv"
TOKENIZED_DIR = DATA_DIR / "tokenized"
SENTIMENT_MAP = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2,
}

SENTIMENT_LABELS = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
NUM_LABELS = 3

MIN_TEXT_LENGTH = 5

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

MODEL_OUTPUT_DIR = MODELS_DIR / "bertimbau-sentiment"
CLASS_WEIGHTS_FILE = DATA_DIR / "class_weights.json"
