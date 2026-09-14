import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASETS_DIR = BASE_DIR / "datasets"
EVALUATION_DIR = BASE_DIR / "evaluation"

DB_PATH = BASE_DIR / "modi_ocr.db"

MODEL_FILE = MODELS_DIR / "trained_modi_cnn.keras"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.json"
EVALUATION_FILE = EVALUATION_DIR / "evaluation_metrics.json"

# Ensure required directories exist
for folder in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, DATASETS_DIR, EVALUATION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
