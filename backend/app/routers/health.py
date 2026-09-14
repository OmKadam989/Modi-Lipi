from fastapi import APIRouter
from app.ocr.cnn_recognizer import CNNCharacterRecognizer

router = APIRouter(prefix="/api", tags=["Health"])

cnn_engine = CNNCharacterRecognizer()

@router.get("/health")
def health_check():
    cnn_engine.load_model()
    return {
        "status": "online",
        "service": "Modi Lipi OCR Engine",
        "model_loaded": cnn_engine.is_model_loaded,
        "database": "sqlite",
        "version": "1.0.0"
    }
