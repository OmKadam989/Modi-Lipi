import json
from fastapi import APIRouter
from app.config import EVALUATION_FILE, MODEL_FILE
from app.models.schemas import ModelEvaluationResponse

router = APIRouter(prefix="/api", tags=["Metrics"])

@router.get("/metrics", response_model=ModelEvaluationResponse)
def get_model_metrics():
    if not EVALUATION_FILE.exists() or not MODEL_FILE.exists():
        return ModelEvaluationResponse(
            status="unavailable",
            model_loaded=False,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_classes=0,
            class_names=[],
            confusion_matrix=[],
            training_accuracy_history=[],
            training_loss_history=[]
        )

    try:
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelEvaluationResponse(**data)
    except Exception as e:
        return ModelEvaluationResponse(
            status="error",
            model_loaded=False,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_classes=0,
            class_names=[],
            confusion_matrix=[],
            training_accuracy_history=[],
            training_loss_history=[]
        )
