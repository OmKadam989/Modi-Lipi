import json
import numpy as np
from pathlib import Path
from app.config import MODEL_FILE, LABEL_ENCODER_FILE
from app.ocr.devanagari_mapper import DevanagariMapper

class CNNCharacterRecognizer:
    def __init__(self):
        self.model = None
        self.label_map = {}
        self.inv_label_map = {}
        self.is_model_loaded = False
        self.load_model()

    def load_model(self):
        """Attempts to load the trained CNN model and class label mapping file."""
        if MODEL_FILE.exists() and LABEL_ENCODER_FILE.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(MODEL_FILE))
                with open(LABEL_ENCODER_FILE, "r", encoding="utf-8") as f:
                    self.label_map = json.load(f)
                
                # Inverse label map: int_index -> character string
                self.inv_label_map = {v: k for k, v in self.label_map.items()}
                self.is_model_loaded = True
                print(f"CNN Recognizer: Successfully loaded model from {MODEL_FILE}")
            except Exception as e:
                print(f"CNN Recognizer: Error loading model: {e}")
                self.is_model_loaded = False
        else:
            print("CNN Recognizer: Model file or label encoder not found. Running in model-unavailable mode.")
            self.is_model_loaded = False

    def predict_character(self, char_input: np.ndarray, top_k: int = 3) -> dict:
        """
        Receives normalized (32, 32, 1) float character image matrix.
        Returns predicted Modi character, Devanagari mapped character, confidence float, and top-k alternatives.
        """
        if not self.is_model_loaded or self.model is None:
            return {
                "model_available": False,
                "predicted_modi_char": "?",
                "predicted_devanagari_char": "?",
                "confidence": 0.0,
                "alternatives": []
            }

        # Add batch dimension: (1, 32, 32, 1)
        if len(char_input.shape) == 3:
            batch_input = np.expand_dims(char_input, axis=0)
        else:
            batch_input = char_input

        probs = self.model.predict(batch_input, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_predictions = []
        for idx in top_indices:
            idx_int = int(idx)
            modi_char = self.inv_label_map.get(idx_int, "?")
            devanagari_char = DevanagariMapper.map_char(modi_char)
            conf = float(probs[idx_int])
            top_predictions.append({
                "modi_char": modi_char,
                "devanagari_char": devanagari_char,
                "confidence": round(conf, 4)
            })

        best_pred = top_predictions[0]
        return {
            "model_available": True,
            "predicted_modi_char": best_pred["modi_char"],
            "predicted_devanagari_char": best_pred["devanagari_char"],
            "confidence": best_pred["confidence"],
            "alternatives": top_predictions[1:]
        }
