from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CharacterAlternative(BaseModel):
    modi_char: str
    devanagari_char: str
    confidence: float

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class CharacterResult(BaseModel):
    id: Optional[int] = None
    char_index: int
    bounding_box: BoundingBox
    predicted_modi_char: str
    predicted_devanagari_char: str
    confidence: float
    alternatives: List[CharacterAlternative] = []
    user_corrected_char: Optional[str] = None
    crop_image_url: Optional[str] = None

class WordResult(BaseModel):
    id: Optional[int] = None
    word_index: int
    bounding_box: BoundingBox
    raw_modi_word: str
    corrected_modi_word: str
    devanagari_word: str
    characters: List[CharacterResult] = []

class LineResult(BaseModel):
    id: Optional[int] = None
    line_index: int
    bounding_box: BoundingBox
    words: List[WordResult] = []

class PageResult(BaseModel):
    page_number: int
    lines: List[LineResult] = []

class OCRStatistics(BaseModel):
    pages: int
    lines: int
    words: int
    characters: int
    average_confidence: float
    processing_time_ms: float
    model_status: str

class DocumentOCRResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    metadata: Dict[str, Any]
    preprocessed_images: Dict[str, str]  # base64 data URLs for stages (original, grayscale, binarized, deskewed)
    pages: List[PageResult] = []
    modi_text: str
    corrected_text: str
    devanagari_text: str
    statistics: OCRStatistics

class ManualCorrectionRequest(BaseModel):
    document_id: str
    character_id: Optional[int] = None
    word_id: Optional[int] = None
    corrected_char: Optional[str] = None
    corrected_word: Optional[str] = None

class DictionarySuggestion(BaseModel):
    suggested_word: str
    devanagari_word: str
    distance: int
    confidence_score: float

class ModelEvaluationResponse(BaseModel):
    status: str
    model_loaded: bool
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_classes: int
    class_names: List[str]
    confusion_matrix: List[List[int]] = []
    training_accuracy_history: List[float] = []
    training_loss_history: List[float] = []
