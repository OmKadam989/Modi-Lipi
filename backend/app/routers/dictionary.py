from fastapi import APIRouter, Query
from typing import List
from app.ocr.dictionary_corrector import DictionaryCorrector
from app.models.schemas import DictionarySuggestion

router = APIRouter(prefix="/api/dictionary", tags=["Dictionary"])

corrector = DictionaryCorrector()

@router.get("/suggest", response_model=List[DictionarySuggestion])
def get_dictionary_suggestions(
    word: str = Query(..., description="Target word to find post-correction suggestions for"),
    max_distance: int = Query(2, description="Maximum Levenshtein edit distance")
):
    suggestions = corrector.find_suggestions(word, max_distance=max_distance)
    return [DictionarySuggestion(**s) for s in suggestions]
