import time
import uuid
import json
import sqlite3
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List, Optional

from app.config import UPLOADS_DIR
from app.database import get_db_connection
from app.ocr.preprocessing import Preprocessor
from app.ocr.segmentation import Segmenter
from app.ocr.cnn_recognizer import CNNCharacterRecognizer
from app.ocr.devanagari_mapper import DevanagariMapper
from app.ocr.reconstruction import TextReconstructor
from app.utils.pdf_utils import process_pdf_bytes
from app.models.schemas import DocumentOCRResponse, ManualCorrectionRequest

router = APIRouter(prefix="/api/ocr", tags=["OCR Processing"])

preprocessor = Preprocessor()
segmenter = Segmenter()
cnn_recognizer = CNNCharacterRecognizer()

@router.post("/upload", response_model=DocumentOCRResponse)
async def upload_and_process_document(file: UploadFile = File(...)):
    start_time = time.time()

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPG, JPEG, PNG, or PDF.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save original file
    doc_id = str(uuid.uuid4())[:12]
    saved_filename = f"{doc_id}_{file.filename}"
    saved_filepath = UPLOADS_DIR / saved_filename
    with open(saved_filepath, "wb") as f:
        f.write(contents)

    # Convert PDF or single Image
    if ext == ".pdf":
        page_bytes_list = process_pdf_bytes(contents)
    else:
        page_bytes_list = [contents]

    pages_result = []
    total_lines = 0
    total_words = 0
    total_chars = 0
    confidence_sum = 0.0
    
    first_page_preprocessed = {}

    cnn_recognizer.load_model()
    model_status_str = "Loaded" if cnn_recognizer.is_model_loaded else "Unavailable"

    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert Document record
    cursor.execute("""
    INSERT INTO documents (id, filename, filepath, status, page_count)
    VALUES (?, ?, ?, ?, ?)
    """, (doc_id, file.filename, str(saved_filepath), "processing", len(page_bytes_list)))
    conn.commit()

    for page_idx, pbytes in enumerate(page_bytes_list):
        # 1. Preprocessing Pipeline
        prep_out = preprocessor.run_pipeline(pbytes)
        if page_idx == 0:
            first_page_preprocessed = {
                "original_b64": prep_out["original_b64"],
                "grayscale_b64": prep_out["grayscale_b64"],
                "binarized_b64": prep_out["binarized_b64"],
                "deskewed_b64": prep_out["deskewed_b64"],
            }

        # 2. Contour Segmentation (Document -> Lines -> Words -> Characters)
        seg_lines = segmenter.segment_document(prep_out["raw_gray"], prep_out["raw_binary"])

        page_lines_structured = []

        for line_data in seg_lines:
            total_lines += 1
            l_idx = line_data["line_index"]
            l_bbox = line_data["bbox"]

            cursor.execute("""
            INSERT INTO lines (document_id, line_index, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, l_idx, l_bbox["x"], l_bbox["y"], l_bbox["w"], l_bbox["h"]))
            line_db_id = cursor.lastrowid

            line_words_structured = []

            for word_data in line_data["words"]:
                total_words += 1
                w_idx = word_data["word_index"]
                w_bbox = word_data["bbox"]

                cursor.execute("""
                INSERT INTO words (line_id, word_index, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (line_db_id, w_idx, w_bbox["x"], w_bbox["y"], w_bbox["w"], w_bbox["h"]))
                word_db_id = cursor.lastrowid

                word_chars_structured = []
                word_raw_modi = []
                word_devanagari = []

                for char_data in word_data["characters"]:
                    total_chars += 1
                    c_idx = char_data["char_index"]
                    c_bbox = char_data["bbox"]
                    crop_input = char_data["char_crop_input"]

                    # 3. CNN Inference
                    pred_res = cnn_recognizer.predict_character(crop_input, top_k=3)
                    pred_modi = pred_res["predicted_modi_char"]
                    pred_dev = pred_res["predicted_devanagari_char"]
                    conf = pred_res["confidence"]

                    confidence_sum += conf
                    word_raw_modi.append(pred_modi)
                    word_devanagari.append(pred_dev)

                    # Save character prediction to DB
                    cursor.execute("""
                    INSERT INTO characters (word_id, char_index, bbox_x, bbox_y, bbox_w, bbox_h,
                                            predicted_modi_char, predicted_devanagari_char, confidence, top_k_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (word_db_id, c_idx, c_bbox["x"], c_bbox["y"], c_bbox["w"], c_bbox["h"],
                          pred_modi, pred_dev, conf, json.dumps(pred_res["alternatives"], ensure_ascii=False)))
                    char_db_id = cursor.lastrowid

                    word_chars_structured.append({
                        "id": char_db_id,
                        "char_index": c_idx,
                        "bounding_box": c_bbox,
                        "predicted_modi_char": pred_modi,
                        "predicted_devanagari_char": pred_dev,
                        "confidence": conf,
                        "alternatives": pred_res["alternatives"],
                        "user_corrected_char": None
                    })

                raw_w_str = "".join(word_raw_modi)
                dev_w_str = "".join(word_devanagari)

                cursor.execute("""
                UPDATE words SET raw_modi_word = ?, corrected_modi_word = ?, devanagari_word = ?
                WHERE id = ?
                """, (raw_w_str, raw_w_str, dev_w_str, word_db_id))

                line_words_structured.append({
                    "id": word_db_id,
                    "word_index": w_idx,
                    "bounding_box": w_bbox,
                    "raw_modi_word": raw_w_str,
                    "corrected_modi_word": raw_w_str,
                    "devanagari_word": dev_w_str,
                    "characters": word_chars_structured
                })

            page_lines_structured.append({
                "id": line_db_id,
                "line_index": l_idx,
                "bounding_box": l_bbox,
                "words": line_words_structured
            })

        pages_result.append({
            "page_number": page_idx + 1,
            "lines": page_lines_structured
        })

    # 4. Text Reconstruction
    raw_modi_text, corrected_modi_text, devanagari_text = TextReconstructor.reconstruct_document_text(
        pages_result[0]["lines"] if pages_result else []
    )

    elapsed_ms = (time.time() - start_time) * 1000.0
    avg_confidence = round(confidence_sum / total_chars, 4) if total_chars > 0 else 0.0

    # Update Document record in DB
    cursor.execute("""
    UPDATE documents
    SET status = 'completed', processing_time_ms = ?, raw_modi_text = ?, corrected_modi_text = ?, devanagari_text = ?
    WHERE id = ?
    """, (elapsed_ms, raw_modi_text, corrected_modi_text, devanagari_text, doc_id))
    conn.commit()
    conn.close()

    statistics_obj = {
        "pages": len(page_bytes_list),
        "lines": total_lines,
        "words": total_words,
        "characters": total_chars,
        "average_confidence": avg_confidence,
        "processing_time_ms": round(elapsed_ms, 2),
        "model_status": model_status_str
    }

    return DocumentOCRResponse(
        document_id=doc_id,
        filename=file.filename,
        status="completed",
        metadata={"uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        preprocessed_images=first_page_preprocessed,
        pages=pages_result,
        modi_text=raw_modi_text,
        corrected_text=corrected_modi_text,
        devanagari_text=devanagari_text,
        statistics=statistics_obj
    )

@router.post("/correct")
def apply_manual_correction(req: ManualCorrectionRequest = Body(...)):
    """Applies user manual correction for a character or word and re-evaluates outputs."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if req.character_id and req.corrected_char is not None:
        new_dev = DevanagariMapper.map_char(req.corrected_char)
        cursor.execute("""
        UPDATE characters
        SET user_corrected_char = ?, predicted_devanagari_char = ?
        WHERE id = ?
        """, (req.corrected_char, new_dev, req.character_id))
        conn.commit()

    conn.close()
    return {"status": "success", "message": "Correction applied successfully."}

@router.get("/result/{doc_id}")
def get_ocr_result(doc_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    doc = cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found.")

    conn.close()
    return dict(doc)
