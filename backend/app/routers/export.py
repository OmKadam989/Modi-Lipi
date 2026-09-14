from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
import io
from app.database import get_db_connection
from app.utils.exporter import Exporter

router = APIRouter(prefix="/api/export", tags=["Export"])

@router.get("/{doc_id}")
def export_ocr_result(doc_id: str, format: str = Query("txt", description="Export format: txt | json | pdf")):
    conn = get_db_connection()
    cursor = conn.cursor()

    doc = cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document ID not found.")

    doc_dict = dict(doc)
    doc_data = {
        "document_id": doc_dict["id"],
        "filename": doc_dict["filename"],
        "modi_text": doc_dict.get("raw_modi_text", ""),
        "corrected_text": doc_dict.get("corrected_modi_text", ""),
        "devanagari_text": doc_dict.get("devanagari_text", ""),
        "statistics": {
            "lines": 0,
            "words": 0,
            "characters": 0,
            "average_confidence": 0.95,
            "processing_time_ms": doc_dict.get("processing_time_ms", 0.0),
            "model_status": "Loaded"
        }
    }
    conn.close()

    fmt = format.lower()

    if fmt == "txt":
        content = Exporter.generate_txt(doc_data)
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="modi_ocr_{doc_id}.txt"'}
        )
    elif fmt == "json":
        content = Exporter.generate_json(doc_data)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="modi_ocr_{doc_id}.json"'}
        )
    elif fmt == "pdf":
        pdf_bytes = Exporter.generate_pdf(doc_data)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="modi_ocr_{doc_id}.pdf"'}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use txt, json, or pdf.")
