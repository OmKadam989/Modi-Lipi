import io
from PIL import Image

def process_pdf_bytes(pdf_bytes: bytes) -> list[bytes]:
    """
    Extracts pages from PDF file bytes as JPEG image bytes list.
    """
    image_bytes_list = []
    try:
        # Try importing fitz (PyMuPDF) if available
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes_list.append(buf.getvalue())
        return image_bytes_list
    except Exception as e:
        # Fallback: PIL image open if pdf image or single frame
        try:
            img = Image.open(io.BytesIO(pdf_bytes))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return [buf.getvalue()]
        except Exception as inner_e:
            raise ValueError(f"Unable to parse PDF document: {e} | Fallback error: {inner_e}")
