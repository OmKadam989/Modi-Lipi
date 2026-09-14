import sys
from pathlib import Path
import uvicorn

# Automatically add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("Starting Modi Lipi OCR FastAPI Backend on http://127.0.0.1:8000...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
