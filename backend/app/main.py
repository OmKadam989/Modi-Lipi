import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.routers import health, metrics, dictionary, export, ocr

app = FastAPI(
    title="Modi Lipi OCR & Digitization API Framework",
    description="A CNN-Based Deep Learning Framework for OCR and Digitization of Historical Modi Lipi Documents",
    version="1.0.0"
)

# CORS configuration to allow local React frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

# Include routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(dictionary.router)
app.include_router(export.router)
app.include_router(ocr.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
