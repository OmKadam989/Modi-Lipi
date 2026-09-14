import sqlite3
import json
from pathlib import Path
from app.config import DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL,
        page_count INTEGER DEFAULT 1,
        processing_time_ms REAL DEFAULT 0.0,
        raw_modi_text TEXT DEFAULT '',
        corrected_modi_text TEXT DEFAULT '',
        devanagari_text TEXT DEFAULT ''
    )
    """)

    # Lines table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        line_index INTEGER NOT NULL,
        bbox_x INTEGER NOT NULL,
        bbox_y INTEGER NOT NULL,
        bbox_w INTEGER NOT NULL,
        bbox_h INTEGER NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    )
    """)

    # Words table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id INTEGER NOT NULL,
        word_index INTEGER NOT NULL,
        bbox_x INTEGER NOT NULL,
        bbox_y INTEGER NOT NULL,
        bbox_w INTEGER NOT NULL,
        bbox_h INTEGER NOT NULL,
        raw_modi_word TEXT DEFAULT '',
        corrected_modi_word TEXT DEFAULT '',
        devanagari_word TEXT DEFAULT '',
        FOREIGN KEY (line_id) REFERENCES lines(id) ON DELETE CASCADE
    )
    """)

    # Characters table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS characters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word_id INTEGER NOT NULL,
        char_index INTEGER NOT NULL,
        bbox_x INTEGER NOT NULL,
        bbox_y INTEGER NOT NULL,
        bbox_w INTEGER NOT NULL,
        bbox_h INTEGER NOT NULL,
        predicted_modi_char TEXT DEFAULT '',
        predicted_devanagari_char TEXT DEFAULT '',
        confidence REAL DEFAULT 0.0,
        top_k_json TEXT DEFAULT '[]',
        user_corrected_char TEXT DEFAULT NULL,
        crop_image_path TEXT DEFAULT '',
        FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
    )
    """)

    # Model evaluation metrics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        epochs_trained INTEGER,
        confusion_matrix_json TEXT
    )
    """)

    conn.commit()
    conn.close()
