import cv2
import numpy as np
from pathlib import Path

out_dir = Path("datasets/sample_documents")
out_dir.mkdir(parents=True, exist_ok=True)

def draw_modi_sample_document(filename: str):
    # Create parchment-colored / aged document canvas
    w, h = 800, 1000
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    
    # Add subtle parchment texture gradient & noise
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 210, 255).astype(np.uint8)

    # Draw heading / border rules
    cv2.line(img, (50, 60), (w - 50, 60), (40, 30, 20), 2)
    cv2.line(img, (50, 68), (w - 50, 68), (40, 30, 20), 1)

    # Draw lines of Modi Lipi characters and word groups
    # Modi characters representation glyph sequences
    modi_lines = [
        ["𑘀", "𑘎", "𑘰", "𑘐", "𑘟", " ", "𑘢", "𑘝", "𑘿", "𑘨", " ", "𑘥", "𑘰", "𑘐", "𑘱"],
        ["𑘀", "𑘕", "𑘿", "𑘗", "𑘰", " ", "𑘢", "𑘝", "𑘿", "𑘨", " ", "𑘭", "𑘡", "𑘟", " "],
        ["𑘬", "𑘮", "𑘰", "𑘕", "𑘲", " ", "𑘨", "𑘰", "𑘕", "𑘹", " ", "𑘭", "𑘨", "𑘎", "𑘰", "𑘨"],
        ["𑘦", "𑘳", "𑘎", "𑘰", "𑘦", " ", "𑘢", "𑘹", "𑘉", "𑘪", "𑘹", " ", "𑘡", "𑘱", "𑘪", "𑘰", "𑘚", "𑘰"],
        ["𑘨", "𑘧", "𑘝", "𑘹", "𑘓", "𑘹", " ", "𑘮", "𑘳", "𑘎", "𑘴", "𑘦", " ", "𑘐", "𑘰", "𑘪"]
    ]

    y_offset = 130
    for line in modi_lines:
        x_offset = 70
        for char in line:
            if char == " ":
                x_offset += 25
                continue
            
            # Draw realistic cursive character loops and strokes
            cx, cy = x_offset + 12, y_offset + 15
            cv2.circle(img, (cx, cy), 8, (20, 20, 20), 2)
            cv2.line(img, (cx - 10, cy - 8), (cx + 10, cy - 8), (20, 20, 20), 2)
            cv2.line(img, (cx, cy), (cx + 8, cy + 12), (20, 20, 20), 2)
            x_offset += 32
        
        y_offset += 70

    # Draw bottom seal circle
    cv2.circle(img, (w // 2, y_offset + 80), 45, (60, 30, 20), 3)

    cv2.imwrite(str(out_dir / filename), img)
    print(f"Sample document saved to {out_dir / filename}")

draw_modi_sample_document("sample_modi_doc1.png")
draw_modi_sample_document("sample_modi_doc2.jpg")
