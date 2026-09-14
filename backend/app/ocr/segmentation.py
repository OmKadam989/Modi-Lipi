import cv2
import numpy as np

class Segmenter:
    def __init__(self, target_char_size: tuple[int, int] = (32, 32)):
        self.target_char_size = target_char_size

    def pad_and_resize_char(self, char_crop: np.ndarray) -> np.ndarray:
        """Pads character crop image to square aspect ratio and resizes to target size (32x32)."""
        h, w = char_crop.shape[:2]
        if h == 0 or w == 0:
            return np.zeros(self.target_char_size, dtype=np.float32)

        # Make square by padding equal borders
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        squared = cv2.copyMakeBorder(
            char_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        resized = cv2.resize(squared, self.target_char_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)  # (32, 32, 1)

    def segment_document(self, gray_img: np.ndarray, binary_img: np.ndarray):
        """
        Hierarchical 3-level contour segmentation:
        Document -> Lines -> Words -> Characters
        Returns structured list of detected bounding boxes and character crops.
        """
        img_h, img_w = binary_img.shape[:2]

        # 1. LINE DETECTION
        # Use horizontal morph close to bridge horizontal character gaps within line
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, img_w // 20), 3))
        line_mask = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_line)

        contours_lines, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_boxes = []
        for cnt in contours_lines:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 8 and w > 10:  # Filter noise lines
                line_boxes.append((x, y, w, h))

        # Sort lines top-to-bottom
        line_boxes.sort(key=lambda b: b[1])

        # If no lines found, treat entire image as 1 line
        if not line_boxes:
            line_boxes = [(0, 0, img_w, img_h)]

        hierarchical_result = []

        for line_idx, (lx, ly, lw, lh) in enumerate(line_boxes):
            line_gray = gray_img[ly:ly+lh, lx:lx+lw]
            line_bin = binary_img[ly:ly+lh, lx:lx+lw]

            # 2. WORD DETECTION
            # Use smaller horizontal kernel to connect character components into words
            kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (max(4, lw // 35), 2))
            word_mask = cv2.morphologyEx(line_bin, cv2.MORPH_CLOSE, kernel_word)

            contours_words, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_boxes = []
            for wcnt in contours_words:
                wx, wy, ww, wh = cv2.boundingRect(wcnt)
                if ww > 3 and wh > 5:
                    word_boxes.append((wx, wy, ww, wh))

            # Sort words left-to-right
            word_boxes.sort(key=lambda b: b[0])

            if not word_boxes:
                word_boxes = [(0, 0, lw, lh)]

            words_data = []

            for word_idx, (wx, wy, ww, wh) in enumerate(word_boxes):
                word_bin = line_bin[wy:wy+wh, wx:wx+ww]
                
                # Full doc absolute bounding box for word
                abs_wx = lx + wx
                abs_wy = ly + wy

                # 3. CHARACTER DETECTION
                contours_chars, _ = cv2.findContours(word_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                char_boxes = []
                for ccnt in contours_chars:
                    cx, cy, cw, ch = cv2.boundingRect(ccnt)
                    if cw > 2 and ch > 4:
                        char_boxes.append((cx, cy, cw, ch))

                # Sort characters left-to-right
                char_boxes.sort(key=lambda b: b[0])

                if not char_boxes:
                    char_boxes = [(0, 0, ww, wh)]

                chars_data = []

                for char_idx, (cx, cy, cw, ch) in enumerate(char_boxes):
                    abs_cx = abs_wx + cx
                    abs_cy = abs_wy + cy

                    char_crop = word_bin[cy:cy+ch, cx:cx+cw]
                    processed_char = self.pad_and_resize_char(char_crop)

                    chars_data.append({
                        "char_index": char_idx,
                        "bbox": {"x": abs_cx, "y": abs_cy, "w": cw, "h": ch},
                        "char_crop_raw": char_crop,
                        "char_crop_input": processed_char
                    })

                words_data.append({
                    "word_index": word_idx,
                    "bbox": {"x": abs_wx, "y": abs_wy, "w": ww, "h": wh},
                    "characters": chars_data
                })

            hierarchical_result.append({
                "line_index": line_idx,
                "bbox": {"x": lx, "y": ly, "w": lw, "h": lh},
                "words": words_data
            })

        return hierarchical_result
