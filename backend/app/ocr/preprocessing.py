import cv2
import numpy as np
import base64

def numpy_to_base64(img: np.ndarray, format_ext: str = ".png") -> str:
    """Helper to convert OpenCV image matrix to base64 data URL for frontend display."""
    success, buffer = cv2.imencode(format_ext, img)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

class Preprocessor:
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image file bytes. Ensure file is a valid JPG/PNG/JPEG.")
        return img

    @staticmethod
    def load_image_from_path(file_path: str) -> np.ndarray:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image at path: {file_path}")
        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def denoise(gray: np.ndarray) -> np.ndarray:
        """Apply Bilateral Filter to preserve edges while clearing document noise."""
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def binarize(gray: np.ndarray) -> np.ndarray:
        """Otsu's thresholding to produce clean binary mask (0=bg, 255=fg/text)."""
        # Inverse thresholding so text is white (255) on black background (0) for contour analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def deskew(gray: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculates text skew angle and rotates both grayscale and binarized images."""
        coords = np.column_stack(np.where(binary > 0))
        if coords.shape[0] < 10:
            return gray, binary, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If angle is minor, don't overrotate
        if abs(angle) < 0.5 or abs(angle) > 25.0:
            return gray, binary, 0.0

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed_gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        deskewed_bin = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed_gray, deskewed_bin, float(angle)

    def run_pipeline(self, image_bytes: bytes):
        """Runs complete preprocessing pipeline and returns intermediate images as base64 data URLs."""
        color_img = self.load_image_from_bytes(image_bytes)
        gray = self.to_grayscale(color_img)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        binary = self.binarize(enhanced)
        deskewed_gray, deskewed_bin, skew_angle = self.deskew(enhanced, binary)

        # Standard display binary (black text on white bg for preview)
        preview_binary = cv2.bitwise_not(deskewed_bin)

        return {
            "original_b64": numpy_to_base64(color_img),
            "grayscale_b64": numpy_to_base64(deskewed_gray),
            "binarized_b64": numpy_to_base64(preview_binary),
            "deskewed_b64": numpy_to_base64(deskewed_gray),
            "skew_angle": skew_angle,
            "raw_gray": deskewed_gray,
            "raw_binary": deskewed_bin, # White text on black bg for contour segmentation
            "color_img": color_img
        }
