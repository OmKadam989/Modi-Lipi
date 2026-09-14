from typing import List, Dict, Any

class TextReconstructor:
    @staticmethod
    def reconstruct_document_text(page_lines: List[Dict[str, Any]]) -> tuple[str, str, str]:
        """
        Reconstructs original character sequence into line-by-line:
        1. Raw Modi text
        2. Corrected Modi text
        3. Devanagari text
        """
        raw_modi_lines = []
        corrected_modi_lines = []
        devanagari_lines = []

        for line in page_lines:
            raw_words = []
            corrected_words = []
            dev_words = []

            for word in line.get("words", []):
                raw_w = word.get("raw_modi_word", "")
                corr_w = word.get("corrected_modi_word", "")
                dev_w = word.get("devanagari_word", "")

                if not raw_w and word.get("characters"):
                    raw_w = "".join([c.get("predicted_modi_char", "") for c in word["characters"]])
                if not corr_w and word.get("characters"):
                    corr_w = "".join([c.get("user_corrected_char") or c.get("predicted_modi_char", "") for c in word["characters"]])
                if not dev_w and word.get("characters"):
                    dev_w = "".join([c.get("predicted_devanagari_char", "") for c in word["characters"]])

                raw_words.append(raw_w)
                corrected_words.append(corr_w)
                dev_words.append(dev_w)

            raw_modi_lines.append(" ".join(raw_words))
            corrected_modi_lines.append(" ".join(corrected_words))
            devanagari_lines.append(" ".join(dev_words))

        raw_modi_text = "\n".join(raw_modi_lines)
        corrected_modi_text = "\n".join(corrected_modi_lines)
        devanagari_text = "\n".join(devanagari_lines)

        return raw_modi_text, corrected_modi_text, devanagari_text
