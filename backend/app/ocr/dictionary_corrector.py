"""
Dictionary-Based Post-Correction Module (Rule-Based & Levenshtein Edit Distance)
"""

from typing import List, Dict, Any

HISTORICAL_MARATHI_LEXICON = [
    "छत्रपती", "शहाजी", "पेशवे", "निवाडा", "पत्र", "फर्मान", "जहागीर",
    "रयतेचे", "हुकूम", "गाव", "सनद", "इनाम", "सरकार", "मुकाम", "श्रीमंत",
    "राज्य", "आज्ञा", "कारभारी", "अष्टप्रधान", "किल्ला", "मावळा", "स्वराज्य",
    "सुभेदार", "किल्लेदार", "जमीन", "महसूल", "लेखा", "वतन", "कागदपत्रे",
    "इतिहास", "दस्तऐवज", "शिवछत्रपती", "बाजीराव", "माधवराव", "तानाजी"
]

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class DictionaryCorrector:
    def __init__(self, dictionary_words: List[str] = None):
        self.dictionary = dictionary_words or HISTORICAL_MARATHI_LEXICON

    def find_suggestions(self, target_word: str, max_distance: int = 2, top_n: int = 5) -> List[Dict[str, Any]]:
        if not target_word:
            return []

        suggestions = []
        target_len = len(target_word)

        for dict_word in self.dictionary:
            dist = levenshtein_distance(target_word, dict_word)
            if dist <= max_distance:
                # Calculate similarity ratio
                max_len = max(target_len, len(dict_word))
                similarity = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                
                suggestions.append({
                    "suggested_word": dict_word,
                    "devanagari_word": dict_word,
                    "distance": dist,
                    "confidence_score": round(similarity, 3)
                })

        # Sort by edit distance ascending, then similarity descending
        suggestions.sort(key=lambda x: (x["distance"], -x["confidence_score"]))
        return suggestions[:top_n]
