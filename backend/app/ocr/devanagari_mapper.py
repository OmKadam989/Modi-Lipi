"""
Modi Lipi to Devanagari Script Conversion Mapping Engine
"""

MODI_TO_DEVANAGARI_MAP = {
    # Vowels
    "\u11600": "अ",
    "\u11601": "आ",
    "\u11602": "इ",
    "\u11603": "ई",
    "\u11604": "उ",
    "\u11605": "ऊ",
    "\u11606": "ऋ",
    "\u11607": "ॠ",
    "\u11608": "ऌ",
    "\u11609": "ए",
    "\u1160A": "ऐ",
    "\u1160B": "ओ",
    "\u1160C": "औ",
    # Consonants
    "\u1160E": "क",
    "\u1160F": "ख",
    "\u11610": "ग",
    "\u11611": "घ",
    "\u11612": "ङ",
    "\u11613": "च",
    "\u11614": "छ",
    "\u11615": "ज",
    "\u11616": "झ",
    "\u11617": "ञ",
    "\u11618": "ट",
    "\u11619": "ठ",
    "\u1161A": "ड",
    "\u1161B": "ढ",
    "\u1161C": "ण",
    "\u1161D": "त",
    "\u1161E": "थ",
    "\u1161F": "द",
    "\u11620": "ध",
    "\u11621": "न",
    "\u11622": "प",
    "\u11623": "फ",
    "\u11624": "ब",
    "\u11625": "भ",
    "\u11626": "म",
    "\u11627": "य",
    "\u11628": "र",
    "\u11629": "ल",
    "\u1162A": "व",
    "\u1162B": "श",
    "\u1162C": "ष",
    "\u1162D": "स",
    "\u1162E": "ह",
    "\u1162F": "ळ",
    # Dependent Vowel Signs (Matras)
    "\u11630": "ा",
    "\u11631": "ि",
    "\u11632": "ी",
    "\u11633": "ु",
    "\u11634": "ू",
    "\u11635": "ृ",
    "\u11637": "े",
    "\u11638": "ै",
    "\u11639": "ो",
    "\u1163A": "ौ",
    "\u1163B": "ं",
    "\u1163C": "ः",
    "\u1163D": "ँ",
    # Digits
    "\u11650": "०",
    "\u11651": "१",
    "\u11652": "२",
    "\u11653": "३",
    "\u11654": "४",
    "\u11655": "५",
    "\u11656": "६",
    "\u11657": "७",
    "\u11658": "८",
    "\u11659": "९",
}

class DevanagariMapper:
    @staticmethod
    def map_char(modi_char: str) -> str:
        """Maps a single Modi character to its Devanagari equivalent."""
        return MODI_TO_DEVANAGARI_MAP.get(modi_char, modi_char)

    @staticmethod
    def map_text(modi_text: str) -> str:
        """Converts a full Modi Lipi text string to Devanagari script."""
        result = []
        for ch in modi_text:
            result.append(MODI_TO_DEVANAGARI_MAP.get(ch, ch))
        return "".join(result)
