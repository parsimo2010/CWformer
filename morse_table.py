"""
Complete ITU Morse code table: letters, digits, punctuation, and prosigns.

Prosigns are keyed with angle-bracket names like '<AR>' and are decoded as a
single printed token.  They share bit-sequences with some punctuation (e.g.
'<AR>' == '+' == '.-.-.') — the table keeps both entries; prosigns shadow
the punctuation synonyms in the combined decode table.

Tree structure
--------------
The Morse tree is a binary trie where left (0) = dit ('.') and right (1) = dah ('-').
Depth-first, left = dit branch, right = dah branch.
Characters with longer codes are deeper in the tree.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Raw code table  (code string -> display character)
# ---------------------------------------------------------------------------

# Letters
_LETTERS: dict[str, str] = {
    ".-":   "A",
    "-...": "B",
    "-.-.": "C",
    "-..":  "D",
    ".":    "E",
    "..-.": "F",
    "--.":  "G",
    "....": "H",
    "..":   "I",
    ".---": "J",
    "-.-":  "K",
    ".-..": "L",
    "--":   "M",
    "-.":   "N",
    "---":  "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.":  "R",
    "...":  "S",
    "-":    "T",
    "..-":  "U",
    "...-": "V",
    ".--":  "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
}

# Digits
_DIGITS: dict[str, str] = {
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    "-----": "0",
}

# Punctuation — common subset used in amateur CW / QSOs.
# Removed: ' ! ) : ; - _ " $ @  (6–7 element sequences, never/rarely seen on air)
_PUNCTUATION: dict[str, str] = {
    ".-.-.-": ".",
    "--..--": ",",
    "..--..": "?",
    "-..-.":  "/",
    "-.--.":  "(",      # also <KN>
    ".-...":  "&",      # also <AS>
    "-...-":  "=",      # also <BT>
    ".-.-.":  "+",      # also <AR>
}

# Prosigns  (sent without inter-letter space; displayed with angle brackets)
# Removed: <HH> (8 elements), <SOS> (9 elements) — rarely needed for QSO;
# <SOS> decodes naturally as S-O-S when letter spaces are present.
_PROSIGNS: dict[str, str] = {
    ".-.-.":   "<AR>",    # end of message  (= +)
    "...-.-":  "<SK>",    # end of contact / silent key
    "-...-":   "<BT>",    # paragraph separator  (= =)
    "-.--.":   "<KN>",    # go ahead, specific station only  (= ()
    ".-...":   "<AS>",    # wait  (= &)
    "-.-.-":   "<CT>",    # start of transmission  (KA)
    "...-.":   "<SN>",    # understood  (VE)
}

# Combined decode table: code -> char.
# Prosigns shadow punctuation synonyms; all others coexist.
DECODE_TABLE: dict[str, str] = {}
DECODE_TABLE.update(_LETTERS)
DECODE_TABLE.update(_DIGITS)
DECODE_TABLE.update(_PUNCTUATION)
DECODE_TABLE.update(_PROSIGNS)   # prosigns overwrite punctuation synonyms

# Encode table: char -> code  (excludes prosign overlaps for punctuation)
ENCODE_TABLE: dict[str, str] = {}
for code, char in _LETTERS.items():
    ENCODE_TABLE[char] = code
for code, char in _DIGITS.items():
    ENCODE_TABLE[char] = code
for code, char in _PUNCTUATION.items():
    if char not in ENCODE_TABLE:
        ENCODE_TABLE[char] = code
for code, char in _PROSIGNS.items():
    ENCODE_TABLE[char] = code

# ---------------------------------------------------------------------------
# Morse binary trie
# ---------------------------------------------------------------------------

@dataclass
class MorseNode:
    char: str | None = None          # character at this node (None = internal)
    children: dict[str, "MorseNode"] = field(default_factory=dict)  # '.' or '-'

    def get(self, element: str) -> "MorseNode | None":
        return self.children.get(element)

    def step(self, element: str) -> "MorseNode":
        """Return child node, creating it if absent."""
        if element not in self.children:
            self.children[element] = MorseNode()
        return self.children[element]

    @property
    def is_terminal(self) -> bool:
        return self.char is not None

    @property
    def has_children(self) -> bool:
        return bool(self.children)


def _build_tree() -> MorseNode:
    root = MorseNode()
    for code, char in DECODE_TABLE.items():
        node = root
        for element in code:
            node = node.step(element)
        node.char = char
    return root


MORSE_TREE: MorseNode = _build_tree()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def decode_elements(elements: str) -> str | None:
    """Decode a dot-dash string like '.-' → 'A'. Returns None if unknown."""
    return DECODE_TABLE.get(elements)


def encode_char(char: str) -> str | None:
    """Encode a character to its dot-dash string. Returns None if unknown."""
    return ENCODE_TABLE.get(char.upper())


def is_valid_prefix(elements: str) -> bool:
    """Return True if `elements` is a valid prefix in the Morse tree."""
    node = MORSE_TREE
    for e in elements:
        node = node.get(e)  # type: ignore[assignment]
        if node is None:
            return False
    return True


def all_codes() -> list[tuple[str, str]]:
    """Return all (code, char) pairs sorted by code length then alphabetically."""
    return sorted(DECODE_TABLE.items(), key=lambda x: (len(x[0]), x[0]))
