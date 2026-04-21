"""
vocab.py — CTC character vocabulary for CWformer.

Index 0 is always reserved for the CTC blank token.
The vocabulary covers:
  • blank  (index 0)
  • space  (index 1)
  • A–Z    (indices 2–27)
  • 0–9    (indices 28–37)
  • punctuation: .,?/(&=+  (indices 38–45)
  • prosigns as single tokens: AR SK BT KN AS CT  (indices 46–51)

Punctuation rationale: only 5-element sequences commonly seen on air.
Removed: ' ! ) : ; - _ " $ @  (6–7 element sequences, never/rarely on air).
Prosign rationale: standard ITU operating prosigns only.
Removed: SOS (9 elements, decodes as S-O-S with letter spacing in QSOs),
         DN (non-standard; code identical to /, creating contradictory labels).

After import the following module-level names are available:
  char_to_idx : Dict[str, int]
  idx_to_char : Dict[int, str]
  num_classes : int
  BLANK_IDX   : int  (always 0)
  PROSIGNS    : List[str]

CTC decoding utilities:
  decode_ctc(log_probs, blank_idx=0, strip_trailing_space=False) -> str
    Greedy argmax CTC decode for a single sample.
    Shared by train.py and inference.py to avoid code duplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    # Avoid hard torch import at module level so vocab.py stays lightweight.
    # At runtime decode_ctc() imports torch lazily on first call.
    from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLANK_TOKEN: str = "<blank>"
BLANK_IDX: int = 0

LETTERS: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS: List[str] = [str(d) for d in range(10)]
PUNCTUATION: List[str] = list(".,?/(&=+")
PROSIGNS: List[str] = ["AR", "SK", "BT", "KN", "AS", "CT"]


# ---------------------------------------------------------------------------
# Build vocabulary
# ---------------------------------------------------------------------------

def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Construct character ↔ index mappings."""
    tokens: List[str] = (
        [BLANK_TOKEN]   # index 0 — CTC blank
        + [" "]          # index 1 — word-separator space
        + LETTERS        # indices 2–27  (26 letters)
        + DIGITS         # indices 28–37 (10 digits)
        + PUNCTUATION    # indices 38–45 (8 punctuation: .,?/(&=+)
        + PROSIGNS       # indices 46–51 (6 prosigns: AR SK BT KN AS CT)
    )   # total 52 classes (= 1 blank + 1 space + 26 + 10 + 8 + 6)
    c2i: Dict[str, int] = {tok: idx for idx, tok in enumerate(tokens)}
    i2c: Dict[int, str] = {idx: tok for idx, tok in enumerate(tokens)}
    return c2i, i2c


char_to_idx, idx_to_char = _build_vocab()

#: Total number of output classes (including blank).
num_classes: int = len(char_to_idx)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_vocab(path: str = "vocab.json") -> None:
    """Persist the vocabulary to *path* as JSON.

    The file contains two keys:
      ``char_to_idx`` — maps token strings to integer indices.
      ``idx_to_char`` — maps string-encoded indices back to tokens.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "char_to_idx": char_to_idx,
        # JSON keys must be strings, so we cast int keys to str.
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_vocab(path: str = "vocab.json") -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load a saved vocabulary from *path*.

    Returns ``(char_to_idx, idx_to_char)`` with integer keys in the second dict.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    c2i: Dict[str, int] = data["char_to_idx"]
    i2c: Dict[int, str] = {int(k): v for k, v in data["idx_to_char"].items()}
    return c2i, i2c


def encode(text: str) -> List[int]:
    """Convert a text string (may include prosign tokens) to a list of indices.

    Tokens are resolved greedily: prosigns (multi-character) are matched first
    when they appear as whole whitespace-delimited words.

    Args:
        text: upper-case string; prosigns appear as space-delimited words.

    Returns:
        List of integer vocabulary indices (no blank tokens).
    """
    indices: List[int] = []
    words = text.split(" ")
    for w_idx, word in enumerate(words):
        if not word:
            continue
        if word in char_to_idx:
            # Prosign or single-character token
            indices.append(char_to_idx[word])
        else:
            for ch in word:
                if ch in char_to_idx:
                    indices.append(char_to_idx[ch])
        # Insert space between words (but not after the last one)
        if w_idx < len(words) - 1 and " " in char_to_idx:
            # Only add if the next non-empty word exists
            remaining = [w for w in words[w_idx + 1:] if w]
            if remaining:
                indices.append(char_to_idx[" "])
    return indices


def decode(indices: List[int]) -> str:
    """Convert a list of vocabulary indices back to a string.

    Prosign tokens are returned as their string representations (e.g. ``"AR"``).
    """
    return "".join(idx_to_char.get(i, "") for i in indices if i != BLANK_IDX)


def decode_ctc(
    log_probs: "Tensor",
    blank_idx: int = 0,
    strip_trailing_space: bool = False,
) -> str:
    """Greedy CTC decoding for a single sample.

    Collapses repeated consecutive tokens, removes blank tokens, then maps
    indices to characters.  This is the canonical implementation shared by
    the training loop (``train.py``) and both inference classes.

    Args:
        log_probs: ``(time, num_classes)`` log-probability tensor for one
            sample (not a batch).
        blank_idx: Index of the CTC blank token (default 0).
        strip_trailing_space: If ``True``, strip trailing space characters
            from the decoded string.  Useful when computing CER because the
            model often assigns space to trailing silence frames.

    Returns:
        Decoded string.
    """
    import torch  # lazy import — keeps vocab.py lightweight

    indices = torch.argmax(log_probs, dim=-1).cpu().tolist()

    # Collapse consecutive duplicates
    collapsed: List[int] = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(idx)
            prev = idx

    text = "".join(
        idx_to_char[i]
        for i in collapsed
        if i != blank_idx and i in idx_to_char
    )
    if strip_trailing_space:
        text = text.rstrip(" ")
    return text


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Vocabulary size : {num_classes}")
    print(f"Blank index     : {BLANK_IDX}")
    print(f"Space index     : {char_to_idx[' ']}")
    print(f"Prosigns        : {PROSIGNS}")
    sample = "CQ DE W1AW AR"
    enc = encode(sample)
    dec = decode(enc)
    print(f"Encode/decode   : {sample!r} → {enc} → {dec!r}")
    save_vocab("vocab.json")
    print("Saved vocab.json")
