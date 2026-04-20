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
  beam_search_ctc(log_probs, beam_width=10, blank_idx=0, strip_trailing_space=False) -> str
    CTC prefix beam search decode (Graves 2012).  More accurate than greedy
    in noisy conditions; O(T × beam_width × vocab_size) time.
"""

from __future__ import annotations

import json
import math
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
# CTC prefix beam search
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    _neg_inf = float("-inf")
    if a == _neg_inf:
        return b
    if b == _neg_inf:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def beam_search_ctc(
    log_probs: "Tensor",
    beam_width: int = 10,
    blank_idx: int = 0,
    strip_trailing_space: bool = False,
) -> str:
    """CTC prefix beam search decoding for a single sample.

    Implements the standard CTC prefix beam search from Graves et al. (2012).
    Each beam (output prefix) tracks two log-probabilities:

    - ``p_b``  — probability of all paths producing this prefix that end in blank
    - ``p_nb`` — probability of all paths producing this prefix that end in the
      last non-blank token

    The two-probability representation correctly handles the CTC merging rule:
    a path "a a" collapses to "a" (single char), but "a _ a" (blank between)
    produces "aa" (two chars).

    Performance: O(T × beam_width × top_k) where top_k = min(beam_width×2, C-1).
    Token pruning keeps only the most probable non-blank tokens at each step,
    which is typically 6–10× faster than iterating over all C tokens while
    having negligible accuracy impact (low-probability tokens rarely survive
    beam pruning anyway).

    Args:
        log_probs: ``(time, num_classes)`` log-probability tensor for one sample.
        beam_width: Number of beams to keep at each step.  ``beam_width=1`` is
            equivalent to greedy decoding (but slower; use ``decode_ctc``
            for greedy).  Good starting points: 10 (fast), 50 (accurate).
        blank_idx: Index of the CTC blank token.
        strip_trailing_space: If ``True``, strip trailing space characters.

    Returns:
        Best decoded string.
    """
    import torch   # lazy import — keeps vocab.py lightweight
    import numpy as np

    NEG_INF = float("-inf")

    # Convert to numpy once for fast per-element access (stay in log space)
    log_probs_np: np.ndarray = log_probs.cpu().float().numpy()  # (T, C)
    T, C = log_probs_np.shape

    if T == 0:
        return ""

    # Token pruning: only expand the top-k non-blank tokens at each timestep.
    # With beam_width=10 and C=52, using top_k=20 instead of 51 non-blank
    # tokens gives ~2.5× speedup with negligible accuracy loss (the remaining
    # tokens have tiny probability and won't survive beam pruning).
    top_k = min(beam_width * 2, C - 1)

    # beams: dict mapping prefix (tuple of token ints) → (log_p_b, log_p_nb)
    beams: dict = {(): (0.0, NEG_INF)}  # empty prefix: p_b=1, p_nb=0

    def _update(d: dict, key: tuple, lpb: float, lpnb: float) -> None:
        if key in d:
            ob, onb = d[key]
            d[key] = (_log_add(ob, lpb), _log_add(onb, lpnb))
        else:
            d[key] = (lpb, lpnb)

    for t in range(T):
        log_p_t: np.ndarray = log_probs_np[t]  # already in log space, (C,)

        # Blank log-prob
        lp_blank = float(log_p_t[blank_idx])

        # Top-k non-blank token indices for this timestep
        non_blank_mask = np.ones(C, dtype=bool)
        non_blank_mask[blank_idx] = False
        nb_ids = np.where(non_blank_mask)[0]
        top_ids = nb_ids[np.argsort(log_p_t[nb_ids])[::-1][:top_k]]

        new_beams: dict = {}

        for prefix, (log_p_b, log_p_nb) in beams.items():
            log_p_tot = _log_add(log_p_b, log_p_nb)

            # ---- Blank: same prefix, via blank path only ----------------
            _update(new_beams, prefix, log_p_tot + lp_blank, NEG_INF)

            # ---- Top-k non-blank tokens ---------------------------------
            for c in top_ids:
                c = int(c)
                lp_c = float(log_p_t[c])

                if prefix and prefix[-1] == c:
                    # Same token as last char:
                    # Case A — repeat without extending (non-blank prev only)
                    _update(new_beams, prefix, NEG_INF, log_p_nb + lp_c)
                    # Case B — new copy after a blank
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_b + lp_c)
                else:
                    # Normal extension with a different (or first) token
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_tot + lp_c)

        # Prune to top beam_width by total log-probability
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best = max(beams, key=lambda p: _log_add(beams[p][0], beams[p][1]))
    text = "".join(idx_to_char.get(i, "") for i in best if i != blank_idx)
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
