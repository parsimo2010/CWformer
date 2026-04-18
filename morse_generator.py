"""
morse_generator.py — Synthetic Morse code audio generation for CWNet.

Generates float32 audio at config.sample_rate for CW-Former training
and validation.

Features:
  • Three timing parameters sampled independently per sample:
      dah_dit_ratio  — dah duration in units of one dit (ITU = 3.0)
      ics_factor     — multiplier on the standard 3-dit inter-char gap
      iws_factor     — multiplier on the standard 7-dit inter-word gap
  • AGC simulation: noise amplitude is modulated inversely to the signal
    envelope, matching the noise-floor drift seen in real HF recordings.
  • QSB: slow sinusoidal amplitude fading within a sample.
  • White AWGN noise for maximum generation speed.
  • Slow sinusoidal frequency drift (tests peak-bin tracking).
  • Four key type simulations (straight, bug, paddle, cootie).
  • QRM, QRN, bandpass filter, real HF noise mixing augmentations.

Timing follows the PARIS standard:
  dit duration      = 1 unit
  dah duration      = dah_dit_ratio units   (default 3.0)
  intra-char gap    = 1 unit
  inter-char gap    = 3 × ics_factor units
  inter-word gap    = 7 × iws_factor units
  1 unit            = 60 / (wpm × 50) seconds
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional numba JIT for the AGC inner loop (significant speedup for long samples).
# Falls back to a pure-Python loop if numba is not installed.
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _agc_envelope_kernel(
        sig_sq: np.ndarray, alpha_atk: float, alpha_rel: float
    ) -> np.ndarray:
        n = len(sig_sq)
        envelope = np.empty(n, dtype=np.float64)
        e = 0.0
        for i in range(n):
            alpha = alpha_atk if sig_sq[i] >= e else alpha_rel
            e += alpha * (sig_sq[i] - e)
            envelope[i] = e
        return envelope

except ImportError:

    def _agc_envelope_kernel(  # type: ignore[misc]
        sig_sq: np.ndarray, alpha_atk: float, alpha_rel: float
    ) -> np.ndarray:
        n = len(sig_sq)
        envelope = np.empty(n, dtype=np.float64)
        e = 0.0
        for i in range(n):
            alpha = alpha_atk if sig_sq[i] >= e else alpha_rel
            e += alpha * (sig_sq[i] - e)
            envelope[i] = e
        return envelope

from config import MorseConfig
from vocab import PROSIGNS


# ---------------------------------------------------------------------------
# Morse code table (inline, so this module is self-contained)
# ---------------------------------------------------------------------------

MORSE_TABLE: Dict[str, str] = {
    # Letters
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",
    "E": ".",     "F": "..-.",  "G": "--.",   "H": "....",
    "I": "..",    "J": ".---",  "K": "-.-",   "L": ".-..",
    "M": "--",    "N": "-.",    "O": "---",   "P": ".--.",
    "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",
    "Y": "-.--",  "Z": "--..",
    # Digits
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
    # Punctuation — common 5-element sequences only (matches vocab.py)
    # Removed: ' ! ) : ; - _ " $ @  (6–7 element sequences, never/rarely on air)
    ".": ".-.-.-", ",": "--..--", "?": "..--..",
    "/": "-..-.",  "(": "-.--.",  "&": ".-...",
    "=": "-...-",  "+": ".-.-.",
    # Prosigns (transmitted as uninterrupted sequences)
    "AR": ".-.-.",  "SK": "...-.-", "BT": "-...-",
    "KN": "-.--.",  "AS": ".-...",  "CT": "-.-.-",
}

_ENCODABLE: frozenset = frozenset(MORSE_TABLE.keys())

LETTERS: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS: List[str] = [str(d) for d in range(10)]
PUNCTUATION: List[str] = list(".,?/(&=+")

# Common CW abbreviations heard on the air
CW_ABBREVIATIONS: List[str] = [
    "CQ", "DE", "EE", "73", "88", "RST", "UR", "ES", "TNX", "FB",
    "OM", "HI", "K", "R", "QTH", "QSL", "QRZ", "AGN", "PSE", "WX",
    "GL", "GE", "GM", "GA", "GN", "DX", "ANT", "RIG", "HR", "NR",
]

_KEY_TYPES = ("straight", "bug", "paddle")


def _select_key_type(
    weights: Tuple[float, ...],
    rng: np.random.Generator,
) -> str:
    """Select a key type based on configured weights.

    Supports 3-tuple (straight, bug, paddle) or 4-tuple (+ cootie).
    """
    # Normalise to handle both 3- and 4-tuples
    w = list(weights)
    if len(w) < 4:
        w.append(0.0)  # no cootie if not specified
    total = sum(w)
    if total <= 0:
        return "paddle"
    r = float(rng.random()) * total
    cum = 0.0
    for key_type, wt in zip(("straight", "bug", "paddle", "cootie"), w):
        cum += wt
        if r < cum:
            return key_type
    return "paddle"


# ---------------------------------------------------------------------------
# Real-world augmentation helpers
# ---------------------------------------------------------------------------

def _agc_noise_modulation(
    signal: np.ndarray,
    noise: np.ndarray,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
    depth_db: float,
) -> np.ndarray:
    """Scale noise amplitude inversely to signal envelope (AGC simulation).

    During marks the noise is attenuated by *depth_db*.  During spaces it
    returns to full amplitude with the *release_ms* time constant.

    This replicates the effect of a radio AGC that reduces IF gain when a
    strong signal is present, causing the noise floor visible between elements
    to be significantly higher than the noise floor during marks.  The result
    in the SNR feature is that inter-element and inter-word spaces have an
    elevated, slowly-decaying noise baseline rather than a flat negative value.
    """
    alpha_atk = 1.0 - math.exp(-1.0 / max(1.0, attack_ms  * 1e-3 * sample_rate))
    alpha_rel = 1.0 - math.exp(-1.0 / max(1.0, release_ms * 1e-3 * sample_rate))

    sig_sq   = signal.astype(np.float64) ** 2
    envelope = _agc_envelope_kernel(sig_sq, alpha_atk, alpha_rel)

    peak = envelope.max()
    if peak < 1e-12:
        return noise      # no signal — AGC has nothing to react to

    envelope /= peak      # normalised: 1.0 at strongest mark, ~0 in deep spaces

    # Noise gain: 1/depth_lin during peak marks, 1.0 during spaces
    depth_lin  = 10.0 ** (depth_db / 20.0)                # > 1
    noise_gain = (1.0 / (1.0 + (depth_lin - 1.0) * envelope)).astype(np.float32)
    return noise * noise_gain


def _apply_qsb(
    signal: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    depth_db: float,
) -> np.ndarray:
    """Apply slow sinusoidal amplitude fading (QSB) to the signal.

    The fading rate is 0.05–0.3 Hz, producing mark-to-mark amplitude
    variation over several seconds — matching propagation fading on HF.
    """
    fade_freq  = np.float32(rng.uniform(0.05, 0.3))
    fade_phase = np.float32(rng.uniform(0.0, 2 * math.pi))
    fade_db    = np.float32(depth_db / 2.0) * np.sin(
        np.float32(2 * math.pi) * fade_freq * t + fade_phase
    )
    fade_lin   = np.float32(10.0) ** (fade_db.astype(np.float32) / np.float32(20.0))
    return signal * fade_lin


# ---------------------------------------------------------------------------
# HF noise loader (cached, thread-safe)
# ---------------------------------------------------------------------------

import threading as _threading

_hf_noise_cache: Dict[str, np.ndarray] = {}
_hf_noise_lock = _threading.Lock()


def _load_hf_noise_files(noise_dir: str, target_sr: int = 16000) -> List[np.ndarray]:
    """Load and cache all WAV files from noise_dir, resampled to target_sr."""
    import glob
    import soundfile as sf

    key = f"{noise_dir}:{target_sr}"
    with _hf_noise_lock:
        if key in _hf_noise_cache:
            return list(_hf_noise_cache[key])

    noise_files = sorted(glob.glob(str(Path(noise_dir) / "noise_*.wav")))
    if not noise_files:
        return []

    buffers = []
    for fpath in noise_files:
        data, sr = sf.read(fpath, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        # Resample if needed (simple linear interpolation — noise doesn't need high quality)
        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, n_out)
            idx_floor = np.floor(indices).astype(np.int64)
            idx_ceil = np.minimum(idx_floor + 1, len(data) - 1)
            frac = (indices - idx_floor).astype(np.float32)
            data = data[idx_floor] * (1.0 - frac) + data[idx_ceil] * frac
        buffers.append(data)

    with _hf_noise_lock:
        # Store as a single array tuple for the cache key
        _hf_noise_cache[key] = buffers
    return buffers


def _get_hf_noise_segment(
    noise_dir: str,
    n_samples: int,
    rng: np.random.Generator,
    target_sr: int = 16000,
) -> Optional[np.ndarray]:
    """Extract a random segment of real HF noise, resampled to target_sr."""
    buffers = _load_hf_noise_files(noise_dir, target_sr)
    if not buffers:
        return None

    # Pick a random file
    buf = buffers[int(rng.integers(len(buffers)))]
    if len(buf) < n_samples:
        # Tile if recording is shorter than needed (unlikely)
        repeats = (n_samples // len(buf)) + 1
        buf = np.tile(buf, repeats)

    # Random start position
    max_start = len(buf) - n_samples
    start = int(rng.integers(0, max(1, max_start)))
    return buf[start:start + n_samples].copy()


# ---------------------------------------------------------------------------
# QRM — interfering CW signals
# ---------------------------------------------------------------------------

def _apply_qrm(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    n_interferers: int,
    base_freq: float,
    freq_offset_min: float,
    freq_offset_max: float,
    amplitude_min: float,
    amplitude_max: float,
    duration_sec: float,
) -> np.ndarray:
    """Add interfering CW signals at nearby frequencies.

    Each interferer is a simple CW signal with random text, speed, and
    frequency offset from the target signal. This simulates QRM from
    other operators on adjacent frequencies.
    """
    n = len(audio)
    result = audio.copy()

    for _ in range(n_interferers):
        # Random frequency offset (can be positive or negative)
        offset = float(rng.uniform(freq_offset_min, freq_offset_max))
        if rng.random() < 0.5:
            offset = -offset
        interferer_freq = base_freq + offset

        # Random amplitude
        amp = float(rng.uniform(amplitude_min, amplitude_max))

        # Random keying pattern: random on/off with random speeds
        # Use a simple random binary keying rather than full Morse generation
        # to avoid recursive dependency and keep it fast
        interferer_wpm = float(rng.uniform(10.0, 35.0))
        dit_dur = 60.0 / (interferer_wpm * 50.0)

        t = np.arange(n, dtype=np.float32) / sample_rate

        # Generate random on/off keying envelope
        envelope = np.zeros(n, dtype=np.float32)
        pos = 0
        is_on = bool(rng.random() < 0.5)
        while pos < n:
            # Random element duration (dits and dahs)
            if is_on:
                dur_samples = int((dit_dur * float(rng.choice([1.0, 3.0]))) * sample_rate)
            else:
                dur_samples = int((dit_dur * float(rng.choice([1.0, 3.0, 7.0]))) * sample_rate)
            dur_samples = max(1, dur_samples)
            end = min(pos + dur_samples, n)
            if is_on:
                envelope[pos:end] = 1.0
            pos = end
            is_on = not is_on

        # Apply soft keying (5ms rise/fall)
        rise_samples = max(2, int(0.005 * sample_rate))
        ramp = (np.sin(np.linspace(0, math.pi / 2, rise_samples)) ** 2).astype(np.float32)
        # Find transitions and apply ramps
        diff = np.diff(envelope)
        onsets = np.where(diff > 0.5)[0] + 1
        offsets = np.where(diff < -0.5)[0] + 1
        for onset in onsets:
            r = min(rise_samples, n - onset)
            if r > 0:
                envelope[onset:onset + r] = np.minimum(envelope[onset:onset + r], ramp[:r])
        for offset_pos in offsets:
            r = min(rise_samples, n - offset_pos)
            if r > 0:
                envelope[offset_pos:offset_pos + r] = np.minimum(
                    envelope[offset_pos:offset_pos + r], ramp[:r][::-1] if r == rise_samples else ramp[:r][::-1]
                )

        # Generate tone
        phase = np.cumsum(np.float32(2 * math.pi * interferer_freq / sample_rate) * np.ones(n, dtype=np.float32))
        phase += np.float32(rng.uniform(0, 2 * math.pi))
        tone = np.sin(phase).astype(np.float32)

        result += amp * envelope * tone

    return result


# ---------------------------------------------------------------------------
# QRN — impulsive atmospheric noise (static crashes)
# ---------------------------------------------------------------------------

def _apply_qrn(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    rate: float,
    duration_ms_min: float,
    duration_ms_max: float,
    amplitude_min: float,
    amplitude_max: float,
    signal_rms: float,
) -> np.ndarray:
    """Add impulsive noise simulating atmospheric static crashes (QRN).

    Lightning-generated impulses are modelled as short bursts of filtered
    noise with exponential decay envelopes, Poisson-distributed in time.
    """
    total_sec = len(audio) / sample_rate
    n_impulses = int(rng.poisson(rate * total_sec))
    if n_impulses == 0:
        return audio

    result = audio.copy()
    for _ in range(n_impulses):
        # Random position
        pos = int(rng.integers(0, len(audio)))

        # Random duration and amplitude
        dur_ms = float(rng.uniform(duration_ms_min, duration_ms_max))
        dur_samples = max(1, int(dur_ms * 0.001 * sample_rate))
        amp = float(rng.uniform(amplitude_min, amplitude_max)) * signal_rms

        # Exponential decay envelope
        t_imp = np.arange(dur_samples, dtype=np.float32) / sample_rate
        decay_rate = float(rng.uniform(20.0, 100.0))  # faster = sharper crack
        envelope = np.exp(-decay_rate * t_imp).astype(np.float32)

        # Broadband noise burst (static crash)
        burst = rng.normal(0.0, 1.0, dur_samples).astype(np.float32) * envelope * amp

        # Random polarity (static crashes can be positive or negative)
        if rng.random() < 0.5:
            burst = -burst

        # Add to audio
        end = min(pos + dur_samples, len(audio))
        actual = end - pos
        result[pos:end] += burst[:actual]

    return result


# ---------------------------------------------------------------------------
# Receiver bandpass filter simulation
# ---------------------------------------------------------------------------

def _apply_bandpass(
    audio: np.ndarray,
    sample_rate: int,
    center_freq: float,
    bandwidth: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter simulating a CW receiver filter.

    Real CW filters have 200-500 Hz bandwidth centered on the beat note.
    This removes out-of-band noise and any QRM outside the passband.
    """
    from scipy.signal import butter, sosfilt

    low = center_freq - bandwidth / 2.0
    high = center_freq + bandwidth / 2.0
    nyq = sample_rate / 2.0

    # Clamp to valid range
    low = max(10.0, low) / nyq
    high = min(nyq - 10.0, high) / nyq

    if low >= high or high >= 1.0 or low <= 0.0:
        return audio  # can't filter with these parameters

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Word-list helpers
# ---------------------------------------------------------------------------

def load_wordlist(path: str = "wordlist.txt") -> Optional[List[str]]:
    """Load a word list, filtering to encodable characters only."""
    p = Path(path)
    if not p.exists():
        return None
    words: List[str] = []
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            word = line.strip().upper()
            if word and all(ch in _ENCODABLE for ch in word):
                words.append(word)
    return words or None


def _random_word(rng: np.random.Generator, wordlist: Optional[List[str]]) -> str:
    if wordlist and rng.random() < 0.7:
        return wordlist[rng.integers(len(wordlist))]
    length = int(rng.integers(2, 8))
    return "".join(LETTERS[rng.integers(len(LETTERS))] for _ in range(length))


def _random_number(rng: np.random.Generator) -> str:
    length = int(rng.integers(1, 5))
    return "".join(DIGITS[rng.integers(len(DIGITS))] for _ in range(length))


def generate_text(
    rng: np.random.Generator,
    min_chars: int = 8,
    max_chars: int = 60,
    wordlist: Optional[List[str]] = None,
) -> str:
    """Generate random text suitable for Morse encoding.

    Mix: ~60 % alpha words, ~20 % numbers, ~20 % mixed.
    Prosigns appended ~10 % of the time.
    """
    target = int(rng.integers(min_chars, max_chars + 1))
    words: List[str] = []
    total = 0

    while total < target:
        kind = rng.random()
        if kind < 0.15:
            # Common CW abbreviation (CQ, DE, 73, etc.)
            word = CW_ABBREVIATIONS[int(rng.integers(len(CW_ABBREVIATIONS)))]
        elif kind < 0.60:
            word = _random_word(rng, wordlist)
        elif kind < 0.80:
            word = _random_number(rng)
        else:
            word = _random_word(rng, wordlist)
            if rng.random() < 0.3 and PUNCTUATION:
                word += PUNCTUATION[int(rng.integers(len(PUNCTUATION)))]
        words.append(word)
        total += len(word) + 1

    text = " ".join(words)
    if rng.random() < 0.10:
        prosign = PROSIGNS[int(rng.integers(len(PROSIGNS)))]
        text = text + " " + prosign

    return text.strip()


# ---------------------------------------------------------------------------
# Morse timing → element list
# ---------------------------------------------------------------------------

Element = Tuple[bool, float]   # (is_tone, duration_seconds)


def _char_complexity(code: str) -> float:
    """Count dit↔dah transitions in a Morse code string (0 = uniform, higher = harder).

    Used by straight-key simulation: characters with more transitions are keyed
    slightly slower because the operator's wrist must reverse direction.
    """
    transitions = 0
    for i in range(1, len(code)):
        if code[i] != code[i - 1]:
            transitions += 1
    # Normalise: max transitions in a 5-element code is 4 (e.g. ".-.-.")
    return transitions / max(len(code) - 1, 1)


def text_to_elements(
    text: str,
    unit_dur: float,
    timing_jitter: float,
    rng: np.random.Generator,
    dah_dit_ratio: float = 3.0,
    ics_factor: float = 1.0,
    iws_factor: float = 1.0,
    key_type: str = "paddle",
    speed_drift_max: float = 0.0,
    farnsworth_stretch: float = 1.0,
    multi_op_speed_range: Optional[Tuple[float, float]] = None,
) -> List[Element]:
    """Convert text to (is_tone, duration_seconds) element pairs.

    Parameters
    ----------
    text : str
        Upper-case string; prosigns as space-delimited words.
    unit_dur : float
        Duration of one dit in seconds.
    timing_jitter : float
        Gaussian jitter std dev as a fraction of element duration.
        0 = perfect timing.  Interpreted differently per key_type.
    rng : np.random.Generator
    dah_dit_ratio : float
        Dah duration in dits (ITU standard = 3.0).
    ics_factor : float
        Inter-character gap multiplier (standard = 1.0 → 3 dits).
    iws_factor : float
        Inter-word gap multiplier (standard = 1.0 → 7 dits).
    key_type : str
        One of "straight", "bug", "paddle", "cootie".  Controls how jitter
        is applied:
        - straight: high jitter on all elements, per-char speed variation,
                    per-element dah/dit ratio variation.
        - bug: minimal dit jitter (mechanical), moderate dah jitter (manual),
               variable spacing.
        - paddle: minimal element jitter (electronic), moderate spacing jitter.
        - cootie: alternating side contacts; symmetric high jitter on all
                  elements, no inherent dit/dah distinction — operator must
                  time everything manually.
    speed_drift_max : float
        Slow WPM variation within the transmission as a fraction of unit_dur.
        0.0 = constant speed.  Applied as sinusoidal modulation across words.
    farnsworth_stretch : float
        Spacing stretch factor for Farnsworth timing.  1.0 = standard timing.
        >1.0 = inter-character and inter-word spaces stretched by this factor
        while intra-character timing stays at the faster character speed.
    multi_op_speed_range : tuple of (float, float), optional
        If set, applies abrupt speed changes at 1-3 random word boundaries,
        simulating operator changes on a multi-op station. Values are
        (min_multiplier, max_multiplier) for the speed change.
    """

    # ---- Multi-operator speed changes ------------------------------------
    # Pre-compute word indices where abrupt speed changes occur and their
    # multiplier values.
    multi_op_changes: Dict[int, float] = {}
    if multi_op_speed_range is not None:
        lo, hi = multi_op_speed_range
        words_temp = [w for w in text.split(" ") if w]
        if len(words_temp) >= 3:
            n_changes = int(rng.integers(1, min(4, len(words_temp) // 2) + 1))
            # Pick random word boundary indices (not first or last word)
            change_indices = sorted(rng.choice(
                range(1, len(words_temp)), size=min(n_changes, len(words_temp) - 1),
                replace=False,
            ))
            for idx in change_indices:
                multi_op_changes[int(idx)] = float(rng.uniform(lo, hi))

    # ---- Speed drift: sinusoidal modulation across the transmission ------
    # Pre-sample drift parameters; actual modulation applied per-word below.
    if speed_drift_max > 0.0:
        drift_freq = float(rng.uniform(0.3, 1.5))   # cycles per ~10 words
        drift_phase = float(rng.uniform(0.0, 2 * math.pi))
        drift_amplitude = float(rng.uniform(0.0, speed_drift_max))
    else:
        drift_freq = 0.0
        drift_phase = 0.0
        drift_amplitude = 0.0

    words = [w for w in text.split(" ") if w]
    n_words = len(words)

    # ---- Key-type-aware jitter functions ---------------------------------

    def _jitter_straight(units: float, is_dit: bool, is_dah: bool,
                         char_cplx: float, local_ud: float,
                         local_ddr: float) -> float:
        """Straight key: high jitter on everything, per-char speed factor."""
        # Per-character speed: simple chars faster, complex chars slower
        # ±10% at max complexity
        speed_factor = 1.0 + 0.10 * (0.5 - char_cplx) * float(rng.normal(0.0, 1.0))
        speed_factor = max(0.85, min(1.15, speed_factor))
        nominal = units * local_ud * speed_factor
        if is_dah:
            # Per-element dah/dit ratio variation for straight keys
            ratio_jitter = float(rng.normal(0.0, 0.08))  # ±~8% of ratio
            nominal = nominal * (1.0 + ratio_jitter)
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        noise = rng.normal(0.0, timing_jitter * nominal)
        return max(nominal + noise, nominal * 0.1)

    def _jitter_bug(units: float, is_dit: bool, is_dah: bool,
                    char_cplx: float, local_ud: float,
                    local_ddr: float) -> float:
        """Bug: mechanical dits (minimal jitter), manual dahs (moderate jitter)."""
        nominal = units * local_ud
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        if is_dit:
            # Mechanical dits: very consistent (~15% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.15 * nominal)
        elif is_dah:
            # Manual dahs: moderate jitter (~80% of configured jitter)
            # Also per-dah ratio variation
            ratio_jitter = float(rng.normal(0.0, 0.06))
            nominal = nominal * (1.0 + ratio_jitter)
            noise = rng.normal(0.0, timing_jitter * 0.80 * nominal)
        else:
            # Spacing: manual, full jitter
            noise = rng.normal(0.0, timing_jitter * nominal)
        return max(nominal + noise, nominal * 0.1)

    def _jitter_paddle(units: float, is_dit: bool, is_dah: bool,
                       char_cplx: float, local_ud: float,
                       local_ddr: float) -> float:
        """Paddle: electronic elements (minimal jitter), manual spacing."""
        nominal = units * local_ud
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        if is_dit or is_dah:
            # Electronic elements: very consistent (~10% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.10 * nominal)
        else:
            # Manual spacing: moderate jitter (~60% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.60 * nominal)
        return max(nominal + noise, nominal * 0.1)

    def _jitter_cootie(units: float, is_dit: bool, is_dah: bool,
                       char_cplx: float, local_ud: float,
                       local_ddr: float) -> float:
        """Cootie/sideswiper: alternating contacts, symmetric high jitter.

        The operator switches between two paddle contacts for every element.
        There is no mechanical dit/dah distinction — timing is entirely manual.
        Result: symmetric but high jitter on all elements, with occasional
        overshoot where dits are slightly long or dahs slightly short compared
        to the target ratio.
        """
        nominal = units * local_ud
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        # All elements get high jitter (~90% of configured jitter)
        noise = rng.normal(0.0, timing_jitter * 0.90 * nominal)
        # Cootie operators tend to drift the dah/dit ratio toward the middle
        # (dahs shorter than intended, dits longer than intended)
        if is_dah:
            # Dahs tend to be slightly short (ratio compression)
            ratio_drift = float(rng.normal(-0.05, 0.08))
            nominal = nominal * (1.0 + ratio_drift)
        elif is_dit:
            # Dits tend to be slightly long (ratio compression)
            ratio_drift = float(rng.normal(0.03, 0.06))
            nominal = nominal * (1.0 + ratio_drift)
        return max(nominal + noise, nominal * 0.1)

    jitter_fn = {"straight": _jitter_straight,
                 "bug":      _jitter_bug,
                 "paddle":   _jitter_paddle,
                 "cootie":   _jitter_cootie}.get(key_type, _jitter_paddle)

    # ---- Build elements --------------------------------------------------
    elements: List[Element] = []

    _multi_op_factor = 1.0  # persists across word boundaries

    for w_idx, word in enumerate(words):
        # Multi-operator: abrupt speed step at designated word boundaries
        if w_idx in multi_op_changes:
            _multi_op_factor = multi_op_changes[w_idx]

        # Speed drift: modulate unit_dur per word
        if drift_amplitude > 0.0 and n_words > 1:
            phase = drift_phase + drift_freq * (w_idx / max(n_words - 1, 1)) * 2 * math.pi
            local_ud = unit_dur * (1.0 + drift_amplitude * math.sin(phase))
        else:
            local_ud = unit_dur

        # Apply multi-op speed factor
        local_ud = local_ud * _multi_op_factor

        chars: List[str] = [word] if word in MORSE_TABLE else list(word)

        for c_idx, ch in enumerate(chars):
            if ch not in MORSE_TABLE:
                continue
            code = MORSE_TABLE[ch]
            cplx = _char_complexity(code)

            for e_idx, sym in enumerate(code):
                if sym == ".":
                    dur = jitter_fn(1.0, True, False, cplx, local_ud, dah_dit_ratio)
                    elements.append((True, dur))
                elif sym == "-":
                    dur = jitter_fn(dah_dit_ratio, False, True, cplx, local_ud, dah_dit_ratio)
                    elements.append((True, dur))
                # Intra-character gap
                if e_idx < len(code) - 1:
                    dur = jitter_fn(1.0, False, False, cplx, local_ud, dah_dit_ratio)
                    elements.append((False, dur))

            # Inter-character gap (3 × ics_factor dits), stretched by Farnsworth
            if c_idx < len(chars) - 1:
                dur = jitter_fn(3.0 * ics_factor * farnsworth_stretch, False, False, cplx, local_ud, dah_dit_ratio)
                elements.append((False, dur))

        # Inter-word gap (7 × iws_factor dits), stretched by Farnsworth
        if w_idx < len(words) - 1:
            dur = jitter_fn(7.0 * iws_factor * farnsworth_stretch, False, False, 0.0, local_ud, dah_dit_ratio)
            elements.append((False, dur))

    return elements


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

def synthesize_audio(
    elements: List[Element],
    sample_rate: int,
    base_freq: float,
    tone_drift: float,
    snr_db: float,
    rng: np.random.Generator,
    trailing_silence_sec: float = 0.0,
    target_amplitude: float = 0.9,
    agc_depth_db: float = 0.0,
    agc_attack_ms: float = 50.0,
    agc_release_ms: float = 400.0,
    qsb_depth_db: float = 0.0,
    rise_time_ms: float = 5.0,
) -> np.ndarray:
    """Render Morse elements to a float32 audio waveform.

    Pipeline (in order):
      1. Carrier + key envelope → signal (message length only)
      2. QSB: slow sinusoidal fading applied to signal
      3. White AWGN generated for full length (message + trailing silence)
      4. AGC: noise amplitude modulated inversely to signal envelope
      5. Mix signal + noise
      6. Normalise to target_amplitude

    Parameters
    ----------
    elements : list of (is_tone, duration_sec)
    sample_rate : int
    base_freq : float
        Carrier centre frequency (Hz).
    tone_drift : float
        Peak sinusoidal frequency drift (Hz).
    snr_db : float
        Target SNR measured against full-band noise (dB).
    rng : np.random.Generator
    trailing_silence_sec : float
        Seconds of noise-only audio appended after the message.
    target_amplitude : float
        Peak amplitude of normalised output.
    agc_depth_db : float
        Noise suppression during peak marks (dB); 0 = disabled.
    agc_attack_ms, agc_release_ms : float
        AGC attack and release time constants (ms).
    qsb_depth_db : float
        Peak-to-peak sinusoidal fading range (dB); 0 = disabled.

    Returns
    -------
    np.ndarray
        Float32 waveform normalised to target_amplitude.
    """
    msg_duration  = sum(d for _, d in elements)
    msg_samples   = max(1, int(math.ceil(msg_duration * sample_rate)))
    tail_samples  = max(0, int(trailing_silence_sec * sample_rate))
    total_samples = msg_samples + tail_samples

    # ---- Carrier with slow sinusoidal frequency drift (float32) ----------
    t = np.arange(msg_samples, dtype=np.float32) / sample_rate
    drift_rate  = rng.uniform(0.05, 0.2)
    drift_phase = rng.uniform(0.0, 2 * math.pi)
    freq       = (base_freq + tone_drift * np.sin(
        np.float32(2 * math.pi * drift_rate) * t + np.float32(drift_phase)
    )).astype(np.float32)
    inst_phase = np.cumsum(np.float32(2 * math.pi / sample_rate) * freq)
    carrier    = np.sin(inst_phase).astype(np.float32)

    # ---- Key envelope with soft rise/fall (prevents key clicks) ----------
    envelope     = np.zeros(msg_samples, dtype=np.float32)
    rise_samples = max(2, int(rise_time_ms * 0.001 * sample_rate))
    # Precompute the full-size ramp once; slice for shorter tone elements.
    _ramp_full = (np.sin(
        np.linspace(0.0, math.pi / 2, rise_samples, endpoint=False)
    ) ** 2).astype(np.float32)

    pos = 0
    for is_tone, duration in elements:
        n     = int(round(duration * sample_rate))
        end   = min(pos + n, msg_samples)
        chunk = end - pos
        if chunk <= 0:
            break
        if is_tone:
            envelope[pos:end] = 1.0
            r = min(rise_samples, chunk // 2)
            if r > 0:
                ramp = _ramp_full[:r]
                envelope[pos:pos + r] = ramp
                envelope[end - r:end] = ramp[::-1]
        pos = end
        if pos >= msg_samples:
            break

    signal = carrier * envelope  # already float32

    # ---- Noise level from message signal power (before QSB) -------------
    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_std = math.sqrt(sig_power / (10.0 ** (snr_db / 10.0))) if sig_power > 1e-12 else 0.01

    # ---- Extend signal to full length (trailing silence = zeros) --------
    if tail_samples > 0:
        signal = np.concatenate([signal, np.zeros(tail_samples, dtype=np.float32)])

    # ---- QSB: slow sinusoidal signal fading -----------------------------
    if qsb_depth_db > 0.0:
        t_full = np.arange(total_samples, dtype=np.float32) / sample_rate
        signal = _apply_qsb(signal, t_full, rng, qsb_depth_db)

    # ---- White AWGN for full audio --------------------------------------
    noise = rng.normal(0.0, noise_std, total_samples).astype(np.float32)

    # ---- AGC: modulate noise inversely to signal envelope ---------------
    if agc_depth_db > 0.0:
        noise = _agc_noise_modulation(
            signal, noise, sample_rate, agc_attack_ms, agc_release_ms, agc_depth_db,
        )

    # ---- Mix ------------------------------------------------------------
    audio = signal + noise

    # ---- Normalise to target amplitude ----------------------------------
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = (audio * (target_amplitude / peak)).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sample(
    config: MorseConfig,
    wpm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    wordlist: Optional[List[str]] = None,
    text: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, str, Dict]:
    """Generate a single synthetic Morse code audio sample.

    Parameters
    ----------
    config : MorseConfig
        Audio generation parameters.
    wpm : float, optional
        Override WPM; randomised from config if None.
    rng : np.random.Generator, optional
        RNG; a fresh one is created if None.
    wordlist : list of str, optional
        Word list for text generation.
    text : str, optional
        Override text; randomly generated if None.

    Returns
    -------
    audio_f32 : np.ndarray
        Float32 waveform, shape ``(N,)``, normalised to ±1.
    text : str
        Upper-case decoded transcript.
    metadata : dict
        Generation parameters (wpm, snr_db, dah_dit_ratio, ics_factor, …).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- WPM ------------------------------------------------------------
    if wpm is None:
        wpm = float(rng.uniform(config.min_wpm, config.max_wpm))
    unit_dur = 60.0 / (wpm * 50.0)

    # ---- Audio parameters -----------------------------------------------
    base_freq = float(rng.uniform(config.tone_freq_min, config.tone_freq_max))
    snr_db = float(rng.uniform(config.min_snr_db, config.max_snr_db))

    # ---- Timing parameters (sampled independently per sample) -----------
    dah_dit_ratio = float(rng.uniform(config.dah_dit_ratio_min, config.dah_dit_ratio_max))
    ics_factor = float(rng.uniform(config.ics_factor_min, config.ics_factor_max))
    iws_factor = float(rng.uniform(config.iws_factor_min, config.iws_factor_max))

    # ---- Per-sample timing jitter ----------------------------------------
    if config.timing_jitter_max > 0:
        jitter = float(rng.uniform(config.timing_jitter, config.timing_jitter_max))
    else:
        jitter = config.timing_jitter

    # ---- Per-sample key type selection -----------------------------------
    key_type = _select_key_type(config.key_type_weights, rng)

    # ---- Per-sample amplitude target ------------------------------------
    if config.signal_amplitude_min < config.signal_amplitude_max:
        target_amplitude = float(
            rng.uniform(config.signal_amplitude_min, config.signal_amplitude_max)
        )
    else:
        target_amplitude = config.signal_amplitude_max

    # ---- Per-sample AGC decision ----------------------------------------
    agc_depth_db = 0.0
    if config.agc_probability > 0.0 and rng.random() < config.agc_probability:
        agc_depth_db = float(rng.uniform(config.agc_depth_db_min, config.agc_depth_db_max))

    # ---- Per-sample QSB decision ----------------------------------------
    qsb_depth_db = 0.0
    if config.qsb_probability > 0.0 and rng.random() < config.qsb_probability:
        qsb_depth_db = float(rng.uniform(config.qsb_depth_db_min, config.qsb_depth_db_max))

    # ---- Farnsworth timing -----------------------------------------------
    farnsworth_stretch = 1.0
    if (config.farnsworth_probability > 0.0
            and rng.random() < config.farnsworth_probability):
        # Character speed multiplier: characters sent faster, spaces stretched.
        # farnsworth_stretch > 1.0 stretches inter-char and inter-word gaps.
        char_speed_mult = float(rng.uniform(
            config.farnsworth_char_speed_min, config.farnsworth_char_speed_max))
        # Characters are sent at char_speed_mult × wpm, so unit_dur shrinks
        # by 1/char_speed_mult.  To keep the *effective* WPM constant, we
        # must stretch the spacing by char_speed_mult.
        unit_dur = unit_dur / char_speed_mult
        farnsworth_stretch = char_speed_mult

    # ---- Text generation -------------------------------------------------
    if text is None:
        text = generate_text(
            rng, min_chars=config.min_chars, max_chars=config.max_chars, wordlist=wordlist,
        )

    # ---- Multi-operator speed change decision ------------------------------
    multi_op_range = None
    if (config.multi_op_probability > 0.0
            and rng.random() < config.multi_op_probability):
        multi_op_range = (config.multi_op_speed_change_min,
                          config.multi_op_speed_change_max)

    # ---- Build Morse elements -------------------------------------------
    elements = text_to_elements(
        text, unit_dur, jitter, rng,
        dah_dit_ratio=dah_dit_ratio,
        ics_factor=ics_factor,
        iws_factor=iws_factor,
        key_type=key_type,
        speed_drift_max=config.speed_drift_max,
        farnsworth_stretch=farnsworth_stretch,
        multi_op_speed_range=multi_op_range,
    )
    if not elements:
        text = "E"
        elements = text_to_elements(
            text, unit_dur, jitter, rng,
            dah_dit_ratio=dah_dit_ratio,
            ics_factor=ics_factor,
            iws_factor=iws_factor,
            key_type=key_type,
            speed_drift_max=config.speed_drift_max,
            farnsworth_stretch=farnsworth_stretch,
            multi_op_speed_range=multi_op_range,
        )

    # ---- Noise std for silence periods ----------------------------------
    # Approximation based on target amplitude; matches synthesize_audio().
    sig_power = 0.5 * target_amplitude ** 2
    noise_std = math.sqrt(sig_power / (10.0 ** (snr_db / 10.0))) if sig_power > 1e-12 else 0.01

    # ---- WPM-based silence durations ------------------------------------
    # Both leading and trailing silence are randomised in
    # [one dah, two inter-word spaces] at the chosen WPM and timing params.
    # This ensures silence always looks like at least a recognisable space
    # to the feature extractor but is not excessively long at slow speeds.
    min_silence_sec = dah_dit_ratio * unit_dur               # one dah
    max_silence_sec = 2.0 * 7.0 * iws_factor * unit_dur     # two word gaps
    leading_sec  = float(rng.uniform(min_silence_sec, max_silence_sec))
    trailing_sec = float(rng.uniform(min_silence_sec, max_silence_sec))

    # ---- Early duration check (bail out before expensive synthesis) ------
    if max_duration_sec is not None:
        element_dur = sum(dur for _, dur in elements)
        if leading_sec + element_dur + trailing_sec > max_duration_sec:
            raise ValueError("sample exceeds max_duration_sec")

    # ---- Per-sample keying waveform shaping --------------------------------
    rise_time_ms = float(rng.uniform(config.rise_time_ms_min, config.rise_time_ms_max))

    # ---- Synthesise audio -----------------------------------------------
    audio_f32 = synthesize_audio(
        elements=elements,
        sample_rate=config.sample_rate,
        base_freq=base_freq,
        tone_drift=config.tone_drift,
        snr_db=snr_db,
        rng=rng,
        trailing_silence_sec=trailing_sec,
        target_amplitude=target_amplitude,
        agc_depth_db=agc_depth_db,
        agc_attack_ms=config.agc_attack_ms,
        agc_release_ms=config.agc_release_ms,
        qsb_depth_db=qsb_depth_db,
        rise_time_ms=rise_time_ms,
    )

    # ---- Prepend leading silence ----------------------------------------
    leading_samples = int(leading_sec * config.sample_rate)
    leading_noise = rng.normal(0.0, noise_std, leading_samples).astype(np.float32)
    audio_f32 = np.concatenate([leading_noise, audio_f32])

    # ---- QRM: interfering CW signals ------------------------------------
    has_qrm = False
    qrm_count = 0
    if config.qrm_probability > 0.0 and rng.random() < config.qrm_probability:
        qrm_count = int(rng.integers(config.qrm_count_min, config.qrm_count_max + 1))
        audio_f32 = _apply_qrm(
            audio_f32, config.sample_rate, rng,
            n_interferers=qrm_count,
            base_freq=base_freq,
            freq_offset_min=config.qrm_freq_offset_min,
            freq_offset_max=config.qrm_freq_offset_max,
            amplitude_min=config.qrm_amplitude_min,
            amplitude_max=config.qrm_amplitude_max,
            duration_sec=len(audio_f32) / config.sample_rate,
        )
        has_qrm = True

    # ---- QRN: impulsive atmospheric noise --------------------------------
    has_qrn = False
    if config.qrn_probability > 0.0 and rng.random() < config.qrn_probability:
        qrn_rate = float(rng.uniform(config.qrn_rate_min, config.qrn_rate_max))
        sig_rms = float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
        audio_f32 = _apply_qrn(
            audio_f32, config.sample_rate, rng,
            rate=qrn_rate,
            duration_ms_min=config.qrn_duration_ms_min,
            duration_ms_max=config.qrn_duration_ms_max,
            amplitude_min=config.qrn_amplitude_min,
            amplitude_max=config.qrn_amplitude_max,
            signal_rms=sig_rms,
        )
        has_qrn = True

    # ---- Real HF noise: mix with existing audio -------------------------
    has_hf_noise = False
    if config.hf_noise_probability > 0.0 and rng.random() < config.hf_noise_probability:
        hf_seg = _get_hf_noise_segment(
            config.hf_noise_dir, len(audio_f32), rng, config.sample_rate,
        )
        if hf_seg is not None:
            # Scale HF noise to match the existing noise level, then mix
            hf_rms = float(np.sqrt(np.mean(hf_seg.astype(np.float64) ** 2)))
            if hf_rms > 1e-9:
                hf_seg = hf_seg * (noise_std / hf_rms)
            mix = config.hf_noise_mix_ratio
            # Replace fraction of existing audio noise with HF noise
            audio_f32 = audio_f32 + mix * hf_seg
            has_hf_noise = True

    # ---- Bandpass filter: simulate CW receiver filter --------------------
    has_bandpass = False
    bandpass_bw = 0.0
    if config.bandpass_probability > 0.0 and rng.random() < config.bandpass_probability:
        bandpass_bw = float(rng.uniform(config.bandpass_bw_min, config.bandpass_bw_max))
        bandpass_order = int(rng.integers(config.bandpass_order_min, config.bandpass_order_max + 1))
        audio_f32 = _apply_bandpass(
            audio_f32, config.sample_rate, base_freq, bandpass_bw,
            order=bandpass_order,
        )
        has_bandpass = True

    # ---- Final normalisation after all augmentations --------------------
    peak = float(np.max(np.abs(audio_f32)))
    if peak > 1e-9:
        audio_f32 = (audio_f32 * (target_amplitude / peak)).astype(np.float32)

    # ---- Random input gain (post-normalisation) -------------------------
    # Simulates non-peak-normalised input at inference time. Drawn
    # log-uniformly in dB, applied as a linear gain, then clipped to the
    # valid float32 audio range. Disabled by default ((0.0, 0.0)).
    lo, hi = config.input_gain_db_range
    if lo != 0.0 or hi != 0.0:
        gain_db = rng.uniform(lo, hi)
        gain_lin = 10.0 ** (gain_db / 20.0)
        audio_f32 = (audio_f32 * gain_lin).astype(np.float32)
        np.clip(audio_f32, -1.0, 1.0, out=audio_f32)

    metadata: Dict = {
        "wpm": wpm,
        "snr_db": snr_db,
        "base_frequency_hz": base_freq,
        "frequency_drift_hz": config.tone_drift,
        "duration_sec": len(audio_f32) / config.sample_rate,
        "timing_jitter": jitter,
        "dah_dit_ratio": dah_dit_ratio,
        "ics_factor": ics_factor,
        "iws_factor": iws_factor,
        "key_type": key_type,
        "target_amplitude": target_amplitude,
        "agc_depth_db": agc_depth_db,
        "qsb_depth_db": qsb_depth_db,
        "leading_silence_sec": leading_sec,
        "trailing_silence_sec": trailing_sec,
        "farnsworth_stretch": farnsworth_stretch,
        "qrm": has_qrm,
        "qrm_count": qrm_count,
        "qrn": has_qrn,
        "hf_noise": has_hf_noise,
        "bandpass": has_bandpass,
        "bandpass_bw": bandpass_bw,
        "multi_op": multi_op_range is not None,
    }
    return audio_f32, text, metadata




# ---------------------------------------------------------------------------
# CLI entry point — generate test audio files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Generate test Morse audio samples")
    parser.add_argument("--n", type=int, default=3, help="Number of samples")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--wpm", type=float, default=None, help="Override WPM")
    args = parser.parse_args()

    from config import create_default_config
    cfg = create_default_config("clean").morse
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(args.n):
        audio, text, meta = generate_sample(cfg, wpm=args.wpm, rng=rng)
        wav_path = out_dir / f"morse_{i:02d}.wav"
        sf.write(str(wav_path), audio, cfg.sample_rate)
        print(
            f"[{i:02d}] {wav_path}  |  {meta['wpm']:.1f} WPM  |  "
            f"{meta['snr_db']:.1f} dB  |  dah/dit={meta['dah_dit_ratio']:.2f}  |  "
            f"ics={meta['ics_factor']:.2f}  iws={meta['iws_factor']:.2f}  |  "
            f"{text!r}"
        )
