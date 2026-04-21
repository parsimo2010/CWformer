"""
dataset_audio.py — Audio-level dataset for CW-Former training.

Generates (audio, text) pairs on-the-fly using morse_generator.py.
The CW-Former model handles mel spectrogram computation internally,
so this dataset outputs raw audio waveforms.

Outputs raw audio (float32 waveforms), pads along the sample dimension,
and the collate function produces (B, N) audio tensors. QSO corpus text
is mixed with random text (configurable ratio).
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import vocab as vocab_module
from config import Config
from morse_generator import generate_sample, generate_text, load_wordlist
from qso_corpus import QSOCorpusGenerator


def _max_chars_for_wpm(wpm: float, max_audio_sec: float) -> int:
    """Estimate max text length that fits within max_audio_sec at given WPM.

    PARIS standard: 50 units per word, ~6 chars per word-slot (incl. space),
    so ~10/WPM seconds per character.  A 25% safety margin accounts for
    leading/trailing silence, dah/dit ratio variance, and ICS/IWS factors.
    """
    chars_per_sec = wpm / 10.0
    return max(8, int(max_audio_sec * chars_per_sec * 0.75))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(IterableDataset):
    """Streaming dataset producing (audio, target, text) tuples for CW-Former.

    Generates synthetic Morse audio samples on-the-fly with all configured
    augmentations (AGC, QSB, QRM, QRN, bandpass, HF noise, etc.).

    Args:
        config: Full pipeline configuration.
        epoch_size: Number of samples per epoch.
        seed: Fixed seed for reproducibility (None = random per worker).
        qso_text_ratio: Fraction of samples using QSO corpus text (0-1).
        max_audio_sec: Maximum audio duration in seconds (longer samples truncated).
    """

    def __init__(
        self,
        config: Config,
        epoch_size: int,
        seed: Optional[int] = None,
        qso_text_ratio: float = 0.5,
        max_audio_sec: float = 30.0,
    ) -> None:
        super().__init__()
        self.morse_cfg = config.morse
        self.epoch_size = epoch_size
        self.seed = seed
        self.qso_text_ratio = qso_text_ratio
        self.max_audio_sec = max_audio_sec
        self.max_audio_samples = int(max_audio_sec * config.morse.sample_rate)
        self.wordlist = load_wordlist()
        self._space_idx = vocab_module.char_to_idx[" "]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, str]]:
        rng = self._make_rng()
        qso_gen = QSOCorpusGenerator(seed=int(rng.integers(0, 2**31)))

        worker_info = torch.utils.data.get_worker_info()
        per_worker = (
            math.ceil(self.epoch_size / worker_info.num_workers)
            if worker_info is not None
            else self.epoch_size
        )

        yielded = 0
        while yielded < per_worker:
            try:
                # Sample WPM first so we can constrain text length to
                # what will actually fit within max_audio_sec.  Without
                # this, most low-WPM + long-text combinations are rejected,
                # wasting worker time even with the early bail-out.
                wpm = float(rng.uniform(
                    self.morse_cfg.min_wpm, self.morse_cfg.max_wpm))
                max_chars = _max_chars_for_wpm(wpm, self.max_audio_sec)

                use_qso = float(rng.random()) < self.qso_text_ratio
                if use_qso:
                    text = qso_gen.generate(
                        min_len=min(8, max_chars), max_len=max_chars)
                else:
                    text = generate_text(
                        rng,
                        min_chars=min(self.morse_cfg.min_chars, max_chars),
                        max_chars=max_chars,
                        wordlist=self.wordlist,
                    )

                audio_f32, text, _ = generate_sample(
                    self.morse_cfg, rng=rng, wordlist=self.wordlist,
                    text=text, wpm=wpm,
                    max_duration_sec=self.max_audio_sec,
                )
            except Exception:
                continue

            if len(audio_f32) < 1600:  # less than 100ms
                continue

            # Skip samples that exceed max duration — truncating audio
            # while keeping the full text target creates mismatched labels
            # that corrupt CTC gradients.
            if len(audio_f32) > self.max_audio_samples:
                continue

            # Target with boundary space tokens
            inner = vocab_module.encode(text)
            if not inner:
                continue
            target_indices = [self._space_idx] + inner + [self._space_idx]
            target_tensor = torch.tensor(target_indices, dtype=torch.long)

            audio_tensor = torch.from_numpy(audio_f32)

            yield audio_tensor, target_tensor, text
            yielded += 1

    def __len__(self) -> int:
        return self.epoch_size

    def _make_rng(self) -> np.random.Generator:
        if self.seed is not None:
            return np.random.default_rng(self.seed)
        # Use OS entropy so each epoch generates fresh data.
        # worker_info.seed is fixed per-worker lifetime with persistent_workers=True,
        # so using it causes identical training data every epoch (genuine overfitting).
        return np.random.default_rng()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Tensor, Tensor, str]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[str]]:
    """Pad audio batch into fixed tensors.

    Returns
    -------
    audio_padded : Tensor, shape (B, max_N) — zero-padded audio
    targets_padded : Tensor, shape (B, max_S) — zero-padded target indices
    audio_lengths : Tensor, shape (B,) — actual audio sample counts
    target_lengths : Tensor, shape (B,) — actual target lengths
    texts : list of str — transcript strings
    """
    audios, targets, texts = zip(*batch)

    audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    B = len(audios)
    max_n = int(audio_lengths.max().item())
    max_s = int(target_lengths.max().item())

    audio_padded = torch.zeros(B, max_n, dtype=torch.float32)
    for i, a in enumerate(audios):
        audio_padded[i, :a.shape[0]] = a

    targets_padded = torch.zeros(B, max_s, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_padded[i, :t.shape[0]] = t

    return audio_padded, targets_padded, audio_lengths, target_lengths, list(texts)
