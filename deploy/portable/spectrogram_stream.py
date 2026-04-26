"""Mel-spectrogram PNG streaming for the web UI.

Maintains a rolling history of mel frames + tracked decoded characters
and renders a small PNG (default 480x140) on demand. Characters are
plotted as ticks aligned with the mel frame where greedy CTC first
emitted them, mirroring the gui.py overlay.
"""
from __future__ import annotations

import base64
import io
import threading
from typing import List, Optional, Tuple

import numpy as np


def _load_overlay_font(size: int = 14):
    """Pick a readable bold mono font, with platform fallbacks.

    The previous hard-coded DejaVu path only resolved on Debian/Pi. On
    Windows / macOS the call fell through to ``ImageFont.load_default()``
    which returns a tiny pixel font, making the char overlay nearly
    illegible. Try a small list of common bold mono fonts first.
    """
    try:
        from PIL import ImageFont  # type: ignore
    except ImportError:
        return None

    candidates = [
        # Linux / Pi
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/Library/Fonts/Andale Mono.ttf",
        # Windows
        "C:/Windows/Fonts/consolab.ttf",   # Consolas Bold
        "C:/Windows/Fonts/consola.ttf",    # Consolas
        "C:/Windows/Fonts/courbd.ttf",     # Courier New Bold
        "C:/Windows/Fonts/cour.ttf",       # Courier New
        "C:/Windows/Fonts/lucon.ttf",      # Lucida Console
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        # Pillow >= 10 supports a size argument on the default bitmap font.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
    except Exception:
        return None


_FONT_CACHE: dict = {}


def _overlay_font(size: int = 14):
    f = _FONT_CACHE.get(size)
    if f is None:
        f = _load_overlay_font(size)
        _FONT_CACHE[size] = f
    return f


def _viridis_lut() -> np.ndarray:
    """Procedural perceptual colormap (256x3 uint8). Matplotlib-free."""
    n = 256
    t = np.linspace(0.0, 1.0, n)
    r = (0.267 + 1.4 * (t - 0.6) ** 2 - 0.4 * (1.0 - t) ** 1.7).clip(0.0, 1.0)
    g = (0.05 + 1.55 * t - 0.7 * t ** 2).clip(0.0, 1.0)
    b = (0.32 + 1.4 * (1.0 - t) ** 1.5 - 0.6 * (1.0 - t) ** 4).clip(0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


_LUT: Optional[np.ndarray] = None


def _lut() -> np.ndarray:
    global _LUT
    if _LUT is None:
        _LUT = _viridis_lut()
    return _LUT


class SpectrogramStream:

    def __init__(
        self,
        n_mels: int = 40,
        history_frames: int = 800,
        width: int = 480,
        height: int = 140,
    ) -> None:
        self.n_mels = n_mels
        self.history = history_frames
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._buf = np.zeros((0, n_mels), dtype=np.float32)
        self._chars: List[Tuple[int, str]] = []
        self._total_frames = 0
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = on

    def reset(self) -> None:
        with self._lock:
            self._buf = np.zeros((0, self.n_mels), dtype=np.float32)
            self._chars = []
            self._total_frames = 0

    def add_frames(self, mel_frames: np.ndarray) -> None:
        """``mel_frames`` shape: (T, n_mels), audio-rate frames (~100 fps)."""
        if mel_frames.ndim != 2 or mel_frames.shape[0] == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, mel_frames], axis=0)
            self._total_frames += mel_frames.shape[0]
            if self._buf.shape[0] > self.history:
                self._buf = self._buf[-self.history:]

    def add_chars(
        self,
        new_chars: List[Tuple[str, int]],
        chunk_first_ctc_frame: int,
        ctc_to_audio_ratio: int = 2,
    ) -> None:
        """Append (char, ctc_frame_idx_within_chunk) tuples. CTC frames at
        50 fps; mel display is 100 fps, hence the 2x ratio."""
        with self._lock:
            for ch, ctc_frame in new_chars:
                audio_frame = (chunk_first_ctc_frame + ctc_frame) * ctc_to_audio_ratio
                self._chars.append((audio_frame, ch))
            # Keep at most ~5x the visible window of chars (off-screen ones
            # cost nothing to skip, but unbounded growth would over time).
            cutoff = self._total_frames - self.history * 5
            if self._chars and self._chars[0][0] < cutoff:
                self._chars = [(f, c) for f, c in self._chars if f >= cutoff]

    def render_base64(self) -> Optional[str]:
        if not self._enabled:
            return None
        png = self._render_png()
        if png is None:
            return None
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    def _render_png(self) -> Optional[bytes]:
        with self._lock:
            buf = self._buf.copy()
            chars = list(self._chars)
            total_frames = self._total_frames

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        text_band = 22
        spec_h = self.height - text_band

        canvas = Image.new("RGB", (self.width, self.height), (12, 12, 18))

        if buf.shape[0] > 0:
            lo, hi = np.percentile(buf, [5.0, 99.0])
            if hi - lo < 1.0:
                hi = lo + 1.0
            norm = ((buf - lo) / (hi - lo)).clip(0.0, 1.0)
            idx = (norm * 255).astype(np.uint8)
            rgb = _lut()[idx]                       # (T, n_mels, 3)
            rgb = np.transpose(rgb, (1, 0, 2))[::-1]  # (n_mels, T, 3), low->bottom
            spec_img = Image.fromarray(rgb, mode="RGB").resize(
                (self.width, spec_h), Image.NEAREST)
            canvas.paste(spec_img, (0, text_band))

        draw = ImageDraw.Draw(canvas)
        font = _overlay_font(14)

        first_visible = max(0, total_frames - buf.shape[0])
        visible = max(1, buf.shape[0])
        for frame_abs, ch in chars:
            if frame_abs < first_visible or frame_abs >= total_frames:
                continue
            x = int((frame_abs - first_visible) / visible * self.width)
            x = min(max(x, 0), self.width - 1)
            draw.line([(x, text_band), (x, self.height - 1)],
                      fill=(255, 240, 120), width=1)
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw, _ = draw.textsize(ch, font=font)
            tx = max(0, min(self.width - tw, x - tw // 2))
            draw.text((tx, 2), ch, fill=(255, 245, 180), font=font)

        out = io.BytesIO()
        canvas.save(out, format="PNG", optimize=False, compress_level=1)
        return out.getvalue()
