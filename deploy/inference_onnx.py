#!/usr/bin/env python3
"""
inference_onnx.py -- Streaming ONNX Runtime inference for causal CW-Former.

Runs the INT8 or FP32 ONNX model with numpy-based mel spectrogram
computation and streaming state management. No PyTorch required at runtime.

Dependencies: numpy, soundfile, onnxruntime
Optional:     sounddevice (for --device), scipy (for resampling)

Usage::

    # Decode a file
    python deploy/inference_onnx.py --model deploy/cwformer_streaming_int8.onnx \\
        --input recordings/test.wav

    # Live from audio device
    python deploy/inference_onnx.py --model deploy/cwformer_streaming_int8.onnx \\
        --device

    # SDR pipe (stdin, raw 16-bit PCM at 16 kHz mono)
    rtl_fm -f 7.030M -M usb -s 16000 | \\
        python deploy/inference_onnx.py --model deploy/cwformer_streaming_int8.onnx --stdin
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary (self-contained -- no project imports needed for greedy decode)
# ---------------------------------------------------------------------------

def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = (
        ["<blank>"]
        + [" "]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [str(d) for d in range(10)]
        + list(".,?/(&=+")
        + ["AR", "SK", "BT", "KN", "AS", "CT"]
    )
    c2i = {tok: idx for idx, tok in enumerate(tokens)}
    i2c = {idx: tok for idx, tok in enumerate(tokens)}
    return c2i, i2c


CHAR_TO_IDX, IDX_TO_CHAR = _build_vocab()
NUM_CLASSES = len(CHAR_TO_IDX)
BLANK_IDX = 0


def greedy_ctc_decode(
    log_probs: np.ndarray, strip_boundary_spaces: bool = True,
) -> str:
    """Greedy CTC decode in pure numpy.  log_probs shape: (T, C)."""
    indices = np.argmax(log_probs, axis=-1)
    collapsed = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(int(idx))
            prev = idx
    text = "".join(IDX_TO_CHAR.get(i, "") for i in collapsed if i != BLANK_IDX)
    if strip_boundary_spaces:
        text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Mel spectrogram (pure numpy -- no PyTorch / torchaudio)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _create_mel_filterbank(
    n_fft: int, sample_rate: int, n_mels: int,
    f_min: float, f_max: float,
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2+1)."""
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - low) / max(center - low, 1e-10)
        down = (high - fft_freqs) / max(high - center, 1e-10)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


class MelComputer:
    """Numpy-based mel spectrogram with streaming support."""

    def __init__(self, config: dict, config_dir: Optional[str] = None) -> None:
        self.n_fft = config["n_fft"]
        self.hop = config["hop_length"]
        self.n_mels = config["n_mels"]
        self.sample_rate = config["sample_rate"]

        # Load the Hann window and mel filterbank tables that were saved
        # next to mel_config.json at export time, if present. They are
        # the bit-exact tables the model was trained with — recomputing
        # them here in numpy (FP64 intermediates) diverges from torch's
        # FP32 construction at boundary FFT bins, producing mel values
        # ~10 orders of magnitude below what training saw. That breaks
        # silent-frame features and causes inter-word-space detection to
        # fail. See quantize_cwformer.py and diag_bin0_frame0.py.
        tables_loaded = False
        if config_dir is not None:
            basis_path = Path(config_dir) / "mel_basis.npy"
            window_path = Path(config_dir) / "mel_window.npy"
            if basis_path.exists() and window_path.exists():
                self.mel_basis = np.load(basis_path).astype(np.float32)
                self.window = np.load(window_path).astype(np.float32)
                tables_loaded = True

        if not tables_loaded:
            # Legacy fallback — only used if the saved tables aren't
            # available. Do NOT rely on this for deployment parity.
            self.window = (
                0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.n_fft) / self.n_fft))
            ).astype(np.float32)
            self.mel_basis = _create_mel_filterbank(
                self.n_fft, self.sample_rate, self.n_mels,
                config["f_min"], config["f_max"],
            )

    def compute(self, audio: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compute log-mel spectrogram (full audio, non-streaming).

        Returns:
            mel: (1, T, n_mels) float32 array.
            n_frames: number of mel frames.
        """
        pad = self.n_fft // 2
        audio_padded = np.pad(audio, (pad, pad)).astype(np.float32)

        n_frames = (len(audio_padded) - self.n_fft) // self.hop + 1
        shape = (n_frames, self.n_fft)
        strides = (audio_padded.strides[0] * self.hop, audio_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape,
                                                  strides=strides)

        windowed = frames * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        power = np.abs(spec) ** 2

        mel = power @ self.mel_basis.T
        mel = np.log(mel + 1e-6).astype(np.float32)

        return mel[np.newaxis, :, :], n_frames

    def compute_streaming(
        self, audio_chunk: np.ndarray, stft_buffer: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mel for an audio chunk with STFT overlap buffer.

        Mirrors the PyTorch logic at
        ``neural_decoder/mel_frontend.py:compute_streaming``:

        - On the very first chunk (``stft_buffer is None``) left-pad by
          ``n_fft // 2`` so the STFT frame grid starts at the same sample
          as the full-forward ``forward()`` path.
        - No per-chunk right-padding. Training right-pads only once at
          the end of the full utterance; padding every chunk invents
          spectral frames over zeros that don't exist in the stream.
        - The carry-forward buffer is ``audio[consumed_up_to:]``, where
          ``consumed_up_to = n_frames * hop``. This is exactly the
          portion of ``audio`` whose samples have not yet been covered
          by a completed STFT frame — the next frame will start there.

        Returns:
            mel: (1, T, n_mels) float32 array.
            new_buffer: carry-forward samples for next call.
        """
        if stft_buffer is not None:
            audio = np.concatenate([stft_buffer, audio_chunk]).astype(np.float32)
        else:
            # First chunk: left-pad by n_fft//2 to match forward()'s grid
            audio = np.pad(audio_chunk, (self.n_fft // 2, 0)).astype(np.float32)

        audio_len = len(audio)
        n_frames = (audio_len - self.n_fft) // self.hop + 1 if audio_len >= self.n_fft else 0

        # Carry forward every sample that hasn't been covered by a full
        # completed frame. The last completed frame covers samples
        # [(n_frames-1)*hop, (n_frames-1)*hop + n_fft). The next frame
        # would start at n_frames*hop — so everything from that index
        # onward must carry to the next call.
        if n_frames > 0:
            consumed_up_to = n_frames * self.hop
            new_buffer = audio[consumed_up_to:].copy()
        else:
            # Not enough samples for a single frame — carry everything.
            new_buffer = audio.copy()

        if n_frames <= 0:
            return np.zeros((1, 0, self.n_mels), dtype=np.float32), new_buffer

        # STFT only over the portion covered by complete frames.
        stft_len = (n_frames - 1) * self.hop + self.n_fft
        audio_for_stft = audio[:stft_len]

        shape = (n_frames, self.n_fft)
        strides = (audio_for_stft.strides[0] * self.hop, audio_for_stft.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio_for_stft, shape=shape,
                                                  strides=strides)

        windowed = frames * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        power = np.abs(spec) ** 2

        mel = power @ self.mel_basis.T
        mel = np.log(mel + 1e-6).astype(np.float32)

        return mel[np.newaxis, :, :], new_buffer


# ---------------------------------------------------------------------------
# Audio loading + device capture
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, return float32 mono."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    return audio


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n_out = int(len(audio) * sr_out / sr_in)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
    """Peak-normalize audio to match training distribution.

    ``morse_generator.generate_sample`` normalizes every training sample to
    ``target_amplitude ∈ [0.5, 0.9]``; file decode must match or the log-mel
    feature distribution at the subsample input drifts out of distribution.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-8:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def list_devices() -> str:
    """List available audio input devices."""
    try:
        import sounddevice as sd
    except ImportError:
        return "[sounddevice not installed: pip install sounddevice]"
    devices = sd.query_devices()
    lines = ["Available audio input devices:", ""]
    for i, d in enumerate(devices):
        if d["max_input_channels"] < 1:
            continue
        default = " *" if i == sd.default.device[0] else "  "
        lines.append(
            f"{default}{i:>3}  {d['name']:<45} "
            f"{int(d['default_samplerate']):>6} Hz"
        )
    lines.append("")
    lines.append("  * = default input device")
    return "\n".join(lines)


def device_stream(
    target_sr: int,
    device: Optional[int] = None,
    chunk_ms: float = 100.0,
) -> Generator[np.ndarray, None, None]:
    """Yield float32 mono chunks from a sounddevice input."""
    import sounddevice as sd

    dev_info = sd.query_devices(device, "input")
    dev_sr = int(dev_info["default_samplerate"])
    chunk_size = max(1, int(dev_sr * chunk_ms / 1000.0))

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

    def _callback(indata, frames, time_info, status):
        try:
            q.put_nowait(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())
        except queue.Full:
            pass

    stream = sd.InputStream(
        device=device, channels=1, samplerate=dev_sr,
        blocksize=chunk_size, dtype="float32", callback=_callback,
    )
    stream.start()
    try:
        while True:
            try:
                chunk = q.get(timeout=1.0)
                if dev_sr != target_sr:
                    chunk = _resample(chunk.flatten(), dev_sr, target_sr)
                else:
                    chunk = chunk.flatten()
                yield chunk
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Callsign detection
# ---------------------------------------------------------------------------

_CALLSIGN_RE = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z]{1,3})\b"
    r"|"
    r"\b(\d[A-Z]\d[A-Z]{1,3})\b"
)


def detect_callsigns(text: str) -> List[str]:
    return [m.group(0) for m in _CALLSIGN_RE.finditer(text.upper())]


# ---------------------------------------------------------------------------
# Live display (ANSI terminal)
# ---------------------------------------------------------------------------

class LiveDisplay:
    """Rolling terminal display for live CW decoding."""

    def __init__(self, max_text_lines: int = 8, status: str = "") -> None:
        self._max_lines = max_text_lines
        self._status = status
        self._callsign = ""
        self._prev_rendered = 0
        self._out = sys.stderr

    def update(self, text: str) -> None:
        width = shutil.get_terminal_size((80, 24)).columns

        calls = detect_callsigns(text)
        if calls:
            self._callsign = calls[-1]

        lines = self._wrap(text, width - 1)
        visible = lines[-self._max_lines:]

        if self._prev_rendered > 0:
            self._out.write(f"\033[{self._prev_rendered}A\033[J")

        call_str = self._callsign if self._callsign else "----"
        header = f"  DE {call_str}  |  {self._status}"
        separator = "-" * min(len(header) + 4, width)
        self._out.write(f"\033[1m{header}\033[0m\n")
        self._out.write(f"{separator}\n")

        for line in visible:
            self._out.write(line + "\n")

        self._prev_rendered = 2 + len(visible)
        self._out.flush()

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        if not text:
            return [""]
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = (current + " " + word) if current else word
        if current:
            lines.append(current)
        return lines or [""]


# ---------------------------------------------------------------------------
# Streaming ONNX Decoder
# ---------------------------------------------------------------------------

class CWFormerStreamingONNX:
    """Streaming CW decoder using ONNX Runtime. No PyTorch required.

    Processes audio chunk-by-chunk with explicit state management.
    No windowing, no stitching — true causal streaming.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        chunk_ms: int = 500,
        max_cache_sec: float = 30.0,
    ) -> None:
        import onnxruntime as ort

        if config_path is None:
            config_path = str(Path(model_path).parent / "mel_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.sample_rate = self.config["sample_rate"]
        self.mel = MelComputer(self.config, config_dir=str(Path(config_path).parent))
        self.chunk_ms = chunk_ms
        self._chunk_samples = int(chunk_ms * self.sample_rate / 1000)

        # Model config
        self._n_layers = self.config.get("n_layers", 12)
        self._n_heads = self.config.get("n_heads", 4)
        self._d_model = self.config.get("d_model", 256)
        self._d_k = self.config.get("d_k", 64)
        self._conv_kernel = self.config.get("conv_kernel", 63)
        self._conv_pad = self._conv_kernel - 1
        self._n_mels = self.config["n_mels"]
        self._subsample_channels = self.config.get("subsample_channels", 256)
        self._max_cache_len = int(max_cache_sec * 50)  # ~50fps after 2x subsample
        self._freq1 = math.ceil(self._n_mels / 2)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])

        self.reset()

    def reset(self) -> None:
        """Reset all streaming state."""
        self._state = self._init_state()
        self._stft_buffer: Optional[np.ndarray] = None
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._all_log_probs: List[np.ndarray] = []
        self._emitted_text = ""

    def _init_state(self) -> Dict[str, np.ndarray]:
        """Create zero-initialized state tensors."""
        state = {}
        state["pos_offset"] = np.array([0], dtype=np.int64)

        for i in range(self._n_layers):
            state[f"kv_k_layer{i}"] = np.zeros(
                (1, self._n_heads, 0, self._d_k), dtype=np.float32)
            state[f"kv_v_layer{i}"] = np.zeros(
                (1, self._n_heads, 0, self._d_k), dtype=np.float32)
            state[f"conv_buf_layer{i}"] = np.zeros(
                (1, self._d_model, self._conv_pad), dtype=np.float32)

        state["sub_buf1"] = np.zeros(
            (1, 1, 2, self._n_mels), dtype=np.float32)
        state["sub_buf2"] = np.zeros(
            (1, self._subsample_channels, 2, self._freq1), dtype=np.float32)

        return state

    def feed_audio(self, audio_chunk: np.ndarray) -> str:
        """Feed audio chunk, return new decoded characters."""
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        new_text = ""
        while len(self._audio_buffer) >= self._chunk_samples:
            chunk = self._audio_buffer[:self._chunk_samples]
            self._audio_buffer = self._audio_buffer[self._chunk_samples:]
            new_text += self._process_chunk(chunk)

        return new_text

    def get_full_text(self) -> str:
        """Get all decoded text so far."""
        if not self._all_log_probs:
            return self._emitted_text
        all_lp = np.concatenate(self._all_log_probs, axis=0)
        return greedy_ctc_decode(all_lp)

    def flush(self) -> str:
        """Process remaining audio. Call at end of stream.

        Right-pads the final chunk with ``n_fft // 2`` zeros so the tail
        frame count matches training's bilateral pad in
        ``MelFrontend.forward``. Without this, the last 1–2 mel frames
        that training saw are missing, truncating the CTC tail.
        """
        if len(self._audio_buffer) > 0:
            pad_right = self.mel.n_fft // 2
            chunk = np.concatenate(
                [self._audio_buffer,
                 np.zeros(pad_right, dtype=np.float32)]
            )
            self._audio_buffer = np.zeros(0, dtype=np.float32)
            return self._process_chunk(chunk)
        return ""

    def decode_file(self, path: str) -> str:
        audio = load_audio(path, self.sample_rate)
        return self.decode_audio(audio)

    def decode_audio(self, audio: np.ndarray) -> str:
        """Decode a complete audio array via streaming chunks.

        Peak-normalizes to 0.7 so the input amplitude distribution matches
        what the model saw during training. Live streams (``decode_live``)
        do NOT normalize — the caller owns live-audio gain.
        """
        self.reset()
        audio = _peak_normalize(audio, target_peak=0.7)
        pos = 0
        while pos < len(audio):
            end = min(pos + self._chunk_samples, len(audio))
            self.feed_audio(audio[pos:end])
            pos = end
        self.flush()
        return self.get_full_text()

    def decode_live(
        self,
        audio_source: Generator[np.ndarray, None, None],
        display: Optional[LiveDisplay] = None,
    ) -> None:
        """Stream from a live audio source with incremental display."""
        self.reset()
        try:
            for chunk in audio_source:
                new_chars = self.feed_audio(chunk)

                if display is not None:
                    display.update(self.get_full_text())
                elif new_chars:
                    print(new_chars, end="", flush=True, file=sys.stderr)

        except KeyboardInterrupt:
            pass

        self.flush()
        final = self.get_full_text()
        if final:
            print(f"\n{final}", flush=True)

    # ---- Internal ----

    def _process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process one audio chunk through the ONNX model."""
        # Compute mel
        mel, self._stft_buffer = self.mel.compute_streaming(
            audio_chunk, self._stft_buffer)

        if mel.shape[1] == 0:
            return ""

        # Build ONNX feed dict
        feed = {"mel_chunk": mel}
        feed["pos_offset"] = self._state["pos_offset"]
        for i in range(self._n_layers):
            feed[f"kv_k_layer{i}"] = self._state[f"kv_k_layer{i}"]
            feed[f"kv_v_layer{i}"] = self._state[f"kv_v_layer{i}"]
        for i in range(self._n_layers):
            feed[f"conv_buf_layer{i}"] = self._state[f"conv_buf_layer{i}"]
        feed["sub_buf1"] = self._state["sub_buf1"]
        feed["sub_buf2"] = self._state["sub_buf2"]

        # Run ONNX
        outputs = self.session.run(None, feed)

        # Parse outputs: log_probs, pos_offset, kv_k*12, kv_v*12, conv*12, sub1, sub2
        idx = 0
        log_probs = outputs[idx]; idx += 1
        self._state["pos_offset"] = outputs[idx]; idx += 1
        # Outputs are emitted in interleaved per-layer order to match
        # input_names (kv_k_0, kv_v_0, kv_k_1, kv_v_1, ...). Parsing
        # these as sequential (all Ks then all Vs) scrambles which K/V
        # belongs to which layer, silently corrupting state carry past
        # chunk 0 (zero-tensors hide it at chunk 0).
        for i in range(self._n_layers):
            kv_k = outputs[idx]; idx += 1
            if kv_k.shape[2] > self._max_cache_len:
                kv_k = kv_k[:, :, -self._max_cache_len:, :]
            self._state[f"kv_k_layer{i}"] = kv_k

            kv_v = outputs[idx]; idx += 1
            if kv_v.shape[2] > self._max_cache_len:
                kv_v = kv_v[:, :, -self._max_cache_len:, :]
            self._state[f"kv_v_layer{i}"] = kv_v
        for i in range(self._n_layers):
            self._state[f"conv_buf_layer{i}"] = outputs[idx]; idx += 1
        self._state["sub_buf1"] = outputs[idx]; idx += 1
        self._state["sub_buf2"] = outputs[idx]; idx += 1

        # Accumulate log_probs (T, 1, C) -> (T, C)
        T_out = log_probs.shape[0]
        if T_out > 0:
            lp = log_probs[:, 0, :]
            self._all_log_probs.append(lp)

            # Decode everything and emit new characters immediately
            all_lp = np.concatenate(self._all_log_probs, axis=0)
            full_text = greedy_ctc_decode(all_lp)
            new_chars = full_text[len(self._emitted_text):]
            self._emitted_text = full_text
            return new_chars

        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CW-Former streaming ONNX inference (no PyTorch required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="Path to ONNX model (fp32 or int8)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Path to mel_config.json (default: same dir as model)")
    parser.add_argument("--input", default=None, metavar="PATH",
                        help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("--device", nargs="?", const=-1, type=int, default=None,
                        metavar="ID",
                        help="Live audio from device (omit ID for default)")
    parser.add_argument("--list-devices", action="store_true", dest="list_devices",
                        help="List available audio input devices and exit")
    parser.add_argument("--stdin", action="store_true",
                        help="Read raw 16-bit PCM from stdin (16 kHz mono)")

    # Decode settings
    parser.add_argument("--chunk-ms", type=int, default=500, metavar="MS",
                        dest="chunk_ms",
                        help="Processing chunk size in milliseconds")

    # Display
    parser.add_argument("--lines", type=int, default=8, metavar="N",
                        help="Number of text lines in live display")

    args = parser.parse_args()

    if args.list_devices:
        print(list_devices())
        return

    if args.input is None and args.device is None and not args.stdin:
        parser.error("Provide --input, --device, or --stdin")

    dec = CWFormerStreamingONNX(
        model_path=args.model,
        config_path=args.config,
        chunk_ms=args.chunk_ms,
    )

    model_name = Path(args.model).name
    status = f"model={model_name} chunk={dec.chunk_ms}ms greedy streaming"

    # ---- File decode ----
    if args.input is not None:
        print(f"[onnx-streaming] {status}", file=sys.stderr)
        transcript = dec.decode_file(args.input)
        print(transcript)
        return

    # ---- Live device decode ----
    if args.device is not None:
        dev_id = args.device if args.device >= 0 else None
        try:
            import sounddevice as sd
            dev_name = sd.query_devices(dev_id, "input")["name"]
        except Exception:
            dev_name = f"device {dev_id}"

        display = LiveDisplay(max_text_lines=args.lines, status=status)
        print(f"[onnx-streaming] Listening on: {dev_name}", file=sys.stderr)
        print(f"[onnx-streaming] {status}", file=sys.stderr)
        print(f"[onnx-streaming] Press Ctrl+C to stop.\n", file=sys.stderr)

        source = device_stream(dec.sample_rate, device=dev_id)
        dec.decode_live(source, display=display)
        return

    # ---- Stdin decode ----
    if args.stdin:
        print(f"[onnx-streaming] {status}", file=sys.stderr)
        print(f"[onnx-streaming] Reading raw 16-bit PCM from stdin...",
              file=sys.stderr)

        def _stdin_stream():
            chunk_bytes = int(0.1 * dec.sample_rate * 2)
            try:
                while True:
                    data = sys.stdin.buffer.read(chunk_bytes)
                    if not data:
                        break
                    audio = np.frombuffer(
                        data, dtype=np.int16).astype(np.float32) / 32768.0
                    yield audio
            except KeyboardInterrupt:
                pass

        display = LiveDisplay(max_text_lines=args.lines, status=status)
        dec.decode_live(_stdin_stream(), display=display)


if __name__ == "__main__":
    main()
