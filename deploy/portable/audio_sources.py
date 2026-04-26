"""Audio source iterators for the portable web decoder.

Each source factory takes a stop ``threading.Event`` and yields
``chunk_samples``-sized float32 arrays. Real-time pacing applies to
file sources; live sources are paced by the underlying capture/process.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Make sibling deploy/ importable so we can re-use inference_onnx helpers.
_DEPLOY = Path(__file__).resolve().parent.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from inference_onnx import (  # type: ignore  # noqa: E402
    _peak_normalize,
    device_stream,
    load_audio,
)


log = logging.getLogger("portable.audio")


def _zero_pad(chunk: np.ndarray, target: int) -> np.ndarray:
    if len(chunk) >= target:
        return chunk[:target]
    return np.concatenate([chunk, np.zeros(target - len(chunk), dtype=np.float32)])


def file_source(
    path: str,
    sample_rate: int,
    chunk_samples: int,
    stop_event: threading.Event,
    realtime: bool = True,
) -> Iterator[np.ndarray]:
    audio = load_audio(path, sample_rate)
    audio = _peak_normalize(audio, target_peak=0.7)
    pos = 0
    chunk_dur = chunk_samples / sample_rate
    t0 = time.monotonic()
    i = 0
    while pos < len(audio) and not stop_event.is_set():
        end = min(pos + chunk_samples, len(audio))
        yield _zero_pad(audio[pos:end], chunk_samples)
        pos = end
        i += 1
        if realtime:
            target = t0 + i * chunk_dur
            while not stop_event.is_set():
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))


def device_source(
    sample_rate: int,
    chunk_samples: int,
    device_id: Optional[int],
    stop_event: threading.Event,
) -> Iterator[np.ndarray]:
    gen = device_stream(target_sr=sample_rate, device=device_id, chunk_ms=100.0)
    buf = np.zeros(0, dtype=np.float32)
    try:
        for raw in gen:
            if stop_event.is_set():
                break
            buf = np.concatenate([buf, raw])
            while len(buf) >= chunk_samples and not stop_event.is_set():
                yield buf[:chunk_samples].copy()
                buf = buf[chunk_samples:]
    finally:
        try:
            gen.close()
        except Exception:
            pass


def command_source(
    command: str,
    sample_rate: int,
    chunk_samples: int,
    stop_event: threading.Event,
) -> Iterator[np.ndarray]:
    """Spawn a shell command and consume its raw 16-bit signed PCM stdout.

    The command must emit interleaved s16le mono at ``sample_rate`` Hz.
    Example: ``rtl_fm -f 14060000 -M usb -s 16000 -``.

    Security: the command runs under the server's user with full shell
    expansion. Only safe because the AP is isolated and authenticated;
    the UI must label it as such.
    """
    if not command.strip():
        raise ValueError("empty command")

    log.info("command_source: spawning %r", command)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        chunk_bytes = chunk_samples * 2  # int16 mono
        while not stop_event.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) < chunk_bytes:
                data = data + bytes(chunk_bytes - len(data))
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            yield audio
    finally:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def list_audio_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError:
        return []
    try:
        default_id = sd.default.device[0] if sd.default.device else None
    except Exception:
        default_id = None
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out: list[dict] = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) >= 1:
            out.append({
                "id": i,
                "name": d.get("name", f"device {i}"),
                "default": (i == default_id),
                "samplerate": int(d.get("default_samplerate", 0) or 0),
            })
    return out
