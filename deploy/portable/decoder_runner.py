"""Threaded driver: audio source -> ONNX streaming decoder -> events.

Wraps :class:`CWFormerStreamingONNX` with a worker thread that pulls
audio from a source iterator, runs each chunk through the model, and
fires three callbacks:

* ``on_text(new_text)`` -- newly decoded characters (suffix only).
* ``on_chunk_mel_chars(mel_frames, new_chars, chunk_first_ctc_frame)``
  -- the mel block that was just consumed plus any new chars annotated
  with their per-chunk CTC-frame index. The runner inlines the same
  ONNX feed/parse path as :meth:`CWFormerStreamingONNX._process_chunk`
  so we get both the mel and the log-probs without recomputing or
  duplicating state.
* ``on_state(state, message)`` -- "running" / "idle" / "error".
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

_DEPLOY = Path(__file__).resolve().parent.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from inference_onnx import (  # type: ignore  # noqa: E402
    BLANK_IDX, IDX_TO_CHAR, CWFormerStreamingONNX, greedy_ctc_decode,
)


log = logging.getLogger("portable.decoder")


def greedy_with_frames(log_probs: np.ndarray) -> List[Tuple[str, int]]:
    """Greedy CTC with frame indices. (char, first-frame-of-run)."""
    if log_probs.size == 0:
        return []
    indices = np.argmax(log_probs, axis=-1)
    out: List[Tuple[str, int]] = []
    prev = -1
    for t, idx in enumerate(indices):
        idx_i = int(idx)
        if idx_i != prev:
            if idx_i != BLANK_IDX:
                ch = IDX_TO_CHAR.get(idx_i, "")
                if ch:
                    out.append((ch, t))
            prev = idx_i
    return out


class DecoderRunner:

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_chunk_mel_chars: Callable[
            [np.ndarray, List[Tuple[str, int]], int], None],
        on_state: Callable[[str, str], None],
    ) -> None:
        self._on_text = on_text
        self._on_chunk = on_chunk_mel_chars
        self._on_state = on_state

        self._dec: Optional[CWFormerStreamingONNX] = None
        self._model_path: Optional[str] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Char tracking across chunks. We re-decode the full accumulated
        # log_probs each chunk (causal CTC -> prefix is stable), and
        # diff against the previously-emitted prefix to find new chars.
        self._prev_chars: List[Tuple[str, int]] = []
        self._chunk_first_ctc_frame = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def decoder(self) -> Optional[CWFormerStreamingONNX]:
        return self._dec

    def load_model(self, model_path: str, chunk_ms: int = 500) -> None:
        with self._lock:
            self._dec = CWFormerStreamingONNX(
                model_path=model_path, chunk_ms=chunk_ms)
            self._model_path = model_path

    def start(
        self,
        source_iter_factory: Callable[
            [threading.Event], Iterator[np.ndarray]],
    ) -> None:
        if self.running:
            return
        if self._dec is None:
            self._on_state("error", "no model loaded")
            return
        self._stop.clear()
        self._dec.reset()
        self._prev_chars = []
        self._chunk_first_ctc_frame = 0
        self._thread = threading.Thread(
            target=self._run, args=(source_iter_factory,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None

    def _run(
        self,
        source_iter_factory: Callable[
            [threading.Event], Iterator[np.ndarray]],
    ) -> None:
        try:
            self._on_state("running", "")
            it = source_iter_factory(self._stop)
            for chunk in it:
                if self._stop.is_set():
                    break
                self._process_chunk(chunk)
            try:
                tail = self._flush()
                if tail:
                    self._on_text(tail)
            except Exception:
                log.exception("flush failed")
            self._on_state("idle", "stream ended")
        except Exception as e:
            log.exception("decoder loop error")
            self._on_state("error", str(e))
        finally:
            self._thread = None

    def _process_chunk(self, audio_chunk: np.ndarray) -> None:
        dec = self._dec
        if dec is None:
            return

        mel, dec._stft_buffer = dec.mel.compute_streaming(
            audio_chunk, dec._stft_buffer)
        if mel.shape[1] == 0:
            return

        # ---- ONNX feed (mirrors CWFormerStreamingONNX._process_chunk) ----
        feed: dict = {"mel_chunk": mel,
                      "pos_offset": dec._state["pos_offset"]}
        for i in range(dec._n_layers):
            feed[f"kv_k_layer{i}"] = dec._state[f"kv_k_layer{i}"]
            feed[f"kv_v_layer{i}"] = dec._state[f"kv_v_layer{i}"]
        for i in range(dec._n_layers):
            feed[f"conv_buf_layer{i}"] = dec._state[f"conv_buf_layer{i}"]
        feed["sub_buf1"] = dec._state["sub_buf1"]
        feed["sub_buf2"] = dec._state["sub_buf2"]

        outputs = dec.session.run(None, feed)

        idx = 0
        log_probs = outputs[idx]; idx += 1
        dec._state["pos_offset"] = outputs[idx]; idx += 1
        for i in range(dec._n_layers):
            kv_k = outputs[idx]; idx += 1
            if kv_k.shape[2] > dec._max_cache_len:
                kv_k = kv_k[:, :, -dec._max_cache_len:, :]
            dec._state[f"kv_k_layer{i}"] = kv_k
            kv_v = outputs[idx]; idx += 1
            if kv_v.shape[2] > dec._max_cache_len:
                kv_v = kv_v[:, :, -dec._max_cache_len:, :]
            dec._state[f"kv_v_layer{i}"] = kv_v
        for i in range(dec._n_layers):
            dec._state[f"conv_buf_layer{i}"] = outputs[idx]; idx += 1
        dec._state["sub_buf1"] = outputs[idx]; idx += 1
        dec._state["sub_buf2"] = outputs[idx]; idx += 1

        T_out = log_probs.shape[0]
        new_charframes: List[Tuple[str, int]] = []
        if T_out > 0:
            lp = log_probs[:, 0, :]
            dec._all_log_probs.append(lp)

            all_lp = np.concatenate(dec._all_log_probs, axis=0)
            full_text = greedy_ctc_decode(all_lp)
            new_chars = full_text[len(dec._emitted_text):]
            dec._emitted_text = full_text

            chars_with_frames = greedy_with_frames(all_lp)
            # Emit only chars added since the previous chunk.
            new_charframes_global = chars_with_frames[len(self._prev_chars):]
            self._prev_chars = chars_with_frames
            # Convert global ctc-frame idx -> within-chunk idx.
            for ch, gf in new_charframes_global:
                local = gf - self._chunk_first_ctc_frame
                new_charframes.append((ch, max(0, local)))

            if new_chars:
                self._on_text(new_chars)

        self._on_chunk(mel[0], new_charframes, self._chunk_first_ctc_frame)
        self._chunk_first_ctc_frame += T_out

    def _flush(self) -> str:
        dec = self._dec
        if dec is None:
            return ""
        if len(dec._audio_buffer) <= 0:
            return ""
        pad_right = dec.mel.n_fft // 2
        chunk = np.concatenate(
            [dec._audio_buffer, np.zeros(pad_right, dtype=np.float32)])
        dec._audio_buffer = np.zeros(0, dtype=np.float32)
        before = dec._emitted_text
        self._process_chunk(chunk)
        return dec._emitted_text[len(before):]
