#!/usr/bin/env python3
"""serve.py -- Portable CWformer web decoder for Raspberry Pi.

Runs a small Flask + Socket.IO server intended to be reached from a
phone or tablet connected to the Pi's own AP (see setup_hotspot.sh).

Usage::

    python deploy/portable/serve.py
    python deploy/portable/serve.py --no-gpio --debug

After ``setup_hotspot.sh`` has been run once, just connect your phone to
the AP and browse to http://192.168.50.1:8080/.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DEPLOY = _THIS_DIR.parent
for p in (_THIS_DIR, _DEPLOY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_socketio import SocketIO

from audio_sources import (  # noqa: E402
    command_source, device_source, file_source, list_audio_devices,
)
from config import PortableConfig  # noqa: E402
from decoder_runner import DecoderRunner  # noqa: E402
from keyer import DEFAULT_MACROS, Keyer, render_macro  # noqa: E402
from log_writer import DecodeLogger  # noqa: E402
from spectrogram_stream import SpectrogramStream  # noqa: E402
from time_sync import query_chrony  # noqa: E402


log = logging.getLogger("portable.serve")


CONFIG_PATH = Path.home() / ".cwformer" / "portable.json"
DEFAULT_LOG_DIR = Path.home() / "cwformer_logs"

# File-picker defaults. The picker walks ``scan_root`` up to ``scan_depth``
# levels deep so the user can choose any model / wav file living near the
# server folder without typing absolute paths. Default scan root is the
# parent of ``deploy/`` (the project root in a normal checkout).
DEFAULT_SCAN_ROOT = _DEPLOY.parent
DEFAULT_SCAN_DEPTH = 3
MODEL_EXTENSIONS = (".onnx",)
WAV_EXTENSIONS = (".wav", ".flac", ".ogg")
_PRUNE_DIR_NAMES = {
    "__pycache__", "node_modules", ".git", ".hg", ".svn",
    ".venv", "venv", "env", ".env",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    ".tox", "build", "dist", "site-packages",
}
_MAX_PICKER_RESULTS = 500


def _scan_files(
    scan_root: Path,
    extensions: Iterable[str],
    max_depth: int,
) -> List[dict]:
    """Recursively list files under ``scan_root`` matching any extension.

    Walks at most ``max_depth`` directories deep from ``scan_root``. Skips
    common noise folders (``.git``, ``__pycache__``, ``venv`` …). Returns
    sorted dicts with ``path`` (absolute), ``rel`` (POSIX-style relative
    to ``scan_root``), ``name`` (basename), and ``size`` in bytes.
    """
    suffixes = {ext.lower() for ext in extensions}
    out: List[dict] = []
    seen: set = set()
    try:
        scan_root_resolved = scan_root.resolve()
    except OSError:
        return out
    if not scan_root_resolved.exists() or not scan_root_resolved.is_dir():
        return out

    base_depth = len(scan_root_resolved.parts)

    for dirpath, dirnames, filenames in os.walk(
        scan_root_resolved, followlinks=False,
    ):
        cur_depth = len(Path(dirpath).parts) - base_depth
        if cur_depth >= max_depth:
            dirnames.clear()
        # Prune hidden + noise dirs in-place so os.walk doesn't descend.
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in _PRUNE_DIR_NAMES
        )
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in suffixes:
                continue
            p = Path(dirpath) / fn
            try:
                rp = p.resolve()
            except OSError:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            try:
                rel = rp.relative_to(scan_root_resolved)
            except ValueError:
                rel = Path(rp.name)
            try:
                size = rp.stat().st_size
            except OSError:
                size = 0
            out.append({
                "path": str(rp),
                "rel": rel.as_posix(),
                "name": rp.name,
                "size": size,
            })
            if len(out) >= _MAX_PICKER_RESULTS:
                return out
    return out


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = PortableConfig.load(CONFIG_PATH)
        if not self.cfg.log_dir:
            self.cfg.log_dir = str(DEFAULT_LOG_DIR)
        if not self.cfg.model_path:
            for name in (
                "cwformer_streaming_int8.onnx",
                "cwformer_streaming_fp32.onnx",
            ):
                p = _DEPLOY / name
                if p.exists():
                    self.cfg.model_path = str(p)
                    break

        scan_root = Path(args.scan_root) if args.scan_root else DEFAULT_SCAN_ROOT
        try:
            scan_root = scan_root.resolve()
        except OSError:
            pass
        self.scan_root: Path = scan_root
        self.scan_depth: int = max(1, int(args.scan_depth))

        self.logger = DecodeLogger(
            Path(self.cfg.log_dir),
            filename=self.cfg.log_filename,
            timestamp_interval_sec=self.cfg.timestamp_interval_sec,
        )
        if self.logger.path is not None:
            self.cfg.log_filename = self.logger.path.name

        self.spec = SpectrogramStream()
        self.spec.set_enabled(self.cfg.spectrogram_enabled)

        gpio_enabled = (not args.no_gpio) and platform.system() == "Linux"
        self.keyer = Keyer(
            pin=self.cfg.keyer_pin,
            wpm=self.cfg.wpm,
            gpio_enabled=gpio_enabled,
        )

        self.decoder = DecoderRunner(
            on_text=self._on_decoded_text,
            on_chunk_mel_chars=self._on_chunk_mel_chars,
            on_state=self._on_decoder_state,
        )
        if self.cfg.model_path and Path(self.cfg.model_path).exists():
            try:
                self.decoder.load_model(
                    self.cfg.model_path, chunk_ms=self.cfg.chunk_ms)
            except Exception:
                log.exception("initial model load failed")

        self._socketio: Optional[SocketIO] = None
        self._stop_threads = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self._spec_thread: Optional[threading.Thread] = None
        self._last_sync = None

    # ---- Setup ----

    def attach_socketio(self, sio: SocketIO) -> None:
        self._socketio = sio

    def attach_keyer_callbacks(self) -> None:
        def on_state(on: bool) -> None:
            if self._socketio is not None:
                self._socketio.emit("key_state", {"on": on})

        def on_text(text: str) -> None:
            self.logger.write_tx(text)
            if self._socketio is not None:
                self._socketio.emit("tx_text", {"text": text})

        self.keyer.set_callbacks(on_state, on_text)

    def start_background(self) -> None:
        self._stop_threads.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        self._spec_thread = threading.Thread(
            target=self._spec_loop, daemon=True)
        self._spec_thread.start()

    def stop_background(self) -> None:
        self._stop_threads.set()
        try:
            self.decoder.stop()
        except Exception:
            log.exception("decoder.stop failed")
        try:
            self.keyer.close()
        except Exception:
            log.exception("keyer.close failed")
        try:
            self.logger.close()
        except Exception:
            log.exception("logger.close failed")

    # ---- Background loops ----

    def _sync_loop(self) -> None:
        while not self._stop_threads.is_set():
            try:
                status = query_chrony()
                self._last_sync = status
                if self._socketio is not None:
                    self._socketio.emit("time_sync", status.to_dict())
            except Exception:
                log.exception("time-sync poll failed")
            self._stop_threads.wait(5.0)

    def _spec_loop(self) -> None:
        interval = max(0.05, self.cfg.spectrogram_interval_ms / 1000.0)
        while not self._stop_threads.is_set():
            try:
                if self.spec.enabled and self._socketio is not None:
                    img = self.spec.render_base64()
                    if img is not None:
                        self._socketio.emit("spectrogram", {"img": img})
            except Exception:
                log.exception("spectrogram render failed")
            self._stop_threads.wait(interval)

    # ---- Decoder callbacks ----

    def _on_decoded_text(self, new_text: str) -> None:
        if not new_text:
            return
        self.logger.write_rx(new_text)
        if self._socketio is not None:
            self._socketio.emit("decoded_text", {"text": new_text})

    def _on_chunk_mel_chars(
        self,
        mel: np.ndarray,
        new_chars: list,
        chunk_first_ctc_frame: int,
    ) -> None:
        self.spec.add_frames(mel)
        if new_chars:
            self.spec.add_chars(new_chars, chunk_first_ctc_frame)

    def _on_decoder_state(self, state: str, msg: str) -> None:
        if self._socketio is not None:
            self._socketio.emit("decoder_state",
                                {"state": state, "message": msg})

    def save_config(self) -> None:
        try:
            self.cfg.save(CONFIG_PATH)
        except Exception:
            log.exception("config save failed")


# ---------------------------------------------------------------------------
# Flask + SocketIO
# ---------------------------------------------------------------------------

state: Optional[AppState] = None
app = Flask(
    __name__,
    static_folder=str(_THIS_DIR / "static"),
    template_folder=str(_THIS_DIR / "templates"),
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def _state() -> AppState:
    assert state is not None, "AppState not initialized"
    return state


@app.route("/")
def index():
    return send_from_directory(
        str(_THIS_DIR / "templates"), "index.html")


@app.route("/api/status")
def api_status():
    s = _state()
    return jsonify({
        "running": s.decoder.running,
        "model_path": s.cfg.model_path,
        "callsign": s.cfg.callsign,
        "wpm": s.cfg.wpm,
        "log_path": str(s.logger.path) if s.logger.path else "",
        "log_filename": s.logger.path.name if s.logger.path else "",
        "spectrogram_enabled": s.spec.enabled,
        "gpio_enabled": s.keyer.hardware_active,
        "source": {
            "kind": s.cfg.source_kind,
            "device_id": s.cfg.device_id,
            "file_path": s.cfg.file_path,
            "command": s.cfg.command,
        },
        "scan_root": str(s.scan_root),
        "scan_depth": s.scan_depth,
        "time_sync": s._last_sync.to_dict() if s._last_sync else None,
    })


@app.route("/api/devices")
def api_devices():
    return jsonify({"devices": list_audio_devices()})


@app.route("/api/models")
def api_models():
    s = _state()
    models = _scan_files(s.scan_root, MODEL_EXTENSIONS, s.scan_depth)
    return jsonify({
        "models": models,
        "scan_root": str(s.scan_root),
        "scan_depth": s.scan_depth,
    })


@app.route("/api/wavs")
def api_wavs():
    s = _state()
    wavs = _scan_files(s.scan_root, WAV_EXTENSIONS, s.scan_depth)
    return jsonify({
        "wavs": wavs,
        "scan_root": str(s.scan_root),
        "scan_depth": s.scan_depth,
        # Older clients displayed ``dir`` — keep it for backwards compat.
        "dir": str(s.scan_root),
    })


@app.route("/api/macros")
def api_macros():
    return jsonify({"macros": DEFAULT_MACROS})


@app.route("/api/log/list")
def api_log_list():
    s = _state()
    return jsonify({
        "logs": s.logger.list_logs(),
        "current": s.logger.path.name if s.logger.path else "",
        "dir": str(s.logger.log_dir),
    })


@app.route("/api/log/download")
def api_log_download():
    s = _state()
    name = request.args.get("name") or (
        s.logger.path.name if s.logger.path else None)
    if not name:
        return ("no log file", 404)
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()
    p = s.logger.log_dir / safe
    if not p.exists() or p.parent.resolve() != s.logger.log_dir.resolve():
        return ("not found", 404)
    return send_file(
        str(p), as_attachment=True, download_name=safe, mimetype="text/plain")


# ---------------------------------------------------------------------------
# SocketIO events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def _on_connect():
    s = _state()
    socketio.emit("init", {
        "config": {
            "callsign": s.cfg.callsign,
            "wpm": s.cfg.wpm,
            "model_path": s.cfg.model_path,
            "source": {
                "kind": s.cfg.source_kind,
                "device_id": s.cfg.device_id,
                "file_path": s.cfg.file_path,
                "command": s.cfg.command,
            },
            "log_filename": s.logger.path.name if s.logger.path else "",
            "spectrogram_enabled": s.spec.enabled,
            "chunk_ms": s.cfg.chunk_ms,
        },
        "decoder_running": s.decoder.running,
        "gpio_enabled": s.keyer.hardware_active,
        "macros": DEFAULT_MACROS,
        "time_sync": s._last_sync.to_dict() if s._last_sync else None,
    })


@socketio.on("start_decode")
def _on_start(payload):
    s = _state()
    if s.decoder.running:
        return {"ok": False, "error": "already running"}

    payload = payload or {}
    cfg = s.cfg
    cfg.source_kind = payload.get("kind", cfg.source_kind)
    if cfg.source_kind == "device":
        dev = payload.get("device_id", cfg.device_id)
        if dev in (None, "", "null"):
            cfg.device_id = None
        else:
            try:
                cfg.device_id = int(dev)
            except (TypeError, ValueError):
                cfg.device_id = None
    elif cfg.source_kind == "file":
        cfg.file_path = str(payload.get("file_path", cfg.file_path) or "")
    elif cfg.source_kind == "command":
        cfg.command = str(payload.get("command", cfg.command) or "")
    else:
        return {"ok": False, "error": f"unknown source kind: {cfg.source_kind}"}

    model_path = str(payload.get("model_path") or cfg.model_path)
    if not model_path or not Path(model_path).exists():
        return {"ok": False, "error": f"model not found: {model_path}"}
    if model_path != cfg.model_path or s.decoder.decoder is None:
        try:
            s.decoder.load_model(model_path, chunk_ms=cfg.chunk_ms)
            cfg.model_path = model_path
        except Exception as e:
            log.exception("model load failed")
            socketio.emit(
                "decoder_state",
                {"state": "error", "message": f"model load: {e}"})
            return {"ok": False, "error": str(e)}

    s.save_config()

    dec = s.decoder.decoder
    if dec is None:
        return {"ok": False, "error": "no decoder"}
    sample_rate = dec.sample_rate
    chunk_samples = dec._chunk_samples

    if cfg.source_kind == "device":
        device_id = cfg.device_id

        def factory(stop_event):
            return device_source(sample_rate, chunk_samples, device_id, stop_event)
    elif cfg.source_kind == "file":
        path = cfg.file_path
        if not path or not Path(path).exists():
            return {"ok": False, "error": f"file not found: {path}"}

        def factory(stop_event):
            return file_source(path, sample_rate, chunk_samples, stop_event)
    else:  # command
        cmd = cfg.command
        if not cmd.strip():
            return {"ok": False, "error": "no command set"}

        def factory(stop_event):
            return command_source(cmd, sample_rate, chunk_samples, stop_event)

    s.spec.reset()
    s.logger.write_event(
        f"DECODE START source={cfg.source_kind} model={Path(cfg.model_path).name}")
    s.decoder.start(factory)
    return {"ok": True}


@socketio.on("stop_decode")
def _on_stop(_):
    s = _state()
    s.decoder.stop()
    s.logger.write_event("DECODE STOP")
    return {"ok": True}


@socketio.on("clear_text")
def _on_clear(_):
    socketio.emit("text_cleared", {})
    return {"ok": True}


@socketio.on("toggle_spectrogram")
def _on_toggle_spec(payload):
    s = _state()
    payload = payload or {}
    on = bool(payload.get("on", not s.spec.enabled))
    s.spec.set_enabled(on)
    s.cfg.spectrogram_enabled = on
    s.save_config()
    return {"ok": True, "on": on}


@socketio.on("set_callsign")
def _on_set_call(payload):
    s = _state()
    call = (payload or {}).get("callsign", s.cfg.callsign)
    s.cfg.callsign = (call or "").strip().upper() or "MYCALL"
    s.save_config()
    socketio.emit("config_updated", {"callsign": s.cfg.callsign})
    return {"ok": True}


@socketio.on("set_wpm")
def _on_set_wpm(payload):
    s = _state()
    try:
        wpm = int((payload or {}).get("wpm", s.cfg.wpm))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid wpm"}
    wpm = max(5, min(60, wpm))
    s.cfg.wpm = wpm
    s.keyer.wpm = wpm
    s.save_config()
    socketio.emit("config_updated", {"wpm": wpm})
    return {"ok": True}


@socketio.on("set_log_filename")
def _on_set_logfile(payload):
    s = _state()
    name = ((payload or {}).get("name", "") or "").strip()
    if not name:
        return {"ok": False, "error": "empty name"}
    s.logger.set_filename(name)
    if s.logger.path is None:
        return {"ok": False, "error": "could not open log file"}
    s.cfg.log_filename = s.logger.path.name
    s.save_config()
    socketio.emit("config_updated",
                  {"log_filename": s.logger.path.name,
                   "log_path": str(s.logger.path)})
    return {"ok": True, "filename": s.logger.path.name}


@socketio.on("send_text")
def _on_send_text(payload):
    s = _state()
    text = ((payload or {}).get("text", "") or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    if not s.keyer.send(text):
        return {"ok": False, "error": "keyer busy"}
    return {"ok": True, "rendered": text}


@socketio.on("send_macro")
def _on_send_macro(payload):
    s = _state()
    payload = payload or {}
    template = payload.get("text", "")
    text = render_macro(
        template,
        mycall=s.cfg.callsign,
        his_call=payload.get("his_call", ""),
        rst_sent=payload.get("rst_sent", "599"),
        rst_rcvd=payload.get("rst_rcvd", "599"),
        name=payload.get("name", ""),
        qth=payload.get("qth", ""),
    )
    if not text.strip():
        return {"ok": False, "error": "empty macro"}
    if not s.keyer.send(text):
        return {"ok": False, "error": "keyer busy"}
    return {"ok": True, "rendered": text}


@socketio.on("cancel_send")
def _on_cancel_send(_):
    _state().keyer.cancel()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable CWformer decoder web server (Pi 5 AP).")
    parser.add_argument("--host", default=None,
                        help="Server bind host (default from saved config).")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default 8080).")
    parser.add_argument("--no-gpio", action="store_true",
                        help="Disable GPIO. Keyer becomes a stub. UI shows "
                             "a 'TX stub' badge so you can tell the keyer "
                             "is not actually keying anything.")
    parser.add_argument("--scan-root", default=None,
                        help="Root directory for the model + wav file "
                             "pickers. The browser shows every .onnx and "
                             ".wav/.flac/.ogg file within --scan-depth "
                             f"levels of this folder. Default: "
                             f"{DEFAULT_SCAN_ROOT} (the project root in a "
                             f"normal checkout).")
    parser.add_argument("--scan-depth", type=int, default=DEFAULT_SCAN_DEPTH,
                        help="Max directory depth for the file pickers. "
                             f"Default: {DEFAULT_SCAN_DEPTH}.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global state
    state = AppState(args)
    state.attach_socketio(socketio)
    state.attach_keyer_callbacks()
    state.start_background()

    host = args.host or state.cfg.host
    port = args.port or state.cfg.port

    log.info("CWformer portable: http://%s:%d/", host, port)
    log.info("  model:    %s", state.cfg.model_path)
    log.info("  log dir:  %s", state.cfg.log_dir)
    log.info("  log file: %s", state.cfg.log_filename or "(auto)")
    log.info("  scan:     %s (depth %d)", state.scan_root, state.scan_depth)
    log.info("  GPIO:     %s",
             "active" if state.keyer.hardware_active else "STUB (no GPIO)")

    try:
        socketio.run(
            app, host=host, port=port,
            allow_unsafe_werkzeug=True, debug=args.debug,
        )
    finally:
        state.stop_background()


if __name__ == "__main__":
    main()
