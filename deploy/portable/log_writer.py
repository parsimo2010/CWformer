"""Decoded text logger with periodic UTC timestamps and rename support.

File format::

    [2026-04-25 14:00:00 UTC] === SESSION START ===
    [2026-04-25 14:00:00 UTC] [time-sync: GPS, stratum=1]
    RX: CQ CQ DE W1ABC W1ABC K
    [2026-04-25 14:01:23 UTC] TX: W1ABC DE K2DEF K2DEF K
    RX: K2DEF DE W1ABC R UR RST 599 ...
    [2026-04-25 14:05:00 UTC]
    RX: ...
    [2026-04-25 14:42:11 UTC] === SESSION END ===

The format is line-oriented and parsable for a future ADIF exporter:
each TX/RX line has a wall-clock anchor, and continuous RX text is
split by 5-minute timestamp markers.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path
from typing import Optional


class DecodeLogger:

    def __init__(
        self,
        log_dir: Path,
        filename: str = "",
        timestamp_interval_sec: int = 300,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp_interval_sec = timestamp_interval_sec

        self._lock = threading.Lock()
        self._file = None
        self._path: Optional[Path] = None
        self._last_ts_time = 0.0
        self._mid_rx_line = False

        self.set_filename(filename or self._auto_filename())

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @staticmethod
    def _auto_filename() -> str:
        return f"decoded_{dt.datetime.now().strftime('%Y%m%d')}.log"

    @staticmethod
    def _utc_now() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _sanitize(name: str) -> str:
        if not name.lower().endswith(".log"):
            name = name + ".log"
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()

    def list_logs(self) -> list[str]:
        return sorted(p.name for p in self.log_dir.glob("*.log"))

    def set_filename(self, name: str) -> None:
        safe = self._sanitize(name)
        if not safe:
            return
        new_path = self.log_dir / safe
        with self._lock:
            self._close_locked()
            self._path = new_path
            self._file = new_path.open("a", encoding="utf-8", buffering=1)
            self._mid_rx_line = False
            self._write_locked(f"[{self._utc_now()}] === SESSION START ===\n")
            self._last_ts_time = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._file is not None:
            if self._mid_rx_line:
                self._file.write("\n")
                self._mid_rx_line = False
            self._file.write(f"[{self._utc_now()}] === SESSION END ===\n")
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _write_locked(self, s: str) -> None:
        if self._file is None:
            return
        try:
            self._file.write(s)
            self._file.flush()
        except Exception:
            pass

    def write_rx(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if not self._mid_rx_line:
                self._write_locked("RX: ")
                self._mid_rx_line = True
            self._write_locked(text)
            self._maybe_timestamp_locked()

    def write_tx(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._mid_rx_line:
                self._write_locked("\n")
                self._mid_rx_line = False
            self._write_locked(f"[{self._utc_now()}] TX: {text}\n")

    def write_event(self, text: str) -> None:
        with self._lock:
            if self._mid_rx_line:
                self._write_locked("\n")
                self._mid_rx_line = False
            self._write_locked(f"[{self._utc_now()}] {text}\n")

    def _maybe_timestamp_locked(self) -> None:
        now = time.monotonic()
        if now - self._last_ts_time >= self.timestamp_interval_sec:
            if self._mid_rx_line:
                self._write_locked("\n")
                self._mid_rx_line = False
            self._write_locked(f"[{self._utc_now()}]\n")
            self._last_ts_time = now
