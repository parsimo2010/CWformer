"""Persisted user-config for the portable web decoder."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PortableConfig:
    # Decoder
    model_path: str = ""
    chunk_ms: int = 500

    # Audio source
    source_kind: str = "device"        # "device" | "file" | "command"
    device_id: Optional[int] = None
    file_path: str = ""
    command: str = ""                  # e.g. "rtl_fm -f 14060000 -M usb -s 16000 -"

    # Logging
    log_dir: str = ""                  # empty -> ~/cwformer_logs
    log_filename: str = ""             # empty -> decoded_YYYYMMDD.log
    timestamp_interval_sec: int = 300

    # Operator + keyer
    callsign: str = "MYCALL"
    wpm: int = 20
    keyer_pin: int = 17                # BCM

    # Display
    spectrogram_enabled: bool = True
    spectrogram_interval_ms: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    @classmethod
    def load(cls, path: Path) -> "PortableConfig":
        if path.exists():
            try:
                with path.open() as f:
                    data = json.load(f)
                allowed = {f.name for f in cls.__dataclass_fields__.values()}
                data = {k: v for k, v in data.items() if k in allowed}
                return cls(**data)
            except Exception:
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)
