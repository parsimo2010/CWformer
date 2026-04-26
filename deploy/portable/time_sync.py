"""GPS time-sync status via chronyc.

Parses ``chronyc tracking`` output and surfaces a state suitable for
the web UI: state ("synced" / "syncing" / "unsynced" / "unavailable"),
source ("GPS" / "NTP" / "none" / refid label), stratum, and a detail
string. Source-agnostic — works with chrony fed by gpsd, an SHM/PPS
refclock, or upstream NTP servers.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeSyncStatus:
    state: str
    source: str
    stratum: int
    last_offset_sec: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "source": self.source,
            "stratum": self.stratum,
            "last_offset_sec": self.last_offset_sec,
            "detail": self.detail,
        }


def _run(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def query_chrony() -> TimeSyncStatus:
    out = _run(["chronyc", "tracking"])
    if out is None:
        return TimeSyncStatus(
            state="unavailable",
            source="none",
            stratum=0,
            last_offset_sec=0.0,
            detail="chronyc unavailable (install chrony)",
        )

    fields: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()

    ref_id = fields.get("Reference ID", "")
    m = re.search(r"\(([^)]+)\)", ref_id)
    ref_name = m.group(1) if m else ref_id.strip()

    try:
        stratum = int(fields.get("Stratum", "0"))
    except ValueError:
        stratum = 0

    last_offset = 0.0
    m2 = re.search(
        r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        fields.get("Last offset", ""),
    )
    if m2:
        try:
            last_offset = float(m2.group(1))
        except ValueError:
            pass

    leap = fields.get("Leap status", "Not synchronised")

    upper = ref_name.upper()
    if "GPS" in upper or upper.startswith("NMEA") or upper.startswith("PPS"):
        source = "GPS"
    elif ref_name in ("", "00000000") or stratum == 0:
        source = "none"
    else:
        source = ref_name

    if leap.lower().startswith("not") or stratum == 0:
        state = "unsynced"
    elif source == "GPS":
        state = "synced"
    elif stratum <= 4:
        state = "synced"
    else:
        state = "syncing"

    detail = (
        f"stratum={stratum}, ref={ref_name or 'none'}, "
        f"leap={leap}, last_offset={last_offset:+.3e}s"
    )
    return TimeSyncStatus(
        state=state,
        source=source,
        stratum=stratum,
        last_offset_sec=last_offset,
        detail=detail,
    )
