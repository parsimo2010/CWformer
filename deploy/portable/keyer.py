"""GPIO Morse keyer with stub mode.

Drives a single GPIO output high for the duration of each dit/dah.
Wire an optoisolator from the pin to the rig's key input. PARIS
timing: 1 dit = 60 / (50 * WPM) seconds.

If ``gpiozero`` isn't installed (e.g. running on Windows for development)
or initialization fails, the keyer silently runs in stub mode — every
dit/dah is logged but no GPIO is touched. The web UI surfaces this with
a "TX stub" badge.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


log = logging.getLogger("portable.keyer")


_MORSE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-",
    "+": ".-.-.", "-": "-....-", "(": "-.--.", ")": "-.--.-", "&": ".-...",
    ":": "---...", ";": "-.-.-.", "@": ".--.-.", "'": ".----.", '"': ".-..-.",
    "$": "...-..-", "!": "-.-.--",
}

# Prosigns sent without inter-letter gaps. Recognized inside <...>.
_PROSIGNS: dict[str, str] = {
    "AR": ".-.-.",   "SK": "...-.-", "BT": "-...-",  "KN": "-.--.",
    "AS": ".-...",   "CT": "-.-.-",  "VE": "...-.",  "BK": "-...-.-",
    "SN": "...-.",
}


def _wpm_dit_sec(wpm: int) -> float:
    return 60.0 / (50.0 * max(1, int(wpm)))


def _expand_prosigns(text: str) -> list[str]:
    """Tokenize text into a list of either single chars or merged prosigns.

    ``"73 <SK>"`` -> ``["7", "3", " ", "SK"]``. The leading ``<``...``>``
    syntax is the only way to send a prosign as run-on symbols.
    """
    out: list[str] = []
    i = 0
    upper = text.upper()
    while i < len(upper):
        ch = upper[i]
        if ch == "<":
            end = upper.find(">", i + 1)
            if end > i:
                tag = upper[i + 1:end]
                if tag in _PROSIGNS:
                    out.append(tag)
                    i = end + 1
                    continue
        out.append(ch)
        i += 1
    return out


@dataclass
class Keyer:
    """GPIO keyer. Stubbed if hardware unavailable."""

    pin: int = 17
    wpm: int = 20
    gpio_enabled: bool = False

    _device: object = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(
        default=None, init=False, repr=False)
    _send_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        # Assigned as instance attributes (not dataclass fields) so the
        # function-descriptor protocol doesn't bind them to ``self`` when
        # they're accessed through ``self._on_state(...)``.
        self._on_state: Callable[[bool], None] = lambda on: None
        self._on_text: Callable[[str], None] = lambda text: None
        self._open()

    @property
    def hardware_active(self) -> bool:
        return self._device is not None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_callbacks(
        self,
        on_state: Callable[[bool], None],
        on_text: Callable[[str], None],
    ) -> None:
        self._on_state = on_state
        self._on_text = on_text

    def _open(self) -> None:
        if not self.gpio_enabled:
            log.info("Keyer: stub mode (GPIO disabled)")
            return
        try:
            from gpiozero import OutputDevice  # type: ignore
            self._device = OutputDevice(
                self.pin, active_high=True, initial_value=False)
            log.info("Keyer: GPIO ready on BCM pin %d", self.pin)
        except Exception as e:
            log.warning("Keyer: GPIO init failed (%s) -- running in stub mode", e)
            self._device = None

    def close(self) -> None:
        self.cancel()
        if self._device is not None:
            try:
                self._device.off()
                self._device.close()
            except Exception:
                pass
            self._device = None

    def _key_on(self) -> None:
        if self._device is not None:
            try:
                self._device.on()
            except Exception:
                log.exception("key_on failed")
        self._on_state(True)

    def _key_off(self) -> None:
        if self._device is not None:
            try:
                self._device.off()
            except Exception:
                log.exception("key_off failed")
        self._on_state(False)

    def _interruptible_sleep(self, sec: float) -> None:
        end = time.monotonic() + sec
        while not self._stop.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.005, remaining))

    def send(self, text: str) -> bool:
        with self._send_lock:
            if self.busy:
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._send_text, args=(text,), daemon=True)
            self._thread.start()
        return True

    def cancel(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._key_off()

    def _send_text(self, text: str) -> None:
        try:
            self._on_text(text)
            tokens = _expand_prosigns(text)
            dit = _wpm_dit_sec(self.wpm)
            dah = 3 * dit
            inter_sym = dit
            inter_letter = 3 * dit
            inter_word = 7 * dit

            prev_was_letter = False
            for tok in tokens:
                if self._stop.is_set():
                    return
                if tok == " ":
                    self._interruptible_sleep(inter_word - inter_letter)
                    prev_was_letter = False
                    continue
                if len(tok) > 1:
                    pattern = _PROSIGNS.get(tok, "")
                else:
                    pattern = _MORSE.get(tok, "")
                if not pattern:
                    continue
                if prev_was_letter:
                    self._interruptible_sleep(inter_letter)
                for j, sym in enumerate(pattern):
                    if self._stop.is_set():
                        return
                    self._key_on()
                    self._interruptible_sleep(dit if sym == "." else dah)
                    self._key_off()
                    if j < len(pattern) - 1:
                        self._interruptible_sleep(inter_sym)
                prev_was_letter = True
        finally:
            self._key_off()


def render_macro(
    template: str,
    *,
    mycall: str,
    his_call: str = "",
    rst_sent: str = "599",
    rst_rcvd: str = "599",
    name: str = "",
    qth: str = "",
) -> str:
    return (
        template
        .replace("{MYCALL}", mycall.upper())
        .replace("{CALL}", his_call.upper())
        .replace("{HIS}", his_call.upper())
        .replace("{RST}", rst_sent)
        .replace("{RSTR}", rst_rcvd)
        .replace("{NAME}", name.upper())
        .replace("{QTH}", qth.upper())
        .upper()
    )


DEFAULT_MACROS: list[dict[str, str]] = [
    {"name": "CQ",       "text": "CQ CQ CQ DE {MYCALL} {MYCALL} K"},
    {"name": "CQ SOTA",  "text": "CQ SOTA CQ SOTA DE {MYCALL} {MYCALL} K"},
    {"name": "Answer",   "text": "{CALL} DE {MYCALL} {MYCALL} K"},
    {"name": "Report",   "text": "{CALL} DE {MYCALL} R UR RST {RST} {RST} = NAME {NAME} = QTH {QTH} = HW? {CALL} DE {MYCALL} K"},
    {"name": "TU 73",    "text": "{CALL} DE {MYCALL} TU 73 73 <SK>"},
    {"name": "AGN?",     "text": "{CALL} DE {MYCALL} AGN AGN K"},
    {"name": "QRZ?",     "text": "QRZ? DE {MYCALL} K"},
]
