"""Primitives for the MATLAB-flavoured values ScanImage writes into TIFF tags.

Everything ScanImage stores in a TIFF - the global ``SI.*`` header, the
per-page frame header, the AUX trigger arrays and the I2C packet lists - is
MATLAB source-ish text, not a standard serialisation format. `tifffile`
ships `matlabstr2py`, which is used for the global header, but it is not
reliable for the trigger fields:

* comma-separated numeric arrays come back with the commas glued onto the
  numbers as strings (``'0.313856210,'``), and older ScanImage versions do
  emit commas where 2022.1 emits spaces;
* I2C packet timestamps come back the same way (``'600.251504290,'``);
* a *single* I2C packet ``{{1.5, [255]}}`` collapses to ``['1.5,', [255]]``,
  which is indistinguishable from a two-element list, so packet count can't
  be recovered.

Hence the trigger-related values get purpose-built parsers here. They are
deliberately permissive about separators and whitespace, because these
strings are the one part of the format that has visibly changed between
ScanImage versions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

__all__ = [
    "I2CPacket",
    "parse_aux_trigger",
    "parse_bool",
    "parse_epoch",
    "parse_i2c_data",
    "parse_matlab_scalar",
    "parse_numeric_array",
]

# A MATLAB-printed number: optional sign, digits with optional decimal point,
# optional exponent. Also covers the bare `.5` form.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_INT_RE = re.compile(r"^[-+]?\d+$")

# Values inside brackets are separated by whitespace (ScanImage 2022.x, and
# the form used in the ScanImage docs) or by commas (older versions).
_SEPARATOR_RE = re.compile(r"[\s,]+")

# One I2C packet: `{<timestamp>, <payload>}` where the payload is a byte
# array (`I2CStoreAsChar = false`), a quoted string (`= true`), or - being
# permissive - a bare scalar. Matching the *inner* braces individually means
# a single packet and a list of packets normalise identically, which is
# exactly what `matlabstr2py` gets wrong.
_I2C_PACKET_RE = re.compile(
    rf"\{{\s*(?P<timestamp>{_NUMBER})\s*,\s*"
    rf"(?P<payload>\[[^\]]*\]|'[^']*'|\"[^\"]*\"|{_NUMBER})\s*\}}"
)

_NONFINITE = {
    "inf": math.inf,
    "+inf": math.inf,
    "-inf": -math.inf,
    "nan": math.nan,
    "-nan": math.nan,
}

_BOOLEANS = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
}


def parse_bool(value: object) -> bool:
    """Parse a ScanImage boolean, which may be ``'true'``/``'false'`` or ``'1'``/``'0'``.

    ScanImage is inconsistent about which spelling it uses for which field,
    so both are accepted (as in `dj-imaging`'s ``parse_SI_boolean``). Real
    `bool`/numeric inputs are passed through, so this is safe to call on
    values that `matlabstr2py` already typed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text not in _BOOLEANS:
        raise ValueError(f"cannot interpret {value!r} as a ScanImage boolean")
    return _BOOLEANS[text]


def parse_numeric_array(text: str) -> np.ndarray:
    """Parse a MATLAB numeric array such as ``'[0.31 0.33]'`` into a ``float64`` array.

    Accepts whitespace- and/or comma-separated elements, any level of
    bracket nesting (flattened - every ScanImage field parsed through here
    is conceptually a flat list), and ``'[]'`` / ``''`` for "no values".
    Non-numeric tokens are skipped rather than raising, so a malformed
    entry costs one value instead of the whole file.
    """
    if text is None:
        return np.empty(0, dtype=np.float64)
    inner = str(text).strip().strip("[]{}").strip()
    if not inner:
        return np.empty(0, dtype=np.float64)

    values: list[float] = []
    for token in _SEPARATOR_RE.split(inner):
        token = token.strip("[]{}")
        if not token:
            continue
        lowered = token.lower()
        if lowered in _NONFINITE:
            values.append(_NONFINITE[lowered])
        elif _NUMBER_RE.match(token):
            values.append(float(token))
    return np.asarray(values, dtype=np.float64)


def parse_aux_trigger(text: str) -> np.ndarray:
    """Parse one ``auxTriggerN`` page-header value into an array of timestamps (seconds).

    ScanImage records up to 1000 triggers per frame on each of the four
    lines; an empty ``'[]'`` (by far the common case) yields an empty array.
    """
    return parse_numeric_array(text)


@dataclass(frozen=True)
class I2CPacket:
    """One I2C packet recorded in a page header.

    Exactly one of `data` / `text` is populated, mirroring ScanImage's
    ``I2CStoreAsChar`` machine-data-file setting: byte arrays are stored as
    `data` (``uint8``), strings as `text`. `raw` keeps the verbatim source
    substring so nothing is lost if a payload turns out to be something
    this parser did not anticipate.
    """

    timestamp: float
    data: np.ndarray | None = None
    text: str | None = None
    raw: str = ""

    @property
    def is_valid_timestamp(self) -> bool:
        """Whether the timestamp is usable (ScanImage uses negatives as sentinels)."""
        return math.isfinite(self.timestamp) and self.timestamp >= 0

    @property
    def payload_length(self) -> int:
        if self.data is not None:
            return int(self.data.size)
        if self.text is not None:
            return len(self.text)
        return 0


def parse_i2c_data(text: str) -> list[I2CPacket]:
    """Parse an ``I2CData`` page-header value into a list of `I2CPacket`.

    Handles both documented on-disk flavours::

        I2CData = {{ts, [b1 b2 ... bN]} ... }   # I2CStoreAsChar = false
        I2CData = {{ts, 'my data'} ... }        # I2CStoreAsChar = true

    and returns ``[]`` for the empty ``'{}'`` case. Negative timestamps are
    *kept* (flagged via `I2CPacket.is_valid_timestamp`) rather than dropped,
    so callers decide whether to filter them.
    """
    if not text:
        return []

    packets: list[I2CPacket] = []
    for match in _I2C_PACKET_RE.finditer(str(text)):
        timestamp = float(match.group("timestamp"))
        payload = match.group("payload").strip()
        data: np.ndarray | None = None
        string: str | None = None

        if payload.startswith("[") or _NUMBER_RE.match(payload):
            byte_values = parse_numeric_array(payload)
            data = byte_values.astype(np.uint8) if byte_values.size else np.empty(0, np.uint8)
        else:
            string = payload[1:-1]

        packets.append(I2CPacket(timestamp=timestamp, data=data, text=string, raw=match.group(0)))
    return packets


def parse_epoch(text: str) -> datetime | None:
    """Parse a ScanImage ``epoch`` value into a `datetime`.

    The on-disk form is a 6-element MATLAB vector
    ``[YYYY MM DD hh mm ss.fff]`` (with the seconds fractional), e.g.
    ``'[2026  8  3 12 35 19.847]'``. Returns `None` when the value is
    absent or cannot be interpreted, since the epoch is informational and
    should never abort a metadata read.
    """
    values = parse_numeric_array(text)
    if values.size < 6:
        return None
    year, month, day, hour, minute = (int(v) for v in values[:5])
    seconds = float(values[5])
    try:
        # The seconds field is fractional, and rounding it can reach a full
        # 60 - which `datetime` rejects - so add it as a duration and let
        # `timedelta` carry into the minute (and beyond) for us.
        return datetime(year, month, day, hour, minute) + timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        return None


def parse_matlab_scalar(text: str) -> object:
    """Best-effort typing of an arbitrary MATLAB-printed header value.

    Used for page-header keys this tool does not know about, so that
    metadata from a newer ScanImage version still comes out usefully typed
    instead of being dropped. Falls back to returning the original string,
    which is always a valid (if unhelpful) answer.
    """
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return ""

    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in _NONFINITE:
        return _NONFINITE[lowered]
    if _INT_RE.match(stripped):
        return int(stripped)
    if _NUMBER_RE.match(stripped):
        return float(stripped)
    if (stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2) or (
        stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2
    ):
        return stripped[1:-1]
    if stripped.startswith("[") and stripped.endswith("]"):
        return parse_numeric_array(stripped).tolist()
    return stripped
