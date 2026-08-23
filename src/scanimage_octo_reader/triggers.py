"""Turning swept AUX/I2C events into tables that are pleasant to load later.

The output format goal is "openable with a bare `np.load`": structured
arrays with named fields, fixed dtypes, and **no** pickled objects. I2C
payloads are variable length, which does not fit a fixed dtype, so they are
split into a fixed-width table (timing plus frame context plus payload
length) and a separate padded ``uint8`` matrix, with the verbatim source
strings kept alongside as JSON for full fidelity.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import numpy as np

from scanimage_octo_reader.page_headers import I2CRecord
from scanimage_octo_reader.parsers import I2CPacket

logger = logging.getLogger(__name__)

__all__ = [
    "I2C_TABLE_DTYPE",
    "aux_summary",
    "decode_i2c_key_values",
    "decode_i2c_payload_text",
    "i2c_payload_matrix",
    "i2c_summary",
    "i2c_table",
]

I2C_TABLE_DTYPE = np.dtype(
    [
        ("timestamp_s", "f8"),
        ("page_index", "i8"),
        ("frame_number", "i8"),
        ("frame_timestamp_s", "f8"),
        ("offset_in_frame_s", "f8"),
        ("volume_index", "i8"),
        # See `page_headers.aux_event_dtype` on why both timebases are given.
        ("volume_timestamp_s", "f8"),
        ("offset_in_volume_s", "f8"),
        ("slice_index", "i2"),
        ("channel", "i2"),
        ("file_index", "i2"),
        # ScanImage uses negative timestamps as sentinels; packets are kept
        # either way and flagged here, so filtering stays the caller's choice.
        ("valid", "?"),
        ("payload_length", "i4"),
        # 'bytes' (I2CStoreAsChar = false) or 'text' (= true).
        ("payload_kind", "S5"),
    ]
)

# The lab convention of encoding "<key>_<value>" into a char payload, e.g.
# 'treadmill_9'. The value must be numeric for the decode to be meaningful.
_KEY_VALUE_RE = re.compile(r"^(?P<key>.+)_(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$")


def i2c_table(records: Sequence[I2CRecord]) -> np.ndarray:
    """Build the fixed-width I2C packet table (timing + frame context)."""
    table = np.zeros(len(records), dtype=I2C_TABLE_DTYPE)
    for index, record in enumerate(records):
        packet = record.packet
        table[index] = (
            packet.timestamp,
            record.page_index,
            record.frame_number,
            record.frame_timestamp_s,
            packet.timestamp - record.frame_timestamp_s,
            record.volume_index,
            record.volume_timestamp_s,
            packet.timestamp - record.volume_timestamp_s,
            record.slice_index,
            record.channel,
            record.file_index,
            packet.is_valid_timestamp,
            packet.payload_length,
            b"text" if packet.text is not None else b"bytes",
        )
    return table


def i2c_payload_matrix(records: Sequence[I2CRecord]) -> np.ndarray:
    """Return byte payloads as a zero-padded ``(n_packets, max_length)`` ``uint8`` array.

    Padded rather than ragged so the result is a plain array that
    `numpy.load` reads without ``allow_pickle``. Use the companion table's
    ``payload_length`` field to trim each row back to its true length.
    Text-flavour payloads are encoded as UTF-8 bytes, so this matrix is
    populated for both I2C flavours.
    """
    payloads: list[np.ndarray] = []
    for record in records:
        packet = record.packet
        if packet.data is not None:
            payloads.append(np.asarray(packet.data, dtype=np.uint8))
        elif packet.text is not None:
            payloads.append(np.frombuffer(packet.text.encode("utf-8"), dtype=np.uint8))
        else:
            payloads.append(np.empty(0, dtype=np.uint8))

    width = max((payload.size for payload in payloads), default=0)
    matrix = np.zeros((len(payloads), width), dtype=np.uint8)
    for index, payload in enumerate(payloads):
        matrix[index, : payload.size] = payload
    return matrix


def _decode_payload_text(packet: I2CPacket) -> tuple[str, bool]:
    """Best-effort UTF-8 decode of one packet's raw payload bytes.

    A text-flavour payload (``I2CStoreAsChar = true``) is already a decoded
    string, so this always succeeds for it. A byte-flavour payload is
    decoded too, on the chance it carries printable text over a
    byte-oriented channel (e.g. ``I2CStoreAsChar = false`` with an ASCII
    reading like this module's own docstring example), but only when *every*
    byte is valid UTF-8 - a genuinely binary payload returns ``("", False)``
    rather than a lossy, potentially misleading guess.
    """
    if packet.text is not None:
        return packet.text, True
    if packet.data is not None and packet.data.size:
        try:
            return bytes(packet.data.tolist()).decode("utf-8"), True
        except UnicodeDecodeError:
            return "", False
    return "", False


def decode_i2c_payload_text(records: Sequence[I2CRecord]) -> np.ndarray:
    """Decode every I2C payload's raw bytes to UTF-8 text, one row per packet.

    Unlike `decode_i2c_key_values`, this makes no assumption about the
    payload's internal structure - no ``'<key>_<value>'`` convention
    required - so it covers arbitrary text payloads (e.g. a plain sensor
    reading like ``'T-42.61'``). `decoded` is `False` (and `text` empty)
    for a payload that could not be decoded as UTF-8 at all, so a failed
    decode is never confused with a genuinely empty payload.

    Returns an array with a `text` field sized to fit the longest decoded
    string (min 1, so the dtype is always valid), even for zero records -
    a plain, pickle-free structure a bare `numpy.load` can read.
    """
    decoded = [_decode_payload_text(record.packet) for record in records]
    max_length = max((len(text) for text, ok in decoded if ok), default=0)
    dtype = np.dtype(
        [
            ("timestamp_s", "f8"),
            ("page_index", "i8"),
            ("frame_number", "i8"),
            ("valid", "?"),
            # 'bytes' (I2CStoreAsChar = false) or 'text' (= true) - the
            # *storage* flavour, independent of whether `decoded` succeeded.
            ("payload_kind", "S5"),
            ("decoded", "?"),
            ("text", f"<U{max(max_length, 1)}"),
        ]
    )
    table = np.zeros(len(records), dtype=dtype)
    for index, (record, (text, ok)) in enumerate(zip(records, decoded, strict=True)):
        packet = record.packet
        table[index] = (
            packet.timestamp,
            record.page_index,
            record.frame_number,
            packet.is_valid_timestamp,
            b"text" if packet.text is not None else b"bytes",
            ok,
            text,
        )
    return table


def decode_i2c_key_values(
    records: Sequence[I2CRecord],
) -> tuple[dict[str, np.ndarray], str | None]:
    """Decode text payloads following the ``'<key>_<value>'`` convention.

    Returns ``(decoded, reason)`` where `decoded` maps each key to a
    structured array of its numeric values over time, and `reason` explains
    why decoding was skipped (``None`` on success). Decoding is all-or-
    nothing on purpose: if even one payload does not match the convention,
    the data is probably not using it, and a partial decode would be a
    silently misleading export.
    """
    if not records:
        return {}, "no I2C packets to decode"

    non_text = sum(1 for record in records if record.packet.text is None)
    if non_text:
        return {}, (
            f"{non_text} of {len(records)} packets store raw bytes rather than text, so the "
            "'<key>_<value>' convention does not apply"
        )

    dtype = np.dtype(
        [
            ("timestamp_s", "f8"),
            ("value", "f8"),
            ("page_index", "i8"),
            ("frame_number", "i8"),
            ("valid", "?"),
        ]
    )

    grouped: dict[str, list[tuple]] = {}
    for record in records:
        match = _KEY_VALUE_RE.match(record.packet.text or "")
        if not match:
            return {}, (
                f"payload {record.packet.text!r} does not match the '<key>_<value>' "
                "convention; skipping the decode rather than exporting a partial result"
            )
        grouped.setdefault(match["key"], []).append(
            (
                record.packet.timestamp,
                float(match["value"]),
                record.page_index,
                record.frame_number,
                record.packet.is_valid_timestamp,
            )
        )

    return {key: np.array(rows, dtype=dtype) for key, rows in grouped.items()}, None


def _describe_events(timestamps: np.ndarray) -> dict[str, object]:
    """Summarise one event timeline: count, span and regularity."""
    finite = timestamps[np.isfinite(timestamps)]
    summary: dict[str, object] = {
        "n_events": int(timestamps.size),
        "n_nonfinite": int(timestamps.size - finite.size),
        "n_negative": int(np.count_nonzero(finite < 0)),
    }
    if finite.size:
        summary["first_timestamp_s"] = float(finite.min())
        summary["last_timestamp_s"] = float(finite.max())
    if finite.size > 1:
        intervals = np.diff(np.sort(finite))
        summary["median_interval_s"] = float(np.median(intervals))
        summary["min_interval_s"] = float(intervals.min())
        summary["max_interval_s"] = float(intervals.max())
    return summary


def aux_summary(aux: dict[int, np.ndarray]) -> dict[str, object]:
    """Summarise the AUX trigger lines for the exported metadata."""
    summary: dict[str, object] = {}
    for line, events in sorted(aux.items()):
        summary[f"aux{line}"] = _describe_events(events["timestamp_s"])
    return summary


def i2c_summary(records: Sequence[I2CRecord]) -> dict[str, object]:
    """Summarise I2C packets: counts, flavour, payload sizes and decoded keys."""
    if not records:
        return {"n_packets": 0}

    timestamps = np.array([record.packet.timestamp for record in records], dtype=np.float64)
    summary = _describe_events(timestamps)
    summary["n_packets"] = summary.pop("n_events")

    n_text = sum(1 for record in records if record.packet.text is not None)
    summary["payload_kind"] = (
        "text" if n_text == len(records) else "bytes" if n_text == 0 else "mixed"
    )
    lengths = np.array([record.packet.payload_length for record in records], dtype=np.int64)
    summary["min_payload_length"] = int(lengths.min())
    summary["max_payload_length"] = int(lengths.max())
    summary["n_payloads_decoded_as_text"] = sum(
        1 for record in records if _decode_payload_text(record.packet)[1]
    )

    decoded, reason = decode_i2c_key_values(records)
    if decoded:
        summary["decoded_keys"] = {key: int(values.size) for key, values in decoded.items()}
    elif reason:
        summary["decode_skipped"] = reason
    return summary
