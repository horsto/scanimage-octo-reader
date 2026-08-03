"""Sweeping the *per-page* ScanImage headers - the core of this tool.

Everything ScanImage records about individual frames, including the vDAQ
AUX trigger timestamps and I2C packets, lives only in each page's
``ImageDescription`` tag. Recovering an event timeline therefore means
visiting every page. This is cheap - only tags are read, never pixel data -
so a 10 GB / 20 000-page file sweeps in about a second.

Parsing strategy: split lines with `str.partition('=')` and convert the
known keys with purpose-built converters. `tifffile.matlabstr2py` would also
parse these headers (and is used for the *global* header), but it mis-parses
exactly the fields that matter most here - see
`scanimage_octo_reader.parsers` for details. Unknown keys are retained as
raw strings, so a newer ScanImage version yields extra information rather
than an error.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from tifffile import TiffFile, TiffPage

from scanimage_octo_reader.geometry import PageMap, ScanGeometry
from scanimage_octo_reader.parsers import I2CPacket, parse_aux_trigger, parse_i2c_data

logger = logging.getLogger(__name__)

__all__ = [
    "AUX_LINES",
    "FRAME_DTYPE",
    "I2CRecord",
    "PageSweep",
    "aux_event_dtype",
    "read_page_descriptions",
    "sweep_pages",
]

# ScanImage supports four auxiliary digital trigger lines.
AUX_LINES = (0, 1, 2, 3)

# Scalar per-page header keys as (header key, output field, dtype, fill).
# The fill value is what an absent or unparseable entry becomes, chosen so
# it cannot be mistaken for data - note ScanImage itself already uses -1 as
# a "not applicable" timestamp sentinel.
_SCALAR_FIELDS: tuple[tuple[str, str, str, float], ...] = (
    ("frameNumbers", "frame_number", "i8", -1),
    ("acquisitionNumbers", "acquisition_number", "i4", -1),
    ("frameNumberAcquisition", "frame_number_acquisition", "i8", -1),
    ("frameTimestamps_sec", "frame_timestamp_s", "f8", np.nan),
    ("acqTriggerTimestamps_sec", "acq_trigger_timestamp_s", "f8", np.nan),
    ("nextFileMarkerTimestamps_sec", "next_file_marker_timestamp_s", "f8", np.nan),
    ("endOfAcquisition", "end_of_acquisition", "i1", -1),
    ("endOfAcquisitionMode", "end_of_acquisition_mode", "i1", -1),
    ("dcOverVoltage", "dc_over_voltage", "i1", -1),
)

FRAME_DTYPE = np.dtype(
    [("page_index", "i8")]
    + [(name, dtype) for _key, name, dtype, _fill in _SCALAR_FIELDS]
    + [
        ("channel", "i2"),
        ("slice_index", "i2"),
        ("frame_repeat_index", "i2"),
        ("volume_index", "i8"),
        # Timestamp of the volume this page belongs to, i.e. of the volume's
        # first page. For volumetric data this - not the per-plane timestamp -
        # is the timeline that per-cell activity actually lives on.
        ("volume_timestamp_s", "f8"),
        ("is_flyback", "?"),
    ]
    + [(f"n_aux{line}", "i2") for line in AUX_LINES]
    + [("n_i2c", "i2"), ("file_index", "i2")]
)


def aux_event_dtype() -> np.dtype:
    """Structured dtype of an AUX trigger event table.

    One row per detected trigger, carrying the frame context it was logged
    in - which is the whole reason ScanImage writes these into per-page
    rather than global metadata.

    Both timebases are provided, because which one is meaningful depends on
    the analysis. `timestamp_s` is the FPGA timestamp of the trigger itself,
    accurate to the sample period. Against it, `frame_timestamp_s` /
    `offset_in_frame_s` locate the event within the *plane* that was being
    scanned, while `volume_timestamp_s` / `offset_in_volume_s` locate it
    within the *volume* - the relevant one when the signal of interest is
    per-cell activity sampled once per volume. For a single-plane
    acquisition the two coincide.
    """
    return np.dtype(
        [
            ("timestamp_s", "f8"),
            ("page_index", "i8"),
            ("frame_number", "i8"),
            ("frame_timestamp_s", "f8"),
            ("offset_in_frame_s", "f8"),
            ("volume_index", "i8"),
            ("volume_timestamp_s", "f8"),
            ("offset_in_volume_s", "f8"),
            ("slice_index", "i2"),
            ("channel", "i2"),
            ("file_index", "i2"),
        ]
    )


@dataclass
class I2CRecord:
    """One I2C packet together with the frame context it was recorded in.

    Carries both timebases for the same reason as `aux_event_dtype`: the
    plane that was being scanned, and the volume it belongs to.
    """

    packet: I2CPacket
    page_index: int
    frame_number: int
    frame_timestamp_s: float
    volume_index: int
    volume_timestamp_s: float
    slice_index: int
    channel: int
    file_index: int


@dataclass
class PageSweep:
    """Result of sweeping every page header of one file (or merged acquisition)."""

    frames: np.ndarray
    aux: dict[int, np.ndarray] = field(default_factory=dict)
    i2c: list[I2CRecord] = field(default_factory=list)
    # Distinct page-header key sets encountered, with page counts. More than
    # one entry means the header layout changed mid-file, which `qc` reports.
    key_sets: dict[tuple[str, ...], int] = field(default_factory=dict)
    # Keys with no dedicated converter, with one example value each.
    unknown_keys: dict[str, str] = field(default_factory=dict)
    # Pages found by following the IFD chain past what tifffile reported
    # (see `read_page_descriptions`); normally zero.
    n_recovered_pages: int = 0

    @property
    def n_pages(self) -> int:
        return int(self.frames.size)

    def aux_counts(self) -> dict[int, int]:
        """Number of trigger events per AUX line (only non-empty lines appear)."""
        return {line: int(events.size) for line, events in self.aux.items()}


def _parse_description(description: str) -> dict[str, str]:
    """Split a page description into ``{key: raw value}`` without typing values."""
    fields: dict[str, str] = {}
    for line in description.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _to_int(text: str, fill: int) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return fill


def _to_float(text: str, fill: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return fill


def read_page_descriptions(tif: TiffFile) -> tuple[list[str], int]:
    """Read every page's ``ImageDescription``, reading tags only (no pixel data).

    Returns ``(descriptions, n_recovered)``, where `n_recovered` counts pages
    found beyond the ones `tifffile` reported.

    Two `tifffile` behaviours have to be worked around here. First, for files
    whose pages share a common layout, `tifffile` may return lightweight
    `TiffFrame` objects that carry data offsets but no tags. For ScanImage
    that would discard every frame header, because the interesting per-page
    metadata differs even when the image layout does not; `TiffPages.get(...,
    aspage=True)` forces a full IFD read.

    Second, `tifffile` can under-report the page count of SI-style classic
    TIFFs by one (reproducible with a `Software` tag beginning with `SI.`).
    Losing the last frame of an acquisition is silent and hard to spot from
    frame numbering, so the IFD chain is followed past the reported end and
    any further pages are recovered.
    """
    pages = tif.pages
    descriptions: list[str] = []
    for index in range(len(pages)):
        page = pages.get(index, aspage=True)
        descriptions.append(getattr(page, "description", "") or "")

    recovered = _recover_trailing_pages(tif, len(descriptions))
    descriptions.extend(recovered)
    return descriptions, len(recovered)


def _next_ifd_offset(tif: TiffFile, offset: int) -> int:
    """Return the offset of the IFD after the one at `offset` (0 if last)."""
    tiff = tif.tiff
    handle = tif.filehandle
    handle.seek(offset)
    tag_count = struct.unpack(tiff.tagnoformat, handle.read(tiff.tagnosize))[0]
    handle.seek(offset + tiff.tagnosize + tag_count * tiff.tagsize)
    return int(struct.unpack(tiff.offsetformat, handle.read(tiff.offsetsize))[0])


def _recover_trailing_pages(tif: TiffFile, n_reported: int) -> list[str]:
    """Follow the IFD chain past the last reported page and read what remains."""
    if n_reported == 0:
        return []

    try:
        last_offset = int(tif.pages[n_reported - 1].offset)
        offset = _next_ifd_offset(tif, last_offset)
    except Exception as exc:  # noqa: BLE001 - recovery must never break a read
        logger.debug("could not inspect the IFD chain past page %d: %s", n_reported - 1, exc)
        return []

    descriptions: list[str] = []
    index = n_reported
    seen = {last_offset}
    while offset and offset not in seen and offset < tif.filehandle.size:
        seen.add(offset)
        try:
            tif.filehandle.seek(offset)
            page = TiffPage(tif, index=index)
            descriptions.append(page.description or "")
            offset = _next_ifd_offset(tif, offset)
        except Exception as exc:  # noqa: BLE001 - a truncated tail is not fatal
            logger.warning("stopped recovering trailing pages at IFD %d: %s", index, exc)
            break
        index += 1

    if descriptions:
        logger.info(
            "recovered %d page(s) that tifffile did not report (it stopped at %d)",
            len(descriptions),
            n_reported,
        )
    return descriptions


def sweep_pages(
    tif: TiffFile,
    geometry: ScanGeometry,
    file_index: int = 0,
    page_offset: int = 0,
    progress: Callable[[int], None] | None = None,
) -> PageSweep:
    """Read every page header of `tif` into a frame table and event tables.

    `page_offset` and `file_index` let several files of a split acquisition
    be swept into one continuous timeline (see
    `scanimage_octo_reader.acquisition`); the defaults are right for a single
    file. Only page indices need offsetting - ScanImage's own frame numbers
    and frame timestamps already run continuously across the files of one
    acquisition (verified on a split sample: file 2 starts at frame 2001 and
    131.24 s where file 1 ends at frame 2000 and 131.17 s).

    `progress`, when given, is called with the number of pages completed, at
    coarse intervals, so a CLI can render a progress bar without paying
    per-page callback overhead.
    """
    descriptions, n_recovered_pages = read_page_descriptions(tif)
    n_pages = len(descriptions)
    page_map = geometry.page_map(n_pages)

    frames = np.zeros(n_pages, dtype=FRAME_DTYPE)
    frames["page_index"] = np.arange(n_pages, dtype=np.int64) + page_offset
    frames["file_index"] = file_index
    frames["channel"] = page_map.channel
    frames["slice_index"] = page_map.slice_index
    frames["frame_repeat_index"] = page_map.frame_repeat_index
    frames["volume_index"] = page_map.volume_index
    frames["is_flyback"] = page_map.is_flyback

    aux_timestamps: dict[int, list[np.ndarray]] = {line: [] for line in AUX_LINES}
    aux_pages: dict[int, list[np.ndarray]] = {line: [] for line in AUX_LINES}
    # Collected during the sweep, then turned into records once the volume
    # timestamps are known (they need the whole frame table).
    i2c_packets: list[I2CPacket] = []
    i2c_pages: list[int] = []
    key_sets: dict[tuple[str, ...], int] = {}
    unknown_keys: dict[str, str] = {}

    known_keys = {key for key, _name, _dtype, _fill in _SCALAR_FIELDS}
    known_keys |= {f"auxTrigger{line}" for line in AUX_LINES}
    known_keys |= {"I2CData", "epoch"}

    for page_index, description in enumerate(descriptions):
        parsed = _parse_description(description)
        signature = tuple(parsed)
        key_sets[signature] = key_sets.get(signature, 0) + 1

        for key, name, dtype, fill in _SCALAR_FIELDS:
            raw = parsed.get(key)
            if raw is None:
                frames[name][page_index] = fill
            elif dtype.startswith("f"):
                frames[name][page_index] = _to_float(raw, float(fill))
            else:
                frames[name][page_index] = _to_int(raw, int(fill))

        for line in AUX_LINES:
            raw = parsed.get(f"auxTrigger{line}")
            if not raw or raw == "[]":
                continue
            timestamps = parse_aux_trigger(raw)
            if timestamps.size:
                aux_timestamps[line].append(timestamps)
                aux_pages[line].append(np.full(timestamps.size, page_index, np.int64))
                frames[f"n_aux{line}"][page_index] = timestamps.size

        raw_i2c = parsed.get("I2CData")
        if raw_i2c and raw_i2c not in ("{}", "[]"):
            packets = parse_i2c_data(raw_i2c)
            frames["n_i2c"][page_index] = len(packets)
            i2c_pages.extend([page_index] * len(packets))
            i2c_packets.extend(packets)

        for key, value in parsed.items():
            if key not in known_keys and key not in unknown_keys:
                unknown_keys[key] = value

        if progress is not None and page_index % 2000 == 1999:
            progress(page_index + 1)

    if progress is not None:
        progress(n_pages)

    # A volume's timestamp is that of its first page, so every page (flyback
    # included) can be placed on the volume timeline that per-cell activity
    # is sampled on.
    frames["volume_timestamp_s"] = _volume_timestamps(frames, geometry.pages_per_volume)

    i2c_records = [
        I2CRecord(
            packet=packet,
            page_index=page_index + page_offset,
            frame_number=int(frames["frame_number"][page_index]),
            frame_timestamp_s=float(frames["frame_timestamp_s"][page_index]),
            volume_index=int(page_map.volume_index[page_index]),
            volume_timestamp_s=float(frames["volume_timestamp_s"][page_index]),
            slice_index=int(page_map.slice_index[page_index]),
            channel=int(page_map.channel[page_index]),
            file_index=file_index,
        )
        for page_index, packet in zip(i2c_pages, i2c_packets)
    ]

    aux = {
        line: _build_aux_table(
            aux_timestamps[line], aux_pages[line], frames, page_map, file_index, page_offset
        )
        for line in AUX_LINES
        if aux_timestamps[line]
    }

    if unknown_keys:
        logger.info(
            "page headers contain %d key(s) without a dedicated converter: %s",
            len(unknown_keys),
            ", ".join(sorted(unknown_keys)),
        )

    return PageSweep(
        frames=frames,
        aux=aux,
        i2c=i2c_records,
        key_sets=key_sets,
        unknown_keys=unknown_keys,
        n_recovered_pages=n_recovered_pages,
    )


def _volume_timestamps(frames: np.ndarray, pages_per_volume: int) -> np.ndarray:
    """Return, for every page, the frame timestamp of its volume's first page.

    For a single-plane acquisition (`pages_per_volume == 1`) this is just the
    frame timestamp itself.
    """
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    stride = max(pages_per_volume, 1)
    if stride == 1:
        return frames["frame_timestamp_s"].astype(np.float64, copy=True)
    first_page_of_volume = (np.arange(frames.size, dtype=np.int64) // stride) * stride
    return frames["frame_timestamp_s"][first_page_of_volume]


def _build_aux_table(
    timestamp_chunks: list[np.ndarray],
    page_chunks: list[np.ndarray],
    frames: np.ndarray,
    page_map: PageMap,
    file_index: int,
    page_offset: int,
) -> np.ndarray:
    """Assemble one AUX line's events into a structured table."""
    timestamps = np.concatenate(timestamp_chunks)
    pages = np.concatenate(page_chunks)

    events = np.zeros(timestamps.size, dtype=aux_event_dtype())
    events["timestamp_s"] = timestamps
    events["page_index"] = pages + page_offset
    events["frame_number"] = frames["frame_number"][pages]
    events["frame_timestamp_s"] = frames["frame_timestamp_s"][pages]
    events["offset_in_frame_s"] = timestamps - frames["frame_timestamp_s"][pages]
    events["volume_index"] = page_map.volume_index[pages]
    events["volume_timestamp_s"] = frames["volume_timestamp_s"][pages]
    events["offset_in_volume_s"] = timestamps - frames["volume_timestamp_s"][pages]
    events["slice_index"] = page_map.slice_index[pages]
    events["channel"] = page_map.channel[pages]
    events["file_index"] = file_index
    return events
