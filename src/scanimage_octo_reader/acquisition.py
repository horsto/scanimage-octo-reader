"""Opening ScanImage TIFFs, and stitching split acquisitions back together.

ScanImage rolls over to a new file every ``logFramesPerFile`` frames, naming
the parts ``<base>_<acquisition>_<fileIndex>.tif`` (versus
``<base>_<acquisition>.tif`` for an acquisition that fits in one file). The
parts form one continuous recording: frame numbers and frame timestamps run
straight through them, so merging is a matter of concatenation plus a page
index offset - no renumbering.

`Recording` is the object the rest of the package (export, plots, QC, CLI)
works with, whether it wraps one file or several.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import TiffFile

from scanimage_octo_reader.geometry import ScanGeometry, compute_geometry
from scanimage_octo_reader.header import ScanImageHeader, read_header, read_tiff_tags
from scanimage_octo_reader.page_headers import AUX_LINES, I2CRecord, PageSweep, sweep_pages
from scanimage_octo_reader.parsers import parse_epoch
from scanimage_octo_reader.triggers import aux_summary, i2c_summary

logger = logging.getLogger(__name__)

__all__ = [
    "Recording",
    "find_acquisition_files",
    "natural_sort",
    "read_recording",
]

# `<base>_<acquisition>_<fileIndex>` (split) and `<base>_<acquisition>` (single).
_SPLIT_FILE_RE = re.compile(r"^(?P<base>.+?)_(?P<acquisition>\d+)_(?P<file_index>\d+)$")
_SINGLE_FILE_RE = re.compile(r"^(?P<base>.+?)_(?P<acquisition>\d+)$")

# Header keys that legitimately differ between files of the same acquisition,
# and so must be ignored when checking that two files really belong together.
_PER_FILE_KEYS = frozenset(
    {
        "SI.hScan2D.logFileCounter",
        "SI.hScan2D.logFilePath",
        "SI.hScan2D.logFileStem",
    }
)

_NUMBER_RE = re.compile(r"(\d+)")


def natural_sort(paths: Sequence[Any]) -> list[Any]:
    """Sort paths in human order, so ``file2`` precedes ``file10``."""

    def key(value: Any) -> list[Any]:
        return [
            int(part) if part.isdigit() else part.lower() for part in _NUMBER_RE.split(str(value))
        ]

    return sorted(paths, key=key)


def _parse_filename(path: Path) -> tuple[str, str, int | None]:
    """Return ``(base, acquisition, file_index)``; `file_index` is None if not split."""
    stem = path.stem
    match = _SPLIT_FILE_RE.match(stem)
    if match:
        return match["base"], match["acquisition"], int(match["file_index"])
    match = _SINGLE_FILE_RE.match(stem)
    if match:
        return match["base"], match["acquisition"], None
    return stem, "", None


def _comparable(frame_data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in frame_data.items() if k not in _PER_FILE_KEYS}


def find_acquisition_files(path: str | Path) -> list[Path]:
    """Find and order all files belonging to the same acquisition as `path`.

    Returns ``[path]`` unless `path` uses the split naming form *and*
    siblings are found whose global header matches (ignoring the per-file
    bookkeeping keys). A name-based match alone is not enough: two unrelated
    acquisitions can easily produce similar names, and merging them would
    silently fabricate a timeline.
    """
    path = Path(path)
    base, acquisition, file_index = _parse_filename(path)
    if not acquisition or file_index is None:
        return [path]

    siblings: list[Path] = []
    for candidate in path.parent.glob(f"*{path.suffix}"):
        if not candidate.is_file():
            continue
        c_base, c_acquisition, c_index = _parse_filename(candidate)
        if c_base == base and c_acquisition == acquisition and c_index is not None:
            siblings.append(candidate)

    if len(siblings) <= 1:
        return [path]

    siblings = natural_sort(siblings)

    reference: dict[str, Any] | None = None
    confirmed: list[Path] = []
    for sibling in siblings:
        try:
            with TiffFile(sibling) as tif:
                frame_data = _comparable(read_header(tif).frame_data)
        except Exception as exc:  # noqa: BLE001 - a bad sibling must not sink the read
            logger.warning("could not inspect possible sibling %s: %s", sibling, exc)
            continue
        if reference is None:
            reference = frame_data
            confirmed.append(sibling)
        elif frame_data == reference:
            confirmed.append(sibling)
        else:
            logger.warning(
                "%s looks like a sibling of %s by name, but its ScanImage header differs; "
                "excluding it from the merged acquisition",
                sibling.name,
                path.name,
            )

    return confirmed or [path]


@dataclass
class Recording:
    """One ScanImage recording: a single file, or a merged split acquisition."""

    paths: list[Path]
    header: ScanImageHeader
    geometry: ScanGeometry
    sweep: PageSweep
    tiff_tags: dict[str, Any] = field(default_factory=dict)
    image_shape: tuple[int, ...] = ()
    dtype: str = ""
    is_bigtiff: bool = False
    pages_per_file: list[int] = field(default_factory=list)
    # Acquisition start wall-clock time, from the per-page `epoch` field.
    epoch: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Output-directory name: the file stem, or the shared stem when merged."""
        if len(self.paths) == 1:
            return self.paths[0].stem
        base, acquisition, _index = _parse_filename(self.paths[0])
        return f"{base}_{acquisition}" if acquisition else self.paths[0].stem

    @property
    def n_pages(self) -> int:
        return self.sweep.n_pages

    @property
    def frames(self) -> np.ndarray:
        return self.sweep.frames

    @property
    def aux(self) -> dict[int, np.ndarray]:
        return self.sweep.aux

    @property
    def i2c(self) -> list[I2CRecord]:
        return self.sweep.i2c

    @property
    def duration_s(self) -> float | None:
        """Span of the frame timestamps, in seconds."""
        timestamps = self.frames["frame_timestamp_s"]
        finite = timestamps[np.isfinite(timestamps)]
        if finite.size < 2:
            return None
        return float(finite.max() - finite.min())

    def summary(self) -> dict[str, Any]:
        """Compact, human-facing description of the recording."""
        header = self.header
        geometry = self.geometry
        return {
            "n_pages": self.n_pages,
            "n_files": len(self.paths),
            "pages_per_file": self.pages_per_file,
            "image_shape": list(self.image_shape),
            "dtype": self.dtype,
            "bigtiff": self.is_bigtiff,
            "n_channels": geometry.n_channels,
            "channels_saved": geometry.channels_saved,
            "volumetric": geometry.volumetric,
            "n_slices": geometry.n_slices,
            "frames_per_slice": geometry.frames_per_slice,
            "flyback_frames": geometry.flyback_frames,
            "pages_per_volume": geometry.pages_per_volume,
            "n_volumes": geometry.n_volumes(self.n_pages),
            "zs": geometry.zs,
            "frame_rate_hz": geometry.frame_rate_hz,
            "volume_rate_hz": geometry.volume_rate_hz,
            "duration_s": self.duration_s,
            "epoch": self.epoch.isoformat() if self.epoch else None,
            "zoom": header.get("SI.hRoiManager.scanZoomFactor"),
            "pixels_per_line": header.get("SI.hRoiManager.pixelsPerLine"),
            "lines_per_frame": header.get("SI.hRoiManager.linesPerFrame"),
            "objective_resolution": header.get("SI.objectiveResolution"),
            "scanner_type": header.get("SI.hScan2D.scannerType"),
            "scanner_name": header.get("SI.hScan2D.name"),
            "bidirectional": header.get("SI.hScan2D.bidirectional"),
            "mroi_enabled": bool(header.get("SI.hRoiManager.mroiEnable")),
            "si_version": header.si_version,
            "sample_position": header.get("SI.hMotors.samplePosition"),
            "axes_position": header.get("SI.hMotors.axesPosition"),
        }

    def trigger_summary(self) -> dict[str, Any]:
        """Summary of what was found on the AUX lines and the I2C bus."""
        return {
            "aux": aux_summary(self.aux),
            "aux_lines_present": sorted(self.aux),
            "aux_lines_empty": [line for line in AUX_LINES if line not in self.aux],
            "i2c": i2c_summary(self.i2c),
        }


def read_recording(
    path: str | Path | Sequence[str | Path],
    merge_acquisition: bool = False,
    progress: Callable[[int], None] | None = None,
) -> Recording:
    """Open one or more ScanImage TIFFs and sweep all their page headers.

    `path` may be a single path or an explicit, ordered list of paths. With
    `merge_acquisition` and a single path, the sibling files of a split
    acquisition are discovered automatically and merged into one continuous
    recording.
    """
    if isinstance(path, (str, Path)):
        paths = find_acquisition_files(path) if merge_acquisition else [Path(path)]
    else:
        paths = [Path(p) for p in path]
    if not paths:
        raise ValueError("no input files given")

    warnings: list[str] = []
    header: ScanImageHeader | None = None
    geometry: ScanGeometry | None = None
    tiff_tags: dict[str, Any] = {}
    image_shape: tuple[int, ...] = ()
    dtype = ""
    is_bigtiff = False
    epoch: datetime | None = None
    pages_per_file: list[int] = []

    frame_chunks: list[np.ndarray] = []
    aux_chunks: dict[int, list[np.ndarray]] = {}
    i2c_records: list[I2CRecord] = []
    key_sets: dict[tuple[str, ...], int] = {}
    unknown_keys: dict[str, str] = {}
    n_recovered_pages = 0
    page_offset = 0

    for file_index, file_path in enumerate(paths):
        with TiffFile(file_path) as tif:
            if not tif.is_scanimage:
                warnings.append(
                    f"{file_path.name} is not recognised by tifffile as a ScanImage file; "
                    "parsing it anyway, but the metadata may be incomplete"
                )
            file_header = read_header(tif)
            if header is None:
                header = file_header
                tiff_tags = read_tiff_tags(tif)
                page = tif.pages[0]
                image_shape = tuple(int(v) for v in page.shape)
                dtype = str(page.dtype)
                is_bigtiff = bool(tif.is_bigtiff)
                epoch = _epoch_from_description(page.description or "")
                # Geometry comes from the first file's header: every file of
                # an acquisition shares the configuration, and this keeps the
                # page->volume mapping identical across the merge.
                geometry = compute_geometry(header, len(tif.pages))
                if geometry.warning:
                    warnings.append(geometry.warning)
            elif _comparable(file_header.frame_data) != _comparable(header.frame_data):
                warnings.append(
                    f"{file_path.name} has a different ScanImage header from "
                    f"{paths[0].name}; the merged timeline may be meaningless"
                )

            assert geometry is not None
            file_sweep = sweep_pages(
                tif,
                geometry,
                file_index=file_index,
                page_offset=page_offset,
                progress=progress,
            )

        pages_per_file.append(file_sweep.n_pages)
        page_offset += file_sweep.n_pages
        frame_chunks.append(file_sweep.frames)
        for line, events in file_sweep.aux.items():
            aux_chunks.setdefault(line, []).append(events)
        i2c_records.extend(file_sweep.i2c)
        for signature, count in file_sweep.key_sets.items():
            key_sets[signature] = key_sets.get(signature, 0) + count
        for key, value in file_sweep.unknown_keys.items():
            unknown_keys.setdefault(key, value)
        n_recovered_pages += file_sweep.n_recovered_pages

    assert header is not None and geometry is not None

    sweep = PageSweep(
        frames=np.concatenate(frame_chunks) if frame_chunks else np.empty(0),
        aux={line: np.concatenate(chunks) for line, chunks in sorted(aux_chunks.items())},
        i2c=i2c_records,
        key_sets=key_sets,
        unknown_keys=unknown_keys,
        n_recovered_pages=n_recovered_pages,
    )

    return Recording(
        paths=paths,
        header=header,
        geometry=geometry,
        sweep=sweep,
        tiff_tags=tiff_tags,
        image_shape=image_shape,
        dtype=dtype,
        is_bigtiff=is_bigtiff,
        pages_per_file=pages_per_file,
        epoch=epoch,
        warnings=warnings,
    )


def _epoch_from_description(description: str) -> datetime | None:
    """Pull the ``epoch`` field out of a page description."""
    for line in description.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "epoch":
            return parse_epoch(value)
    return None
