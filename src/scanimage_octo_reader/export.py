"""Writing a swept recording to disk.

Layout, one directory per recording (named after the TIFF stem, or the shared
stem of a merged acquisition)::

    <out>/<name>/
      metadata.json              global metadata, summary, trigger inventory
      frames.npy                 structured per-page table
      aux/aux0.npy .. aux3.npy   per-line trigger event tables (non-empty only)
      i2c/packets.npy            packet timing + frame context + payload length
      i2c/payloads.npy           zero-padded uint8 payload matrix
      i2c/payload_text.npy       payload bytes decoded to UTF-8 text, where possible
      i2c/packets_raw.json       verbatim source strings
      i2c/decoded_<key>.npy      optional '<key>_<value>' decode
      plots/overview.png         frame timeline + trigger overview
      manifest.json              what this run wrote, plus the QC report

Everything is loadable with a bare `numpy.load` (no ``allow_pickle``) or
`json.load`.

JSON cannot represent ``inf``/``NaN``, which real ScanImage headers do
contain (e.g. ``SI.hBeams.lengthConstants = inf``). Rather than emit the
non-standard ``Infinity`` literal that `json.dump` produces by default -
which many strict parsers reject - non-finite floats are written as the
strings ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``, and the metadata
declares this under ``tool.nonfinite_encoding``. Serialisation runs with
``allow_nan=False`` so any value that escaped the conversion fails loudly
instead of producing invalid JSON.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scanimage_octo_reader.acquisition import Recording
from scanimage_octo_reader.header import nest_dotted_keys
from scanimage_octo_reader.qc import QCReport, check_recording
from scanimage_octo_reader.triggers import (
    decode_i2c_key_values,
    decode_i2c_payload_text,
    i2c_payload_matrix,
    i2c_table,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NONFINITE_ENCODING",
    "ExportResult",
    "build_metadata",
    "default_output_root",
    "export_recording",
    "to_jsonable",
    "write_json",
]

NONFINITE_ENCODING = "string"

_NONFINITE_LABELS = {
    math.inf: "Infinity",
    -math.inf: "-Infinity",
}


def to_jsonable(value: Any) -> Any:
    """Convert `value` into something `json.dump` accepts with ``allow_nan=False``.

    Handles the types that turn up in ScanImage metadata and numpy output:
    nested dicts/sequences, numpy scalars and arrays, `datetime`, `Path`,
    `bytes`, and non-finite floats (see the module docstring).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return _NONFINITE_LABELS[value]
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [to_jsonable(item) for item in items]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write `payload` as indented, strictly-valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _source_info(paths: list[Path]) -> list[dict[str, Any]]:
    sources = []
    for path in paths:
        entry: dict[str, Any] = {"path": str(path.resolve()), "filename": path.name}
        try:
            stat = path.stat()
            entry["size_bytes"] = stat.st_size
            entry["modified"] = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        except OSError:
            pass
        sources.append(entry)
    return sources


def build_metadata(recording: Recording, include_rois: bool = True, flat: bool = False) -> dict:
    """Assemble the full metadata document for `recording`.

    With `flat`, the ``SI.*`` keys are kept exactly as ScanImage wrote them
    (handy for grepping or for comparing against MATLAB); by default they are
    nested into a tree, which is far easier to read for 425 keys.
    """
    from scanimage_octo_reader import __version__

    header = recording.header
    scanimage: dict[str, Any] = {
        "header_version": header.version,
        "si_version": header.si_version,
        "FrameData": header.frame_data if flat else nest_dotted_keys(header.frame_data),
        "framedata_layout": "flat" if flat else "nested",
    }
    if include_rois:
        scanimage["RoiGroups"] = header.roi_groups

    return {
        "tool": {
            "name": "scanimage-octo-reader",
            "version": __version__,
            "created": datetime.now(UTC).isoformat(),
            # JSON has no literal for inf/NaN; see the module docstring.
            "nonfinite_encoding": NONFINITE_ENCODING,
        },
        "source": {
            "files": _source_info(recording.paths),
            "merged_acquisition": len(recording.paths) > 1,
        },
        "summary": recording.summary(),
        "triggers": recording.trigger_summary(),
        "warnings": recording.warnings,
        "scanimage": scanimage,
        "tiff_tags": recording.tiff_tags,
    }


@dataclass
class ExportResult:
    """What an export run produced."""

    directory: Path
    files: list[Path] = field(default_factory=list)
    qc: QCReport | None = None

    def relative_files(self) -> list[str]:
        return [str(path.relative_to(self.directory)) for path in self.files]


def _guard_output_dir(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{directory} already exists and is not empty; pass overwrite to replace its contents"
        )
    directory.mkdir(parents=True, exist_ok=True)


def default_output_root(recording: Recording) -> Path:
    """Where a recording's outputs go when no location is given.

    Next to the TIFF being processed, not in the current working directory:
    exports belong with the data they describe, and a raw-data directory is
    usually the one place both are guaranteed to travel together.
    """
    return recording.paths[0].parent


def export_recording(
    recording: Recording,
    out_dir: str | Path | None = None,
    write_metadata: bool = True,
    write_frames: bool = True,
    write_aux: bool = True,
    write_i2c: bool = True,
    decode_i2c: bool = False,
    make_plot: bool = True,
    include_rois: bool = True,
    flat_header: bool = False,
    plot_formats: str | Sequence[str] | None = None,
    dpi: int = 150,
    overwrite: bool = False,
    run_qc: bool = True,
) -> ExportResult:
    """Write the requested products of `recording` into ``out_dir/<name>/``.

    `out_dir` defaults to the directory holding the source TIFF, so the
    export lands in ``<tif directory>/<name>/``.
    """
    root = default_output_root(recording) if out_dir is None else Path(out_dir)
    directory = root / recording.name
    _guard_output_dir(directory, overwrite)

    result = ExportResult(directory=directory)

    if run_qc:
        result.qc = check_recording(recording)

    if write_metadata:
        path = directory / "metadata.json"
        write_json(path, build_metadata(recording, include_rois=include_rois, flat=flat_header))
        result.files.append(path)

    if write_frames:
        path = directory / "frames.npy"
        np.save(path, recording.frames)
        result.files.append(path)

    if write_aux:
        result.files.extend(_write_aux(recording, directory))

    if write_i2c:
        result.files.extend(_write_i2c(recording, directory, decode_i2c=decode_i2c))

    if make_plot:
        # Imported lazily so that metadata-only use never pays matplotlib's
        # import cost (a noticeable fraction of a second).
        from scanimage_octo_reader.plots import DEFAULT_FORMATS, save_overview_figure

        try:
            result.files.extend(
                save_overview_figure(
                    recording,
                    directory / "plots",
                    formats=plot_formats or DEFAULT_FORMATS,
                    dpi=dpi,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the export
            logger.warning("could not render the overview plot: %s", exc)

    manifest_path = directory / "manifest.json"
    write_json(
        manifest_path,
        {
            "created": datetime.now(UTC).isoformat(),
            "recording": recording.name,
            "source_files": [str(path.resolve()) for path in recording.paths],
            "n_pages": recording.n_pages,
            "pages_per_file": recording.pages_per_file,
            "aux_event_counts": {
                f"aux{line}": int(events.size) for line, events in sorted(recording.aux.items())
            },
            "n_i2c_packets": len(recording.i2c),
            "files": result.relative_files(),
            "qc": result.qc.as_dict() if result.qc else None,
        },
    )
    result.files.append(manifest_path)
    return result


def _write_aux(recording: Recording, directory: Path) -> list[Path]:
    """Write one table per non-empty AUX line."""
    if not recording.aux:
        return []
    aux_dir = directory / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for line, events in sorted(recording.aux.items()):
        path = aux_dir / f"aux{line}.npy"
        np.save(path, events)
        written.append(path)
    return written


def _write_i2c(recording: Recording, directory: Path, decode_i2c: bool) -> list[Path]:
    """Write the I2C packet table, payload matrix/text, raw strings and optional decode."""
    records = recording.i2c
    if not records:
        return []

    i2c_dir = directory / "i2c"
    i2c_dir.mkdir(parents=True, exist_ok=True)
    written = []

    table_path = i2c_dir / "packets.npy"
    np.save(table_path, i2c_table(records))
    written.append(table_path)

    payload_path = i2c_dir / "payloads.npy"
    np.save(payload_path, i2c_payload_matrix(records))
    written.append(payload_path)

    # A general UTF-8 byte decode of every payload - unlike the `decode_i2c`
    # option below, this makes no assumption about the payload's internal
    # structure, so it is always written rather than gated behind a flag.
    payload_text_path = i2c_dir / "payload_text.npy"
    np.save(payload_text_path, decode_i2c_payload_text(records))
    written.append(payload_text_path)

    raw_path = i2c_dir / "packets_raw.json"
    write_json(
        raw_path,
        [
            {
                "page_index": record.page_index,
                "frame_number": record.frame_number,
                "raw": record.packet.raw,
            }
            for record in records
        ],
    )
    written.append(raw_path)

    if decode_i2c:
        decoded, reason = decode_i2c_key_values(records)
        if decoded:
            for key, values in decoded.items():
                safe_key = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in key
                )
                path = i2c_dir / f"decoded_{safe_key}.npy"
                np.save(path, values)
                written.append(path)
        elif reason:
            logger.warning("skipping the I2C '<key>_<value>' decode: %s", reason)

    return written
