"""Read ScanImage TIFF timeseries: structured metadata, AUX triggers and I2C.

`tifffile` is the only TIFF reader used. The typical entry point is
`read_recording`, which opens one file (or a merged split acquisition) and
sweeps every page header:

    from scanimage_octo_reader import read_recording

    recording = read_recording("LC_brain1__00001.tif")
    print(recording.summary())
    print(recording.aux[0]["timestamp_s"])  # AUX line 0 trigger times
"""

from __future__ import annotations

from scanimage_octo_reader.acquisition import (
    Recording,
    find_acquisition_files,
    read_recording,
)
from scanimage_octo_reader.export import build_metadata, export_recording
from scanimage_octo_reader.geometry import ScanGeometry, compute_geometry
from scanimage_octo_reader.header import ScanImageHeader, read_header
from scanimage_octo_reader.page_headers import PageSweep, sweep_pages
from scanimage_octo_reader.parsers import (
    I2CPacket,
    parse_aux_trigger,
    parse_epoch,
    parse_i2c_data,
)
from scanimage_octo_reader.projection import (
    PROJECTION_METHODS,
    ProjectionResult,
    project_recording,
)
from scanimage_octo_reader.qc import QCReport, check_recording
from scanimage_octo_reader.triggers import decode_i2c_key_values, decode_i2c_payload_text


def _resolve_version() -> str:
    """Version from setuptools_scm's generated file, or the installed metadata."""
    try:
        from scanimage_octo_reader._version import version

        return str(version)
    except ImportError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("scanimage-octo-reader")
        except PackageNotFoundError:
            return "0.0.0"
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
        return "0.0.0"


__version__ = _resolve_version()

__all__ = [
    "PROJECTION_METHODS",
    "I2CPacket",
    "PageSweep",
    "ProjectionResult",
    "QCReport",
    "Recording",
    "ScanGeometry",
    "ScanImageHeader",
    "__version__",
    "build_metadata",
    "check_recording",
    "compute_geometry",
    "decode_i2c_key_values",
    "decode_i2c_payload_text",
    "export_recording",
    "find_acquisition_files",
    "parse_aux_trigger",
    "parse_epoch",
    "parse_i2c_data",
    "project_recording",
    "read_header",
    "read_recording",
    "sweep_pages",
]
