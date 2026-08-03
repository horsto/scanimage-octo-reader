"""Synthetic ScanImage TIFFs for the test suite.

Real ScanImage files are 4-10 GB, so the tests build tiny ones with the same
tag structure instead: the ``SI.*`` text in the ``Software`` tag, and a
ScanImage-style frame header in each page's ``ImageDescription``. That keeps
the suite fast and self-contained, and lets us construct the awkward cases
(comma-separated AUX arrays, both I2C flavours, single-packet I2C, stale
slice counts, ``inf`` in the header) that real files only occasionally show.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

# A minimal but realistic global header. Values mirror the sample data:
# `numSlices` is deliberately stale (11) while `actualNumSlices` is 3, which
# is exactly the trap `geometry` has to avoid.
DEFAULT_HEADER: dict[str, object] = {
    "SI.VERSION_MAJOR": 2022,
    "SI.VERSION_MINOR": 1,
    "SI.VERSION_UPDATE": 0,
    "SI.hChannels.channelSave": 1,
    "SI.hStackManager.enable": False,
    "SI.hFastZ.enable": False,
    "SI.hStackManager.numSlices": 11,
    "SI.hStackManager.actualNumSlices": 1,
    "SI.hStackManager.framesPerSlice": 1,
    "SI.hStackManager.numFramesPerVolume": 1,
    "SI.hStackManager.numFramesPerVolumeWithFlyback": 1,
    "SI.hStackManager.zs": 0,
    "SI.hRoiManager.scanFrameRate": 30.0,
    "SI.hRoiManager.scanVolumeRate": 30.0,
    "SI.hRoiManager.scanZoomFactor": 3,
    "SI.hRoiManager.pixelsPerLine": 4,
    "SI.hRoiManager.linesPerFrame": 4,
    "SI.hRoiManager.mroiEnable": 0,
    "SI.hScan2D.scannerType": "RG",
    "SI.hScan2D.name": "Test_Scanner",
    "SI.hScan2D.bidirectional": True,
    "SI.hBeams.lengthConstants": float("inf"),
    "SI.objectiveResolution": 15,
}

FRAME_PERIOD_S = 1.0 / 30.0

# Real ScanImage pads every page's ImageDescription to a fixed size (2001
# bytes in the 2022.1 sample), which keeps the pages uniformly strided. The
# fixtures do the same, both for faithfulness and because variable-length
# descriptions make tifffile mis-locate IFDs in small files.
DESCRIPTION_SIZE = 512


def format_si_value(value: object) -> str:
    """Render a Python value the way ScanImage writes it into the header."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == float("inf"):
            return "Inf"
        if value != value:  # NaN
            return "NaN"
        return repr(value)
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(format_si_value(item).strip("'") for item in value) + "]"
    return str(value)


def build_header(**overrides: object) -> str:
    """Build the ``Software`` tag text, with `overrides` applied to the defaults."""
    header = {**DEFAULT_HEADER, **overrides}
    return "\n".join(f"{key} = {format_si_value(value)}" for key, value in header.items())


def build_page_description(
    frame_number: int,
    timestamp_s: float,
    aux: dict[int, str] | None = None,
    i2c: str = "{}",
    acquisition_number: int = 1,
    end_of_acquisition: int = 0,
    extra: dict[str, str] | None = None,
) -> str:
    """Build one page's ScanImage frame header.

    `aux` maps line number to the *raw* value text, so a test can supply
    ``'[0.1 0.2]'`` or ``'[0.1, 0.2]'`` and exercise either separator
    convention.
    """
    aux = aux or {}
    lines = [
        f"frameNumbers = {frame_number}",
        f"acquisitionNumbers = {acquisition_number}",
        f"frameNumberAcquisition = {frame_number}",
        f"frameTimestamps_sec = {timestamp_s:.9f}",
        "acqTriggerTimestamps_sec = -1.000000000",
        "nextFileMarkerTimestamps_sec = -1.000000000",
        f"endOfAcquisition = {end_of_acquisition}",
        "endOfAcquisitionMode = 0",
        "dcOverVoltage = 0",
        "epoch = [2026  8  3 12 35 19.847]",
    ]
    lines += [f"auxTrigger{line} = {aux.get(line, '[]')}" for line in range(4)]
    lines.append(f"I2CData = {i2c}")
    for key, value in (extra or {}).items():
        lines.append(f"{key} = {value}")
    description = "\n".join(lines)
    if len(description) > DESCRIPTION_SIZE:
        raise ValueError(
            f"page description is {len(description)} bytes, over the fixed "
            f"{DESCRIPTION_SIZE}-byte field; raise DESCRIPTION_SIZE"
        )
    # Pad with NULs, as ScanImage does, so every page's tag is the same size.
    return description.ljust(DESCRIPTION_SIZE, "\0")


def write_tif(
    path,
    descriptions: list[str],
    header: str | None = None,
    shape: tuple[int, int] = (4, 4),
    bigtiff: bool = False,
) -> None:
    """Write a multi-page TIFF that looks like ScanImage output to `path`."""
    header = build_header() if header is None else header
    frame = np.zeros(shape, dtype=np.int16)
    with tifffile.TiffWriter(path, bigtiff=bigtiff) as writer:
        for description in descriptions:
            writer.write(
                frame,
                description=description,
                software=header,
                metadata=None,
                contiguous=False,
            )


def descriptions_for(
    n_pages: int,
    aux_events: dict[int, list[int]] | None = None,
    aux_text: str | None = None,
    i2c_events: dict[int, str] | None = None,
    first_frame_number: int = 1,
    first_timestamp_s: float = 0.0,
    mark_end: bool = True,
) -> list[str]:
    """Build `n_pages` page headers with a regular frame clock.

    `aux_events` maps an AUX line to the page indices that should carry a
    trigger; the timestamp is placed just after that page's frame timestamp,
    as ScanImage would. `aux_text` overrides the rendered array text, so
    separator conventions can be varied.
    """
    aux_events = aux_events or {}
    i2c_events = i2c_events or {}
    descriptions = []
    for page in range(n_pages):
        timestamp = first_timestamp_s + page * FRAME_PERIOD_S
        aux: dict[int, str] = {}
        for line, pages in aux_events.items():
            if page in pages:
                event_time = timestamp + FRAME_PERIOD_S / 3
                aux[line] = aux_text if aux_text else f"[{event_time:.9f} ]"
        descriptions.append(
            build_page_description(
                frame_number=first_frame_number + page,
                timestamp_s=timestamp,
                aux=aux,
                i2c=i2c_events.get(page, "{}"),
                end_of_acquisition=1 if (mark_end and page == n_pages - 1) else 0,
            )
        )
    return descriptions


@pytest.fixture
def single_plane_tif(tmp_path):
    """20 pages, single plane, single channel, two AUX 0 events."""
    path = tmp_path / "plane__00001.tif"
    write_tif(path, descriptions_for(20, aux_events={0: [3, 11]}))
    return path


@pytest.fixture
def volumetric_tif(tmp_path):
    """24 pages = 6 volumes of 3 slices + 1 flyback frame each."""
    path = tmp_path / "volume__00001.tif"
    header = build_header(
        **{
            "SI.hStackManager.enable": True,
            "SI.hFastZ.enable": True,
            "SI.hStackManager.actualNumSlices": 3,
            "SI.hStackManager.numFramesPerVolume": 3,
            "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
            "SI.hStackManager.zs": [10.0, 20.0, 30.0],
            "SI.hRoiManager.scanVolumeRate": 7.5,
        }
    )
    write_tif(path, descriptions_for(24, aux_events={0: [5]}), header=header)
    return path


@pytest.fixture
def two_channel_tif(tmp_path):
    """16 pages, two channels interleaved (channels vary fastest)."""
    path = tmp_path / "channels__00001.tif"
    header = build_header(**{"SI.hChannels.channelSave": [1, 2]})
    descriptions = []
    for page in range(16):
        raw_frame = page // 2  # both channels share a frame number/timestamp
        descriptions.append(
            build_page_description(
                frame_number=raw_frame + 1,
                timestamp_s=raw_frame * FRAME_PERIOD_S,
            )
        )
    write_tif(path, descriptions, header=header)
    return path


@pytest.fixture
def split_acquisition_tifs(tmp_path):
    """Two files of one acquisition, with continuous frame numbers/timestamps."""
    first = tmp_path / "split__00012_00001.tif"
    second = tmp_path / "split__00012_00002.tif"
    write_tif(first, descriptions_for(10, aux_events={0: [2]}, mark_end=False))
    write_tif(
        second,
        descriptions_for(
            10,
            aux_events={0: [4]},
            first_frame_number=11,
            first_timestamp_s=10 * FRAME_PERIOD_S,
        ),
    )
    return first, second
