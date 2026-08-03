"""Working out what each TIFF page *is* in a ScanImage acquisition.

A ScanImage TIFF is a flat stack of pages; which volume, Z-slice and channel
a page belongs to has to be reconstructed from the global header. The rules
(verified against real single-plane, volumetric, and multi-channel
acquisitions in `napari-tiff`, whose ``compute_scanimage_dimensions`` this
module adapts) are:

* channels are always the **fastest-varying** axis: one raw frame-scan
  occupies ``n_channels`` adjacent pages;
* ``framesPerSlice`` repeated frame-scans are captured at each Z-position
  before the scanner moves on, i.e. on disk the order is Z-outer,
  frame-repeat-inner;
* flyback overhead
  (``numFramesPerVolumeWithFlyback - numFramesPerVolume``) is a single raw
  frame appended **once** at the end of the whole volume, not per
  Z-position;
* volumetric interpretation is gated on ``hStackManager.enable`` /
  ``hFastZ.enable``, never on the mere presence of slice-count fields:
  those can be stale leftovers from an earlier configuration. The sample
  data proves the point - ``numSlices = 11`` while ``actualNumSlices = 3``.

Anything inconsistent falls back to a flat, page-is-page interpretation
with an explanation attached, which is always safe to export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scanimage_octo_reader.header import ScanImageHeader

__all__ = ["PageMap", "ScanGeometry", "compute_geometry"]


@dataclass(frozen=True)
class PageMap:
    """Per-page assignment of volume, Z-slice, frame-repeat and channel.

    All arrays are ``n_pages`` long and index-aligned with the TIFF's pages.
    """

    volume_index: np.ndarray
    slice_index: np.ndarray
    frame_repeat_index: np.ndarray
    channel: np.ndarray
    is_flyback: np.ndarray


@dataclass(frozen=True)
class ScanGeometry:
    """How a flat ScanImage page stack decomposes into volumes/slices/channels.

    `on_disk_raw_frames` is the number of raw frame-scans stored per volume
    *including* flyback; `kept_raw_frames` excludes it. `pages_per_volume`
    is the on-disk stride of one volume.
    """

    n_channels: int
    channels_saved: list[int]
    volumetric: bool
    n_slices: int
    frames_per_slice: int
    on_disk_raw_frames: int
    kept_raw_frames: int
    zs: list[float]
    frame_rate_hz: float | None
    volume_rate_hz: float | None
    warning: str | None = None

    @property
    def pages_per_volume(self) -> int:
        return self.on_disk_raw_frames * self.n_channels

    @property
    def flyback_frames(self) -> int:
        return self.on_disk_raw_frames - self.kept_raw_frames

    def n_volumes(self, n_pages: int) -> int:
        """Number of complete volumes contained in `n_pages` pages."""
        if self.pages_per_volume <= 0:
            return 0
        return n_pages // self.pages_per_volume

    def page_map(self, n_pages: int) -> PageMap:
        """Map every page index to its volume, slice, frame-repeat and channel."""
        pages = np.arange(n_pages, dtype=np.int64)
        pages_per_volume = max(self.pages_per_volume, 1)

        volume_index = pages // pages_per_volume
        within_volume = pages % pages_per_volume
        channel = (within_volume % self.n_channels).astype(np.int16)
        raw_frame = within_volume // self.n_channels

        is_flyback = raw_frame >= self.kept_raw_frames
        # Z-outer, frame-repeat-inner within the kept raw frames. Flyback
        # pages get -1 for both, since they belong to no Z-position.
        slice_index = np.where(is_flyback, -1, raw_frame // max(self.frames_per_slice, 1)).astype(
            np.int16
        )
        frame_repeat_index = np.where(
            is_flyback, -1, raw_frame % max(self.frames_per_slice, 1)
        ).astype(np.int16)

        return PageMap(
            volume_index=volume_index,
            slice_index=slice_index,
            frame_repeat_index=frame_repeat_index,
            channel=channel,
            is_flyback=is_flyback,
        )


def _as_int(value: Any) -> int | None:
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)
    except (TypeError, ValueError, IndexError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_as_float_list(item))
            else:
                as_float = _as_float(item)
                if as_float is not None:
                    out.append(as_float)
        return out
    return []


def _channels_saved(header: ScanImageHeader) -> list[int]:
    """Channels written to disk, as a list even when ScanImage stores a scalar."""
    value = header.get("SI.hChannels.channelSave")
    if value is None:
        return [1]
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        channels = [_as_int(v) for v in value]
        return [c for c in channels if c is not None] or [1]
    return [1]


def _flat(reason: str | None, header: ScanImageHeader, channels: list[int]) -> ScanGeometry:
    return ScanGeometry(
        n_channels=max(len(channels), 1),
        channels_saved=channels,
        volumetric=False,
        n_slices=1,
        frames_per_slice=1,
        on_disk_raw_frames=1,
        kept_raw_frames=1,
        zs=_as_float_list(header.get("SI.hStackManager.zs")),
        frame_rate_hz=_as_float(header.get("SI.hRoiManager.scanFrameRate")),
        volume_rate_hz=_as_float(header.get("SI.hRoiManager.scanVolumeRate")),
        warning=reason,
    )


def compute_geometry(header: ScanImageHeader, n_pages: int) -> ScanGeometry:
    """Derive the page layout of an acquisition from its global header.

    Cross-checks ``actualNumSlices x framesPerSlice == numFramesPerVolume``
    and that the resulting stride divides the total page count, before
    committing to a volumetric interpretation; on any inconsistency it
    returns a flat interpretation carrying a `ScanGeometry.warning`
    explaining why, so callers can surface the problem without failing.
    """
    channels = _channels_saved(header)
    n_channels = max(len(channels), 1)

    volumetric = bool(header.get("SI.hStackManager.enable")) or bool(header.get("SI.hFastZ.enable"))
    if not volumetric:
        # Not an error: a plain single-plane timeseries genuinely is flat.
        geometry = _flat(None, header, channels)
        return _validated(geometry, n_pages, header, channels)

    n_slices = _as_int(header.get("SI.hStackManager.actualNumSlices"))
    frames_per_volume = _as_int(header.get("SI.hStackManager.numFramesPerVolume"))
    frames_per_volume_flyback = _as_int(
        header.get("SI.hStackManager.numFramesPerVolumeWithFlyback")
    )

    if not n_slices or not frames_per_volume:
        return _flat(
            "hStackManager/hFastZ report a volumetric acquisition, but the slice-count "
            "fields are missing; treating every page as an independent frame",
            header,
            channels,
        )

    on_disk_raw_frames = frames_per_volume_flyback or frames_per_volume
    if not 0 < frames_per_volume <= on_disk_raw_frames:
        return _flat(
            f"inconsistent frames-per-volume metadata (numFramesPerVolume="
            f"{frames_per_volume}, withFlyback={frames_per_volume_flyback})",
            header,
            channels,
        )

    frames_per_slice = _as_int(header.get("SI.hStackManager.framesPerSlice")) or 1
    frames_per_slice = max(frames_per_slice, 1)

    if n_slices * frames_per_slice != frames_per_volume:
        return _flat(
            f"actualNumSlices ({n_slices}) x framesPerSlice ({frames_per_slice}) does not "
            f"match numFramesPerVolume ({frames_per_volume})",
            header,
            channels,
        )

    geometry = ScanGeometry(
        n_channels=n_channels,
        channels_saved=channels,
        volumetric=True,
        n_slices=n_slices,
        frames_per_slice=frames_per_slice,
        on_disk_raw_frames=on_disk_raw_frames,
        kept_raw_frames=frames_per_volume,
        zs=_as_float_list(header.get("SI.hStackManager.zs")),
        frame_rate_hz=_as_float(header.get("SI.hRoiManager.scanFrameRate")),
        volume_rate_hz=_as_float(header.get("SI.hRoiManager.scanVolumeRate")),
    )
    return _validated(geometry, n_pages, header, channels)


def _validated(
    geometry: ScanGeometry, n_pages: int, header: ScanImageHeader, channels: list[int]
) -> ScanGeometry:
    """Fall back to a flat layout if the derived stride cannot explain `n_pages`.

    A partial trailing volume is normal (an acquisition can be aborted
    mid-volume), so only a stride *larger* than the whole file, or a
    non-positive stride, invalidates the interpretation.
    """
    stride = geometry.pages_per_volume
    if stride <= 0:
        return _flat("derived a non-positive number of pages per volume", header, channels)
    if n_pages and stride > n_pages:
        return _flat(
            f"the header implies {stride} pages per volume, but the file only has "
            f"{n_pages} page(s)",
            header,
            channels,
        )
    return geometry
