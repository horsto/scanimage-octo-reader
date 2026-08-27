"""Collapsing the Z axis of a volumetric recording: one page per volume.

A multi-plane ScanImage acquisition writes every Z-plane as its own TIFF
page, so a 5-plane recording of 5000 timepoints is a flat stack of 25 000+
pages (plus flyback). Many analyses - and most single-plane pipelines - want
the *time* axis back: one page per volume, each page a projection across
that volume's planes.

This module is the only place in the package that reads pixel data in bulk
(`calibration.average_projection` aside, which projects across *all* pages,
time included). Two properties matter for real files, which are several
gigabytes:

* **one read per volume.** Every requested method is computed from the same
  `(n_planes, height, width)` read, so ``mean``, ``max`` and ``std`` together
  cost a single pass over the file.
* **incremental write.** Output pages are appended to a contiguous
  `tifffile` series as they are computed, so peak memory is one volume, not
  one stack. BigTIFF is enabled automatically once the output would exceed
  the 4 GB classic-TIFF limit.

Which pages belong to a volume comes from `geometry.ScanGeometry`, never
from page order alone: channels vary fastest, ``framesPerSlice`` repeats sit
*inside* a Z position, and flyback is a single raw frame at the end of the
whole volume. Flyback pages are never projected - they image no Z position.
Frame repeats *are* included: with ``framesPerSlice > 1`` every page of a
selected plane participates. *How* they participate is `REPEAT_MODES`:

* ``pool`` (the default, and what this module has always done) reduces
  planes and repeats together in one step. Exact and unbiased for ``mean``,
  but ``max`` then selects from that many more samples - a max projection
  grows brighter the more repeats were acquired, which also makes it
  incomparable between recordings with different ``framesPerSlice`` - and
  ``std`` mixes within-plane jitter with between-plane contrast.
* ``average`` first averages each plane's repeats into a single frame, then
  applies the method across planes. ``mean`` is unchanged (every plane
  contributes equally many repeats), ``max`` is taken over denoised
  per-plane frames, and ``std`` becomes a purely between-plane measure.

With ``framesPerSlice == 1`` there is nothing to average and the two modes
coincide; see the README's "Plane projections" section for the numbers.

A partial trailing volume (an acquisition aborted mid-volume) is dropped
rather than projected from an incomplete set of planes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile

from scanimage_octo_reader.acquisition import Recording
from scanimage_octo_reader.geometry import ScanGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "DTYPE_CHOICES",
    "PROJECTION_METHODS",
    "REPEAT_MODES",
    "ProjectionResult",
    "project",
    "project_recording",
    "projection_path",
    "volume_page_offsets",
]

PROJECTION_METHODS = ("mean", "max", "std")

# How `framesPerSlice` repeats enter the projection (see the module
# docstring). `pool` is the default: it is the established behaviour, and for
# `mean` - the default method - the two modes agree exactly anyway.
REPEAT_MODES = ("pool", "average")

# ``auto`` keeps the source dtype for mean/max - an int16 recording stays
# int16, rather than doubling in size - and uses float32 for std, whose
# values are small and not meaningful rounded to the source's integer grid.
DTYPE_CHOICES = ("auto", "int16", "uint16", "float32")

# Classic TIFF cannot address beyond 4 GB; switch to BigTIFF with room to
# spare for the IFDs themselves.
_BIGTIFF_THRESHOLD_BYTES = 3_900_000_000


@dataclass(frozen=True)
class ProjectionResult:
    """One written projection TIFF."""

    path: Path
    method: str
    n_pages: int
    planes: list[int]
    dtype: str
    # How frame repeats were handled, and how many frames the method was
    # consequently reduced over per output page: `max` and `std` depend on
    # both, so both belong in the record of what was written.
    repeats: str = "pool"
    n_reduced_frames: int = 1
    channel: int | None = None
    # Volumes present in the source but not written, because the acquisition
    # ended mid-volume, `limit` cut the output short, or their pages could
    # not be read.
    skipped_volumes: int = 0
    warnings: list[str] = field(default_factory=list)


def project(stack: np.ndarray, method: str) -> np.ndarray:
    """Project `stack` along its first axis with `method`.

    ``max`` is computed in the input dtype (it selects an existing sample,
    so nothing is gained by widening); ``mean`` and ``std`` are computed in
    float32, which represents sums of a handful of 16-bit samples exactly.
    """
    if method == "max":
        return stack.max(axis=0)
    if method == "mean":
        return stack.astype(np.float32, copy=False).mean(axis=0)
    if method == "std":
        return stack.astype(np.float32, copy=False).std(axis=0)
    raise ValueError(
        f"unknown projection method {method!r}; choose from {', '.join(PROJECTION_METHODS)}"
    )


def volume_page_offsets(
    geometry: ScanGeometry, planes: Sequence[int], channel_position: int
) -> np.ndarray:
    """Page offsets, within one volume, of `planes` on one channel.

    Shaped ``(len(planes), framesPerSlice)``: one row per Z position, holding
    that plane's repeated frames. Keeping the repeat axis separate is what
    lets a caller either flatten it (`pool`) or reduce along it first
    (`average`). The page layout of a volume is strictly periodic (see the
    module docstring and `geometry`), so a volume's pages are these offsets
    plus ``volume_index * pages_per_volume``. Flyback pages are never
    included: the offsets are built from Z positions, which flyback has none
    of.
    """
    frames_per_slice = max(geometry.frames_per_slice, 1)
    return np.array(
        [
            [
                (plane * frames_per_slice + repeat) * geometry.n_channels + channel_position
                for repeat in range(frames_per_slice)
            ]
            for plane in planes
        ],
        dtype=np.int64,
    )


def projection_path(directory: Path, name: str, method: str, channel: int | None = None) -> Path:
    """Where one projection is written.

    The recording's name is repeated in the filename, not left to the
    directory alone: projection TIFFs are made to be opened elsewhere (ImageJ,
    suite2p, a napari session) and routinely get moved away from the
    directory that identified them.
    """
    suffix = "" if channel is None else f"_ch{channel}"
    return directory / f"{name}_proj-{method}{suffix}.tif"


class _PageReader:
    """Reads pixel data by *global* page index across a merged acquisition.

    `tifffile` can under-report the page count of an SI-style classic TIFF,
    and the sweep recovers those trailing pages by walking the IFD chain
    (see `page_headers.recover_trailing_pages`). Those pages are real - the
    sweep counts them in `Recording.pages_per_file` - but `tifffile` will not
    index them, so they are kept alongside and looked up by the same global
    index. Skipping them instead would silently drop the last volume of
    every affected file.
    """

    def __init__(self, stack: ExitStack, paths: Sequence[Path], pages_per_file: Sequence[int]):
        from scanimage_octo_reader.page_headers import recover_trailing_pages

        self._files = [stack.enter_context(tifffile.TiffFile(path)) for path in paths]
        self._indexed = [len(tif.pages) for tif in self._files]
        self._recovered = [
            recover_trailing_pages(tif, n_indexed)
            for tif, n_indexed in zip(self._files, self._indexed)
        ]
        counts = list(pages_per_file) or [
            n_indexed + len(recovered)
            for n_indexed, recovered in zip(self._indexed, self._recovered)
        ]
        self._starts = np.cumsum([0, *counts])

    def is_readable(self, index: int) -> bool:
        file_index, local = self._locate(index)
        if not 0 <= file_index < len(self._files):
            return False
        return local < self._indexed[file_index] + len(self._recovered[file_index])

    def read(self, index: int) -> np.ndarray:
        file_index, local = self._locate(index)
        n_indexed = self._indexed[file_index]
        if local < n_indexed:
            return self._files[file_index].pages[local].asarray()
        return self._recovered[file_index][local - n_indexed].asarray()

    def software_tag(self) -> str | None:
        """The first file's verbatim ``Software`` tag (the ``SI.*`` header)."""
        try:
            value = self._files[0].pages[0].tags["Software"].value
        except (IndexError, KeyError, AttributeError):
            return None
        return value if isinstance(value, str) else None

    def _locate(self, index: int) -> tuple[int, int]:
        file_index = int(np.searchsorted(self._starts, index, side="right")) - 1
        return file_index, int(index - self._starts[file_index])


def _resolve_methods(methods: Sequence[str]) -> list[str]:
    if not methods:
        raise ValueError(f"no projection method given; choose from {', '.join(PROJECTION_METHODS)}")
    unknown = [method for method in methods if method not in PROJECTION_METHODS]
    if unknown:
        raise ValueError(
            f"unknown projection method(s): {', '.join(unknown)}; "
            f"choose from {', '.join(PROJECTION_METHODS)}"
        )
    return list(dict.fromkeys(methods))


def _resolve_repeats(repeats: str, geometry: ScanGeometry) -> str:
    """Validate the repeat mode, collapsing it to ``pool`` when it is a no-op."""
    if repeats not in REPEAT_MODES:
        raise ValueError(f"unknown repeat mode {repeats!r}; choose from {', '.join(REPEAT_MODES)}")
    # One frame per Z position: there is nothing to average, and the two
    # modes are the same reduction. Normalising here keeps the read path on
    # one branch (and avoids a pointless float conversion of `max`).
    if geometry.frames_per_slice <= 1:
        return "pool"
    return repeats


def _guard_std_is_defined(methods: Sequence[str], n_reduced_frames: int) -> None:
    """Refuse a standard deviation that would be identically zero.

    With one frame left to reduce over - a single plane, or a single plane
    whose repeats were averaged away first - `std` is not degenerate in an
    interesting way, it is exactly zero everywhere. Writing that stack would
    look like a successful analysis, so it is an error instead.
    """
    if "std" not in methods or n_reduced_frames >= 2:
        return
    raise ValueError(
        "a std projection needs at least 2 frames per output page, but this "
        f"selection leaves {n_reduced_frames}; select more planes, or use "
        "--repeats pool to keep the frame repeats as separate samples"
    )


def _resolve_planes(geometry: ScanGeometry, planes: Sequence[int] | None) -> list[int]:
    if planes is None:
        return list(range(geometry.n_slices))
    resolved = sorted(dict.fromkeys(int(plane) for plane in planes))
    if not resolved:
        raise ValueError("no planes selected")
    out_of_range = [plane for plane in resolved if not 0 <= plane < geometry.n_slices]
    if out_of_range:
        raise ValueError(
            f"plane index/indices {out_of_range} outside 0-{geometry.n_slices - 1} "
            f"({geometry.n_slices} plane(s) in this recording)"
        )
    return resolved


def _resolve_channels(geometry: ScanGeometry, channels: Sequence[int] | None) -> list[int]:
    """Resolve requested ScanImage channel numbers, as saved in the file."""
    saved = list(geometry.channels_saved) or [1]
    if channels is None:
        return saved
    requested = sorted(dict.fromkeys(int(channel) for channel in channels))
    missing = [channel for channel in requested if channel not in saved]
    if missing:
        raise ValueError(f"channel(s) {missing} were not saved in this recording; it has {saved}")
    return requested


def _resolve_dtype(requested: str, method: str, source_dtype: str) -> np.dtype:
    if requested not in DTYPE_CHOICES:
        raise ValueError(
            f"unknown output dtype {requested!r}; choose from {', '.join(DTYPE_CHOICES)}"
        )
    if requested != "auto":
        return np.dtype(requested)
    if method == "std":
        return np.dtype(np.float32)
    try:
        return np.dtype(source_dtype)
    except TypeError:  # pragma: no cover - a Recording always carries a real dtype
        return np.dtype(np.float32)


def _cast(frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a projection to `dtype`, rounding and clipping for integer output."""
    if frame.dtype == dtype:
        return frame
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        return np.clip(np.rint(frame), limits.min, limits.max).astype(dtype)
    return frame.astype(dtype)


def projected_software_tag(
    software: str | None,
    recording: Recording,
    planes: Sequence[int],
    channel: int | None,
    n_pages: int,
) -> str | None:
    """The source's ``SI.*`` header, rewritten to describe the projection.

    Copying the header over verbatim is actively harmful: readers derive the
    page layout *from the header*, gating on ``hStackManager.enable`` /
    ``hFastZ.enable`` and ``actualNumSlices`` (this package's own `geometry`
    does, and so does `napari-tiff`). A projection whose header still claims
    2 planes and a flyback frame is therefore re-folded into volumes on
    opening - the planes reappear, and the page count is silently divided.

    So the fields describing the *layout* are rewritten to what the output
    actually is - a flat, single-plane, single-channel timeseries of one page
    per volume, whose frame rate is the source's volume rate - while
    everything else (zoom, objective resolution, scanner, PMT settings, ...)
    is left exactly as ScanImage wrote it, which is what makes the
    projection still usable with e.g. `socto scalebar`. Only keys already
    present are touched; a header that never claimed to be volumetric is
    left alone. `SI.hStackManager.zs` collapses to the mean Z of the
    projected planes - the projection sits at no single depth, and the
    per-plane values are kept in the JSON `ImageDescription`.
    """
    if not software:
        return None

    geometry = recording.geometry
    overrides = {
        # The layout gates: without these, everything below is ignored and
        # the planes come back.
        "SI.hStackManager.enable": "false",
        "SI.hFastZ.enable": "false",
        "SI.hStackManager.numSlices": "1",
        "SI.hStackManager.actualNumSlices": "1",
        "SI.hStackManager.numFramesPerVolume": "1",
        "SI.hStackManager.numFramesPerVolumeWithFlyback": "1",
        "SI.hStackManager.framesPerSlice": "1",
        "SI.hFastZ.discardFlybackFrames": "false",
        "SI.hFastZ.numDiscardFlybackFrames": "0",
        # Frame averaging happened at acquisition time, upstream of this.
        "SI.hScan2D.logAverageFactor": "1",
        "SI.hStackManager.numVolumes": str(n_pages),
        "SI.hStackManager.actualNumVolumes": str(n_pages),
        # A scalar, so a reader counts one channel per output file even when
        # the source saved several.
        "SI.hChannels.channelSave": str(
            channel if channel is not None else (geometry.channels_saved or [1])[0]
        ),
    }

    # One page is now one volume, so the page ('frame') rate is the volume
    # rate; a reader deriving the time axis from scanFrameRate gets the real
    # interval between output pages.
    if geometry.volume_rate_hz:
        overrides["SI.hRoiManager.scanFrameRate"] = f"{geometry.volume_rate_hz:.10g}"

    zs = [geometry.zs[plane] for plane in planes if plane < len(geometry.zs)]
    if zs:
        overrides["SI.hStackManager.zs"] = f"{sum(zs) / len(zs):.10g}"

    lines = []
    for line in software.splitlines():
        key, separator, _value = line.partition("=")
        replacement = overrides.get(key.strip()) if separator else None
        lines.append(line if replacement is None else f"{key.rstrip()} = {replacement}")
    return "\n".join(lines) + ("\n" if software.endswith("\n") else "")


def _description(
    recording: Recording,
    method: str,
    planes: Sequence[int],
    channel: int | None,
    n_pages: int,
    dtype: np.dtype,
    repeats: str,
    n_reduced_frames: int,
) -> str:
    """The JSON provenance written into the output's first page."""
    from scanimage_octo_reader import __version__
    from scanimage_octo_reader.export import to_jsonable

    geometry = recording.geometry
    zs = geometry.zs
    payload = {
        "tool": {"name": "scanimage-octo-reader", "version": __version__},
        "created": datetime.now(UTC).isoformat(),
        "projection": {
            "method": method,
            "axis": "z",
            "planes": list(planes),
            "plane_zs": [zs[plane] for plane in planes if plane < len(zs)],
            "n_planes": len(planes),
            "frames_per_slice": geometry.frames_per_slice,
            # `max` and `std` depend on how many frames were reduced, not
            # just on which planes - so both are recorded.
            "repeats": repeats,
            "n_reduced_frames": n_reduced_frames,
            "flyback_pages_excluded": geometry.flyback_frames,
            "channel": channel,
            "n_pages": n_pages,
            "page_axis": "volume (time)",
            "source_dtype": recording.dtype,
            "output_dtype": str(dtype),
            "volume_rate_hz": geometry.volume_rate_hz,
            "frame_rate_hz": geometry.frame_rate_hz,
            "scanimage_header": (
                "the source's SI.* header, with the plane/channel layout fields "
                "rewritten to describe this projection"
            ),
        },
        "source": {
            "name": recording.name,
            "files": [str(path.resolve()) for path in recording.paths],
            "n_pages": recording.n_pages,
            "n_slices": geometry.n_slices,
            "channels_saved": geometry.channels_saved,
            "epoch": recording.epoch,
        },
    }
    return json.dumps(to_jsonable(payload))


def _guard_targets(targets: dict[tuple[str, int | None], Path], overwrite: bool) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"{listed} already exist(s); pass overwrite to replace it")


def _n_volumes(recording: Recording, reader: _PageReader, limit: int) -> tuple[int, list[str]]:
    """Volumes that can actually be projected, plus what was dropped and why."""
    geometry = recording.geometry
    warnings: list[str] = []
    complete = geometry.n_volumes(recording.n_pages)
    leftover = recording.n_pages - complete * geometry.pages_per_volume
    if leftover:
        warnings.append(
            f"the last {leftover} page(s) do not make up a whole volume "
            f"({geometry.pages_per_volume} pages each); skipping that partial volume"
        )

    # Page headers can be swept for pages tifffile will not hand out pixel
    # data for; stop at the first such page rather than failing mid-write.
    n_volumes = complete
    for volume in range(complete):
        start = volume * geometry.pages_per_volume
        if not reader.is_readable(start + geometry.pages_per_volume - 1):
            n_volumes = volume
            warnings.append(
                f"pixel data is only readable for the first {n_volumes} volume(s) of "
                f"{complete}; the rest of the file appears truncated"
            )
            break

    if limit and limit < n_volumes:
        n_volumes = limit
    return n_volumes, warnings


def project_recording(
    recording: Recording,
    out_dir: str | Path | None = None,
    methods: Sequence[str] = ("mean",),
    planes: Sequence[int] | None = None,
    channels: Sequence[int] | None = None,
    dtype: str = "auto",
    repeats: str = "pool",
    limit: int = 0,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> list[ProjectionResult]:
    """Write one projection TIFF per method (and per channel) for `recording`.

    Each output has one page per volume - the recording's *time* axis - and
    each page is `method` applied across the selected `planes`. `out_dir`
    defaults to the directory holding the source TIFF, and outputs land in
    ``<out_dir>/<recording name>/``, matching the rest of the package.

    `planes` are zero-based Z indices (default: all), `channels` are
    ScanImage channel numbers as saved in the file (default: all, one file
    each when more than one was saved), and `limit` caps the number of
    volumes written. `repeats` selects how ``framesPerSlice`` repeats are
    handled - see `REPEAT_MODES` and the module docstring; it matters for
    `max` and `std`, not for `mean`. `progress`, when given, is called with
    the number of volumes completed so far.
    """
    from scanimage_octo_reader.export import _ensure_dir, default_output_root

    geometry = recording.geometry
    if geometry.n_slices < 2:
        raise ValueError(
            f"{recording.name} is not a multi-plane recording "
            f"({geometry.n_slices} plane(s)); there is nothing to project across"
        )

    resolved_methods = _resolve_methods(methods)
    resolved_planes = _resolve_planes(geometry, planes)
    resolved_channels = _resolve_channels(geometry, channels)
    resolved_repeats = _resolve_repeats(repeats, geometry)
    multi_channel = len(geometry.channels_saved) > 1

    # How many frames each output page is reduced over: one per plane once
    # the repeats have been averaged away, otherwise every repeat of every
    # selected plane.
    n_reduced_frames = len(resolved_planes) * (
        1 if resolved_repeats == "average" else max(geometry.frames_per_slice, 1)
    )
    _guard_std_is_defined(resolved_methods, n_reduced_frames)

    root = default_output_root(recording) if out_dir is None else Path(out_dir)
    directory = root / recording.name

    results: list[ProjectionResult] = []
    with ExitStack() as stack:
        reader = _PageReader(stack, recording.paths, recording.pages_per_file)
        n_volumes, warnings = _n_volumes(recording, reader, limit)
        if n_volumes <= 0:
            raise ValueError(
                f"{recording.name}: no complete volume could be read "
                f"({recording.n_pages} page(s), {geometry.pages_per_volume} per volume)"
            )

        targets = {
            (method, channel if multi_channel else None): projection_path(
                directory, recording.name, method, channel if multi_channel else None
            )
            for method in resolved_methods
            for channel in resolved_channels
        }
        _guard_targets(targets, overwrite)
        _ensure_dir(directory)

        height, width = (int(size) for size in recording.image_shape[:2])
        dtypes = {
            method: _resolve_dtype(dtype, method, recording.dtype) for method in resolved_methods
        }
        # Rewritten per output, since the channel it claims differs between
        # the per-channel files - see `projected_software_tag`.
        source_software = reader.software_tag()
        software = {
            output_channel: projected_software_tag(
                source_software, recording, resolved_planes, output_channel, n_volumes
            )
            for output_channel in (
                [channel if multi_channel else None for channel in resolved_channels]
            )
        }

        writers: dict[tuple[str, int | None], tifffile.TiffWriter] = {}
        started: set[tuple[str, int | None]] = set()
        for key, path in targets.items():
            method, _channel = key
            n_bytes = n_volumes * height * width * dtypes[method].itemsize
            writers[key] = stack.enter_context(
                tifffile.TiffWriter(path, bigtiff=n_bytes > _BIGTIFF_THRESHOLD_BYTES)
            )

        offsets = {
            channel: volume_page_offsets(
                geometry, resolved_planes, geometry.channels_saved.index(channel)
            )
            for channel in resolved_channels
        }

        for volume in range(n_volumes):
            base = volume * geometry.pages_per_volume
            for channel in resolved_channels:
                output_channel = channel if multi_channel else None
                # One read per volume and channel; every method is computed
                # from it, so extra methods cost arithmetic, not I/O.
                per_plane = offsets[channel]
                planes_stack = np.stack(
                    [reader.read(base + int(offset)) for offset in per_plane.ravel()]
                ).reshape(per_plane.shape + (height, width))
                if resolved_repeats == "average":
                    # Collapse each plane's repeats first, so the method sees
                    # one denoised frame per Z position.
                    planes_stack = planes_stack.astype(np.float32).mean(axis=1)
                else:
                    planes_stack = planes_stack.reshape((-1, height, width))
                for method in resolved_methods:
                    key = (method, output_channel)
                    frame = _cast(project(planes_stack, method), dtypes[method])
                    if key in started:
                        writers[key].write(frame, contiguous=True)
                        continue
                    writers[key].write(
                        frame,
                        contiguous=True,
                        photometric="minisblack",
                        # Our own provenance JSON, not tifffile's "shaped"
                        # metadata; with `contiguous` it is written once.
                        metadata=None,
                        description=_description(
                            recording,
                            method,
                            resolved_planes,
                            output_channel,
                            n_volumes,
                            dtypes[method],
                            resolved_repeats,
                            n_reduced_frames,
                        ),
                        # The source's ScanImage header, with its plane and
                        # channel layout rewritten to match this projection.
                        software=software[output_channel],
                    )
                    started.add(key)
            if progress is not None:
                progress(volume + 1)

        skipped = max(geometry.n_volumes(recording.n_pages) - n_volumes, 0)
        for (method, output_channel), path in targets.items():
            results.append(
                ProjectionResult(
                    path=path,
                    method=method,
                    n_pages=n_volumes,
                    planes=list(resolved_planes),
                    dtype=str(dtypes[method]),
                    repeats=resolved_repeats,
                    n_reduced_frames=n_reduced_frames,
                    channel=output_channel,
                    skipped_volumes=skipped,
                    warnings=list(warnings),
                )
            )

    logger.info(
        "projected %d volume(s) of %s across plane(s) %s into %d file(s)",
        n_volumes,
        recording.name,
        resolved_planes,
        len(results),
    )
    return results
