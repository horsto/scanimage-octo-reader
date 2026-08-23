"""Deriving pixel-to-micron calibration from grid-target recordings.

A "grid target" here is a resolution/distortion test target imaged once per
zoom level: a regularly-spaced lattice of dark lines on a bright background,
photographed so that its pitch runs along either the horizontal (x) or
vertical (y) scan axis - see `parse_grid_filename` for the filename
convention this expects. Some lines are drawn longer than others as decade
markers, but empirically (verified against real recordings) *every* line -
short or long - sits on the same uniform lattice, so peak positions are never
classified by line length: the median spacing between *all* detected line
centres already is the pitch corresponding to the target's known minimum
resolution.

The zoom level a row is filed under is read from the file's own ScanImage
header (`SI.hRoiManager.scanZoomFactor`, the same field `Recording.summary()`
reports) whenever that is possible, since the filename is just an operator's
label and could be wrong. The filename's `parse_grid_filename` zoom is only a
fallback for a header that cannot be read, and disagreement between the two
is surfaced as a warning either way - see `resolve_zoom`.

The measurement (`measure_pitch`) is deliberately not the last word: a badly
extracted line train can still look plausible in isolation, so it rejects
outlier peak-spacings internally and reports whether the remaining peak count
and spread are trustworthy. `build_calibration_table` surfaces an unreliable
measurement as a printable warning message rather than silently including or
dropping it - the CSV itself only ever holds the raw, unopinionated numbers.

A pixel's physical size depends on the acquisition resolution it was sampled
at: the field of view in \u00b5m is set by the zoom and objective alone, so
doubling `pixelsPerLine`/`linesPerFrame` at the same zoom halves \u00b5m/pixel.
`CalibrationRow.resolution_px_x/y` therefore records the pixel count each row
was actually measured at, and `interpolate_calibration` can rescale its
result to a different target resolution on request.

This module also provides the pixel-data helpers `Recording` deliberately
does not (it only exposes per-page *metadata*, not images): `average_projection`
for a memory-friendly mean projection of one or more TIFFs, and
`draw_scale_bar` to burn a scale bar into one.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from re import IGNORECASE
from re import compile as re_compile
from typing import Literal

import numpy as np
import tifffile
from scipy.signal import find_peaks

from scanimage_octo_reader.header import read_header

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationRow",
    "GridFileInfo",
    "PitchMeasurement",
    "average_projection",
    "build_calibration_table",
    "draw_scale_bar",
    "interpolate_calibration",
    "load_calibration_table",
    "measure_pitch",
    "parse_grid_filename",
    "plot_calibration_summary",
    "plot_pitch_diagnostic",
    "resolve_grid_files",
    "resolve_zoom",
    "write_calibration_csv",
]

# The same field `Recording.summary()['zoom']` reports.
_ZOOM_HEADER_KEY = "SI.hRoiManager.scanZoomFactor"

# Filename vs. header zoom disagreement beyond this (in zoom units) is worth
# flagging: the operator's filename label may simply be wrong.
_ZOOM_MISMATCH_TOLERANCE = 0.05

# `<anything>_zoom<int>_<frac>_<horizontal|vertical>_<index>.tif`, e.g.
# `512_zoom1_2_horizontal_00001.tif` -> zoom=1.2, orientation="horizontal".
# A `horizontal` file has its line pitch running along x; `vertical` along y.
_GRID_FILENAME_RE = re_compile(
    r"zoom(?P<zoom_int>\d+)_(?P<zoom_frac>\d+)_(?P<orientation>horizontal|vertical)_(?P<index>\d+)$",
    IGNORECASE,
)

# Standard cutoff for a MAD-based modified z-score outlier test.
_MODIFIED_Z_THRESHOLD = 3.5

_CSV_FIELDS = [
    "zoom",
    "px_to_micron_x",
    "px_to_micron_y",
    "micron_to_px_x",
    "micron_to_px_y",
    "resolution_px_x",
    "resolution_px_y",
    "n_peaks_x",
    "n_peaks_y",
]


@dataclass(frozen=True)
class GridFileInfo:
    """What a grid-target filename encodes."""

    path: Path
    zoom: float
    orientation: Literal["x", "y"]
    index: int


def parse_grid_filename(path: str | Path) -> GridFileInfo | None:
    """Parse a `<...>_zoom<int>_<frac>_<horizontal|vertical>_<index>.tif` filename.

    Returns None (rather than raising) for a name that does not match, so a
    directory listing can filter and warn about what it skips instead of
    failing outright.
    """
    path = Path(path)
    match = _GRID_FILENAME_RE.search(path.stem)
    if not match:
        return None
    zoom = float(f"{match['zoom_int']}.{match['zoom_frac']}")
    axis: Literal["x", "y"] = "x" if match["orientation"].lower() == "horizontal" else "y"
    return GridFileInfo(path=path, zoom=zoom, orientation=axis, index=int(match["index"]))


def resolve_grid_files(inputs: Sequence[str | Path]) -> list[Path]:
    """Expand any directories in `inputs` to their `*.tif` contents.

    Plain files are passed through unchanged, so a mix of directories and
    explicit files both work.
    """
    resolved: list[Path] = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            resolved.extend(sorted(item.glob("*.tif")))
        else:
            resolved.append(item)
    return resolved


def _read_header_zoom(path: Path) -> float | None:
    """Read `SI.hRoiManager.scanZoomFactor` from `path`'s ScanImage header.

    Returns None - never raises - for any file whose header cannot be opened
    or parsed, or that simply has no zoom factor recorded, so callers can
    fall back to another source without special-casing exception types.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            header = read_header(tif)
        value = header.get(_ZOOM_HEADER_KEY)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001 - an unreadable header must not be fatal
        logger.warning("could not read the ScanImage zoom from %s: %s", path, exc)
        return None


def resolve_zoom(info: GridFileInfo) -> tuple[float, str | None]:
    """Resolve the zoom level to file `info` under, preferring the ScanImage header.

    Falls back to the filename-encoded zoom (`info.zoom`) when the header
    cannot be read or carries no zoom factor. Returns `(zoom, warning)`:
    `warning` is a printable message when the header could not be used, or
    when it disagrees with the filename's zoom by more than
    `_ZOOM_MISMATCH_TOLERANCE` - a strong hint that the file is mislabelled -
    and None otherwise.
    """
    header_zoom = _read_header_zoom(info.path)
    if header_zoom is None:
        return info.zoom, (
            f"{info.path.name}: could not read the ScanImage zoom from the header; "
            f"using the filename's zoom ({info.zoom:g})"
        )
    if abs(header_zoom - info.zoom) > _ZOOM_MISMATCH_TOLERANCE:
        return header_zoom, (
            f"{info.path.name}: header zoom ({header_zoom:g}) differs from the filename's "
            f"zoom ({info.zoom:g}); using the header value"
        )
    return header_zoom, None


@dataclass
class PitchMeasurement:
    """The pixel pitch measured from one grid image along one axis.

    `image_shape` is the full frame's `(height, width)`; `resolution_px` is
    the pixel count specifically along the measured `axis` (so `width` for
    `axis="x"`, `height` for `axis="y"`) - the number `um_per_px` is only
    valid at.
    """

    um_per_px: float
    px_pitch: float
    px_pitch_cv: float
    n_peaks: int
    n_peaks_retained: int
    ok: bool
    reason: str | None = None
    image_shape: tuple[int, int] = (0, 0)
    resolution_px: int = 0
    peak_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    profile: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def _reject_outlier_diffs(diffs: np.ndarray) -> np.ndarray:
    """Drop diffs whose modified z-score exceeds the standard 3.5 cutoff.

    A single missed or spurious peak doubles or halves exactly one diff, so
    even one outlier would otherwise skew a plain median.
    """
    median = np.median(diffs)
    mad = np.median(np.abs(diffs - median))
    if mad == 0:
        return diffs
    modified_z = 0.6745 * (diffs - median) / mad
    return diffs[np.abs(modified_z) <= _MODIFIED_Z_THRESHOLD]


def measure_pitch(
    image: np.ndarray,
    axis: Literal["x", "y"],
    pitch_um: float = 10.0,
    center_frac: float = 0.5,
    min_peaks: int = 5,
    max_pitch_cv: float = 0.15,
) -> PitchMeasurement:
    """Measure the pixel pitch of a regularly-spaced grid target along `axis`.

    The profile is built from the column-wise **minimum** (not mean) over the
    central `center_frac` of the perpendicular axis: short "minor" ticks are
    only visible in a narrow hatched band while longer "major" ticks run the
    full frame extent, so a minimum catches every line regardless of its
    length and, unlike a variance-based band search, stays robust under the
    corner vignetting seen at low zoom.

    Every detected line centre is treated the same way (see the module
    docstring), so the median spacing between *all* of them is the pitch
    corresponding to `pitch_um`. `min_peaks` and `max_pitch_cv` gate whether
    the measurement is trusted (`PitchMeasurement.ok`); a failing measurement
    is still returned, not raised, so callers can report *why* one file's
    number should not be used.
    """
    if not 0 < center_frac <= 1:
        raise ValueError(f"center_frac must be in (0, 1], got {center_frac}")

    image_shape = (int(image.shape[0]), int(image.shape[1]))
    resolution_px = image_shape[1] if axis == "x" else image_shape[0]
    data = image if axis == "x" else image.T
    height = data.shape[0]
    lo = int(round(height * (0.5 - center_frac / 2)))
    hi = max(lo + 1, int(round(height * (0.5 + center_frac / 2))))
    profile = data[lo:hi].astype(np.float64).min(axis=0)

    value_range = float(profile.max() - profile.min())
    if value_range <= 0:
        return PitchMeasurement(
            um_per_px=float("nan"),
            px_pitch=float("nan"),
            px_pitch_cv=float("inf"),
            n_peaks=0,
            n_peaks_retained=0,
            ok=False,
            reason="profile has no contrast (flat image)",
            image_shape=image_shape,
            resolution_px=resolution_px,
            profile=profile,
        )

    inverted = profile.max() - profile
    peaks, _ = find_peaks(inverted, distance=3, prominence=value_range * 0.15)

    if peaks.size < 2:
        return PitchMeasurement(
            um_per_px=float("nan"),
            px_pitch=float("nan"),
            px_pitch_cv=float("inf"),
            n_peaks=int(peaks.size),
            n_peaks_retained=0,
            ok=False,
            reason=f"only {peaks.size} line(s) detected; need at least 2 to measure a pitch",
            image_shape=image_shape,
            resolution_px=resolution_px,
            peak_positions=peaks,
            profile=profile,
        )

    diffs = np.diff(peaks).astype(np.float64)
    retained = _reject_outlier_diffs(diffs)
    if retained.size == 0:
        retained = diffs

    px_pitch = float(np.median(retained))
    mad = float(np.median(np.abs(retained - px_pitch)))
    cv = (mad / px_pitch) if px_pitch > 0 else float("inf")
    um_per_px = (pitch_um / px_pitch) if px_pitch > 0 else float("nan")

    n_peaks = int(peaks.size)
    reasons = []
    if n_peaks < min_peaks:
        reasons.append(f"only {n_peaks} line(s) detected (< {min_peaks})")
    if cv > max_pitch_cv:
        reasons.append(f"pitch is too irregular (MAD/median={cv:.1%} > {max_pitch_cv:.0%})")

    return PitchMeasurement(
        um_per_px=um_per_px,
        px_pitch=px_pitch,
        px_pitch_cv=cv,
        n_peaks=n_peaks,
        n_peaks_retained=int(retained.size),
        ok=not reasons,
        reason="; ".join(reasons) or None,
        image_shape=image_shape,
        resolution_px=resolution_px,
        peak_positions=peaks,
        profile=profile,
    )


def average_projection(paths: str | Path | Sequence[str | Path]) -> np.ndarray:
    """Mean-project every page of one or more TIFF files, page by page.

    Accumulates a running sum rather than loading a full stack into memory,
    so this stays cheap for long recordings; for the single-page grid-target
    files it is simply that one page.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    total: np.ndarray | None = None
    count = 0
    for path in paths:
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                frame = page.asarray().astype(np.float64)
                total = frame if total is None else total + frame
                count += 1
    if total is None or count == 0:
        raise ValueError("no pages found to average")
    return total / count


@dataclass
class CalibrationRow:
    """One zoom level's raw calibration: measured, not QC-filtered.

    `resolution_px_x/y` is the pixel count `px_to_micron_x/y` was actually
    measured at (see the module docstring); pass a different target
    resolution to `interpolate_calibration` to rescale for another
    acquisition resolution.
    """

    zoom: float
    px_to_micron_x: float | None = None
    px_to_micron_y: float | None = None
    micron_to_px_x: float | None = None
    micron_to_px_y: float | None = None
    resolution_px_x: int | None = None
    resolution_px_y: int | None = None
    n_peaks_x: int | None = None
    n_peaks_y: int | None = None


def build_calibration_table(
    files: Sequence[str | Path],
    pitch_um: float = 10.0,
    center_frac: float = 0.5,
    min_peaks: int = 5,
    max_pitch_cv: float = 0.15,
) -> tuple[list[CalibrationRow], list[str]]:
    """Build one calibration row per zoom level found in `files`.

    Files that do not match the grid-target naming convention are skipped
    (reported in `messages`, not raised). When more than one file matches the
    same zoom and orientation, their raw measurements are averaged; a
    measurement whose peak count or spread fails `min_peaks`/`max_pitch_cv`
    is still included in that average (nothing is dropped or flagged in the
    output), but is reported in `messages` so it can be surfaced as a
    warning.

    Returns `(rows, messages)`: `rows` sorted by zoom, `messages` a flat list
    of skip/unreliable-measurement warnings suitable for printing.
    """
    messages: list[str] = []
    by_zoom_axis: dict[tuple[float, str], list[PitchMeasurement]] = {}

    for path in files:
        info = parse_grid_filename(path)
        if info is None:
            messages.append(f"skipping {Path(path).name}: does not match the grid filename pattern")
            continue
        zoom, zoom_warning = resolve_zoom(info)
        if zoom_warning:
            messages.append(zoom_warning)
        image = average_projection(info.path)
        measurement = measure_pitch(
            image,
            axis=info.orientation,
            pitch_um=pitch_um,
            center_frac=center_frac,
            min_peaks=min_peaks,
            max_pitch_cv=max_pitch_cv,
        )
        by_zoom_axis.setdefault((zoom, info.orientation), []).append(measurement)
        if not measurement.ok:
            messages.append(
                f"{info.path.name}: unreliable {info.orientation}-pitch measurement - "
                f"{measurement.reason}"
            )

    rows_by_zoom: dict[float, CalibrationRow] = {}

    for (zoom, axis), measurements in by_zoom_axis.items():
        row = rows_by_zoom.setdefault(zoom, CalibrationRow(zoom=zoom))
        um_values = [m.um_per_px for m in measurements if np.isfinite(m.um_per_px)]
        n_peaks_values = [m.n_peaks for m in measurements]
        resolutions = {m.resolution_px for m in measurements if m.resolution_px}

        um_per_px = float(np.mean(um_values)) if um_values else None
        px_per_um = (1.0 / um_per_px) if um_per_px else None
        n_peaks = int(round(np.mean(n_peaks_values))) if n_peaks_values else None
        resolution_px = int(round(np.mean(list(resolutions)))) if resolutions else None
        if len(resolutions) > 1:
            messages.append(
                f"zoom {zoom:g} {axis}-axis: source files disagree on resolution "
                f"({sorted(resolutions)} px); px_to_micron_{axis} may not be comparable "
                "across files"
            )

        if axis == "x":
            row.px_to_micron_x, row.micron_to_px_x, row.n_peaks_x = um_per_px, px_per_um, n_peaks
            row.resolution_px_x = resolution_px
        else:
            row.px_to_micron_y, row.micron_to_px_y, row.n_peaks_y = um_per_px, px_per_um, n_peaks
            row.resolution_px_y = resolution_px

    rows = sorted(rows_by_zoom.values(), key=lambda row: row.zoom)
    return rows, messages


def write_calibration_csv(rows: Sequence[CalibrationRow], out_path: str | Path) -> Path:
    """Write `rows` as CSV: raw `zoom, px_to_micron_x/y, micron_to_px_x/y, resolution_px_x/y,
    n_peaks_x/y`.
    """
    out_path = Path(out_path)
    if out_path.is_dir():
        # A plain `open(..., "w")` on an existing directory raises IsADirectoryError
        # locally, but some network mounts (e.g. smbfs) surface it as a much more
        # confusing `OSError: [Errno 22] Invalid argument` instead.
        raise IsADirectoryError(f"{out_path} is an existing directory; expected a CSV file path")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "zoom": row.zoom,
                    "px_to_micron_x": row.px_to_micron_x,
                    "px_to_micron_y": row.px_to_micron_y,
                    "micron_to_px_x": row.micron_to_px_x,
                    "micron_to_px_y": row.micron_to_px_y,
                    "resolution_px_x": row.resolution_px_x,
                    "resolution_px_y": row.resolution_px_y,
                    "n_peaks_x": row.n_peaks_x,
                    "n_peaks_y": row.n_peaks_y,
                }
            )
    return out_path


def _parse_optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _parse_optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def load_calibration_table(csv_path: str | Path) -> list[CalibrationRow]:
    """Read a calibration CSV written by `write_calibration_csv`, sorted by zoom."""
    rows: list[CalibrationRow] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                CalibrationRow(
                    zoom=float(record["zoom"]),
                    px_to_micron_x=_parse_optional_float(record.get("px_to_micron_x")),
                    px_to_micron_y=_parse_optional_float(record.get("px_to_micron_y")),
                    micron_to_px_x=_parse_optional_float(record.get("micron_to_px_x")),
                    micron_to_px_y=_parse_optional_float(record.get("micron_to_px_y")),
                    resolution_px_x=_parse_optional_int(record.get("resolution_px_x")),
                    resolution_px_y=_parse_optional_int(record.get("resolution_px_y")),
                    n_peaks_x=_parse_optional_int(record.get("n_peaks_x")),
                    n_peaks_y=_parse_optional_int(record.get("n_peaks_y")),
                )
            )
    return sorted(rows, key=lambda row: row.zoom)


def _interpolate_axis(rows: Sequence[CalibrationRow], zoom: float, value_attr: str) -> float | None:
    points = sorted(
        (row.zoom, getattr(row, value_attr)) for row in rows if getattr(row, value_attr) is not None
    )
    if not points:
        return None
    zooms = np.array([point[0] for point in points])
    values = np.array([point[1] for point in points])
    if zooms.size == 1:
        return float(values[0])
    return float(np.interp(zoom, zooms, values))


def _zoom_range(rows: Sequence[CalibrationRow], value_attr: str) -> tuple[float, float] | None:
    zooms = sorted(row.zoom for row in rows if getattr(row, value_attr) is not None)
    return (zooms[0], zooms[-1]) if zooms else None


def interpolate_calibration(
    rows: Sequence[CalibrationRow],
    zoom: float,
    target_resolution_x: int | None = None,
    target_resolution_y: int | None = None,
) -> tuple[float | None, float | None, list[str]]:
    """Linearly interpolate the raw `px_to_micron_x/y` at `zoom` from a calibration table.

    x and y are interpolated independently over their own (typically
    differently-sparse) set of points.

    `zoom` outside the calibrated range is a hard failure, not a clamped
    approximation: `numpy.interp` (used internally for in-range zooms) would
    otherwise silently clamp to the value at the nearest end of the range,
    but that number was not measured at `zoom` and \u00b5m/pixel is not
    constant beyond the calibrated range, so treating it as if it were
    correct would be scientifically wrong. An out-of-range (or otherwise
    unavailable) axis therefore returns None for that axis's `px_to_micron`,
    with an explanatory message in the returned `errors` - the caller must
    not use a `None` result.

    When `target_resolution_x`/`target_resolution_y` are given, the result is
    additionally rescaled from the resolution the calibration was measured at
    (`CalibrationRow.resolution_px_x/y`, itself interpolated at `zoom`) to the
    requested one: \u00b5m/pixel for a fixed field of view scales inversely
    with pixel count (see the module docstring), so
    ``px_to_micron * (measured_resolution / target_resolution)`` gives the
    correct value for an image sampled at a different resolution than the
    calibration images. Silently skipped for an axis whose calibration rows
    carry no resolution (e.g. an older CSV written before this column
    existed).

    Returns `(px_to_micron_x, px_to_micron_y, errors)`.
    """
    errors: list[str] = []

    def _axis_value(
        value_attr: str, resolution_attr: str, target_resolution: int | None, label: str
    ) -> float | None:
        bounds = _zoom_range(rows, value_attr)
        if bounds is None:
            errors.append(f"no {value_attr} calibration points are available at all")
            return None
        lo, hi = bounds
        if not lo <= zoom <= hi:
            errors.append(
                f"zoom {zoom:g} is outside the calibrated {label}-axis range ({lo:g}-{hi:g}); "
                f"refusing to extrapolate {value_attr} rather than silently clamp to a wrong "
                "value - add a calibration point at or near this zoom"
            )
            return None

        value = _interpolate_axis(rows, zoom, value_attr)
        if value is not None and target_resolution:
            measured_resolution = _interpolate_axis(rows, zoom, resolution_attr)
            if measured_resolution:
                value *= measured_resolution / target_resolution
        return value

    px_to_micron_x = _axis_value("px_to_micron_x", "resolution_px_x", target_resolution_x, "x")
    px_to_micron_y = _axis_value("px_to_micron_y", "resolution_px_y", target_resolution_y, "y")

    return px_to_micron_x, px_to_micron_y, errors


def _load_scale_bar_font(size: int):
    """Load a scalable font that can actually render \u00b5 (micro sign).

    Pillow's own default font - even the larger scalable variant available
    since Pillow 10.1 (`ImageFont.load_default(size=...)`) - has no \u00b5
    glyph at all: it silently renders as a "tofu" box, and at its original
    tiny bitmap size the whole label is barely legible besides. Matplotlib
    (already a dependency here) ships DejaVu Sans, which covers it, so that
    is preferred; a system font is tried next, and only as a last resort -
    if truly no scalable font can be found - do we fall back to Pillow's
    bitmap default, with \u00b5 replaced by "u" so the label stays readable
    rather than showing a missing-glyph box.
    """
    from PIL import ImageFont

    try:
        import matplotlib

        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size), True
    except Exception:  # noqa: BLE001 - any font-loading failure just tries the next option
        pass

    for system_font in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # common Linux path
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ):
        try:
            return ImageFont.truetype(system_font, size=size), True
        except Exception:  # noqa: BLE001 - try the next candidate
            continue

    return ImageFont.load_default(), False


def draw_scale_bar(
    image: np.ndarray,
    um_per_px_x: float,
    bar_length_um: float = 50.0,
    *,
    margin_px: int = 20,
    bar_thickness_px: int = 6,
    font_size: int | None = None,
    color: tuple[int, int, int] = (255, 255, 255),
    position: Literal["bottom-right", "bottom-left", "top-right", "top-left"] = "bottom-right",
) -> np.ndarray:
    """Contrast-stretch `image` to 8-bit RGB and burn in a scale bar with a label.

    `font_size` defaults to a size scaled with the image's shorter side, so
    the label stays legible regardless of the recording's resolution rather
    than using Pillow's fixed, tiny default bitmap font.
    """
    from PIL import Image, ImageDraw

    finite = image[np.isfinite(image)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    normalised = np.clip((image - lo) / (hi - lo), 0, 1)
    gray = (normalised * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    pil_image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil_image)

    bar_length_px = max(1, round(bar_length_um / um_per_px_x))
    height, width = gray.shape
    label = f"{bar_length_um:g} \u00b5m"

    if font_size is None:
        font_size = max(12, round(min(height, width) / 32))
    font, supports_micro_sign = _load_scale_bar_font(font_size)
    if not supports_micro_sign:
        label = label.replace("\u00b5", "u")
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_height = text_bbox[3] - text_bbox[1]
    label_gap_px = max(6, round(font_size / 2))

    if "right" in position:
        x1 = width - margin_px
        x0 = x1 - bar_length_px
    else:
        x0 = margin_px
        x1 = x0 + bar_length_px
    y1 = height - margin_px if "bottom" in position else margin_px + bar_thickness_px
    y0 = y1 - bar_thickness_px

    draw.rectangle([x0, y0, x1, y1], fill=color)
    text_y = y0 - text_height - label_gap_px if "bottom" in position else y1 + label_gap_px
    draw.text(((x0 + x1) / 2, text_y), label, fill=color, font=font, anchor="ma")

    return np.array(pil_image)


def plot_calibration_summary(rows: Sequence[CalibrationRow], out_path: str | Path) -> Path:
    """Plot zoom vs. \u00b5m/pixel for both axes, so a calibration run leaves a visual record."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))
    _plot_axis_series(axis, rows, "px_to_micron_x", "x", "#1A1A1A")
    _plot_axis_series(axis, rows, "px_to_micron_y", "y", "#6C3FA8")

    axis.set_xlabel("zoom")
    axis.set_ylabel("\u00b5m / pixel")
    axis.set_title("grid-target pixel calibration")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _plot_axis_series(
    axis, rows: Sequence[CalibrationRow], value_attr: str, label: str, color: str
) -> None:
    points = sorted(
        (row.zoom, getattr(row, value_attr)) for row in rows if getattr(row, value_attr) is not None
    )
    if not points:
        return
    zooms, values = zip(*points, strict=True)
    axis.plot(zooms, values, marker="o", linestyle="-", color=color, markersize=5, label=label)


def plot_pitch_diagnostic(
    measurement: PitchMeasurement,
    info: GridFileInfo,
    out_path: str | Path,
    zoom: float | None = None,
) -> Path:
    """Plot one file's intensity profile with its detected peaks, for debugging.

    `zoom` overrides `info.zoom` in the title - pass the value resolved by
    `resolve_zoom` so the diagnostic label matches what the calibration table
    actually used.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = info.zoom if zoom is None else zoom

    figure, axis = plt.subplots(figsize=(10, 3))
    axis.plot(measurement.profile, color="#1A1A1A", linewidth=0.8)
    if measurement.peak_positions.size:
        axis.plot(
            measurement.peak_positions,
            measurement.profile[measurement.peak_positions],
            "rx",
            markersize=6,
        )
    status = "OK" if measurement.ok else f"FLAGGED: {measurement.reason}"
    height, width = measurement.image_shape
    axis.set_title(
        f"{info.path.name} ({info.orientation}-axis, zoom {zoom:g}, {width}x{height} px) - {status}"
    )
    axis.set_xlabel("pixel")
    axis.set_ylabel("intensity (min-projected band)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)
    return out_path
