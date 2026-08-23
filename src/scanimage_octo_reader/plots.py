"""A single overview figure per recording: is the timeline sane, and what fired?

Three panels sharing one acquisition-time axis:

1. **Frame interval** - inter-frame interval against frame time, with the
   median as a reference. Dropped frames and clock hiccups stand out here
   long before they show up in any summary number.
2. **Trigger raster** - one row per non-empty AUX line plus one for I2C.
   Sparse lines are drawn as event ticks; a line carrying thousands of events
   (ScanImage happily records a trigger on *every* frame) would render as a
   solid block, so above `DENSE_EVENT_THRESHOLD` the row switches to a binned
   event-rate trace instead.
3. **Cumulative event count** - a regular stimulus train reads as a straight
   line, and a dropout as a kink.

The non-interactive ``Agg`` backend is selected on import, so this works
headless and over SSH; these figures are files, never windows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from scanimage_octo_reader.triggers import filter_valid_aux_events, filter_valid_i2c_records

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from scanimage_octo_reader.acquisition import Recording

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FORMATS",
    "DENSE_EVENT_THRESHOLD",
    "FONT_SCALE",
    "PALETTE",
    "build_overview_figure",
    "save_overview_figure",
]

# Above this many events, a raster row becomes an unreadable solid block, so
# it is rendered as a binned rate trace instead.
DENSE_EVENT_THRESHOLD = 5000

# Number of bins used for those rate traces.
_RATE_BINS = 600

# Written for every figure: a raster for looking at, and a vector version for
# dropping into a figure. See `_apply_style` on keeping PDF text editable.
DEFAULT_FORMATS = ("png", "pdf")

FONT_SCALE = 1.25

# One scheme for the whole figure: near-black for measured data, violet for
# everything derived from it or overlaid on it. Sticking to a single accent hue
# means nothing in the figure competes for attention on colour alone.
INK = "#1A1A1A"
VIOLET = "#6C3FA8"

# Violet family, ordered by lightness so several trigger lines stay separable
# without leaving the scheme.
PALETTE = [
    VIOLET,
    "#3B1F5E",  # deep violet
    "#9163CB",  # light violet
    "#553C9A",  # indigo violet
    "#B79CED",  # pale violet
    "#7E5A9B",  # muted violet
]

# Roles are fixed so a colour always means the same thing across panels: the
# measured frame clock is ink, the median reference and the derived rate traces
# are violet.
_INTERVAL_COLOUR = INK
_MEDIAN_COLOUR = VIOLET
_RATE_COLOUR = VIOLET
# Trigger lines take these in order, so AUX 0 keeps its colour whether or not
# AUX 1 is present.
_EVENT_COLOURS = PALETTE


def _apply_style() -> None:
    """Apply the figure style.

    Set per-figure rather than globally at import time, so importing this
    module never changes the appearance of a caller's own plots.

    Font sizes are left to seaborn's context scaled by `FONT_SCALE`, so the
    whole figure scales coherently instead of being pinned point by point.
    """
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=PALETTE,
        font_scale=FONT_SCALE,
        rc={
            "figure.facecolor": "white",
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "savefig.bbox": "tight",
            # Keep text as text in vector output: matplotlib's default Type 3
            # fonts are converted to outlines by most editors, so labels
            # cannot be edited or re-typeset. Type 42 (TrueType) embeds real
            # glyphs, which Illustrator/Inkscape treat as editable text.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # SVG: 'none' leaves text as <text> referencing the font by name.
            "svg.fonttype": "none",
        },
    )


def _frame_times(recording: Recording) -> np.ndarray:
    """Frame timestamps, one per frame-scan (channel duplicates removed)."""
    frames = recording.frames
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    channels = frames["channel"]
    timestamps = frames["frame_timestamp_s"][channels == channels.min()]
    return timestamps[np.isfinite(timestamps)]


def _volume_times(recording: Recording) -> np.ndarray:
    """One timestamp per complete volume - the per-cell sampling timeline."""
    frames = recording.frames
    stride = recording.geometry.pages_per_volume
    if frames.size == 0 or stride <= 0:
        return np.empty(0, dtype=np.float64)
    n_volumes = frames.size // stride
    if n_volumes < 1:
        return np.empty(0, dtype=np.float64)
    starts = frames["frame_timestamp_s"][: n_volumes * stride : stride]
    return starts[np.isfinite(starts)]


def _event_series(recording: Recording) -> list[tuple[str, np.ndarray]]:
    """Every non-empty event timeline as ``(label, timestamps)``, top row last.

    Sentinel (negative-timestamp) events are excluded - see
    `scanimage_octo_reader.triggers.filter_valid_aux_events` - since a stale
    one from well before this acquisition began would otherwise dominate the
    shared time axis and squash every real event into a sliver at one edge.
    """
    series: list[tuple[str, np.ndarray]] = []
    for line, events in sorted(recording.aux.items()):
        timestamps = filter_valid_aux_events(events)["timestamp_s"]
        if timestamps.size:
            series.append((f"AUX {line}", np.sort(timestamps)))
    valid_i2c = filter_valid_i2c_records(recording.i2c)
    if valid_i2c:
        i2c_times = np.array([record.packet.timestamp for record in valid_i2c], dtype=np.float64)
        if i2c_times.size:
            series.append(("I2C", np.sort(i2c_times)))
    return series


def _plot_frame_intervals(axis, frame_times: np.ndarray, volume_times: np.ndarray) -> None:
    if frame_times.size < 2:
        axis.text(
            0.5,
            0.5,
            "not enough frame timestamps to show intervals",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_ylabel("frame interval (ms)")
        return

    intervals_ms = np.diff(frame_times) * 1e3
    median_ms = float(np.median(intervals_ms))
    # Plot against the *later* frame of each pair, so an anomaly lines up
    # with the frame that arrived late.
    axis.plot(
        frame_times[1:],
        intervals_ms,
        linewidth=0.7,
        color=_INTERVAL_COLOUR,
        label="plane-to-plane",
    )
    axis.axhline(
        median_ms,
        color=_MEDIAN_COLOUR,
        linewidth=1.1,
        linestyle="--",
        label=f"median {median_ms:.3f} ms ({1e3 / median_ms:.4g} Hz)",
    )
    axis.set_ylabel("plane interval (ms)")
    # Intervals cluster tightly around a large-ish value, and matplotlib's
    # default offset notation ('+3.331e1' in a corner) is easy to misread and
    # collides with the panel title. Absolute tick labels are unambiguous.
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.legend(loc="best")

    # The volume interval is the sampling period that matters for per-cell
    # activity. It cannot share this axis (it is a multiple of the plane
    # interval), so it is stated in the title rather than drawn.
    title = f"frame clock - median {median_ms:.3f} ms ({1e3 / median_ms:.4g} Hz per plane)"
    if volume_times.size > 2:
        volume_median_ms = float(np.median(np.diff(volume_times)) * 1e3)
        title += f"; {volume_median_ms:.3f} ms per volume ({1e3 / volume_median_ms:.4g} Hz)"
    axis.set_title(title, loc="left")

    # A few dropped frames would otherwise flatten the whole trace; clip the
    # view to the bulk of the distribution and say so.
    low, high = np.percentile(intervals_ms, [0.5, 99.5])
    span = max(high - low, median_ms * 0.02, 1e-6)
    lower, upper = low - span, high + span
    if intervals_ms.min() < lower or intervals_ms.max() > upper:
        axis.set_ylim(lower, upper)
        axis.text(
            0.01,
            0.04,
            "y-axis clipped to the 0.5-99.5 percentile range",
            transform=axis.transAxes,
        )


def _plot_raster(
    axis, series: list[tuple[str, np.ndarray]], time_span: tuple[float, float]
) -> None:
    labels = []
    for row, (label, timestamps) in enumerate(series):
        colour = _EVENT_COLOURS[row % len(_EVENT_COLOURS)]
        if timestamps.size > DENSE_EVENT_THRESHOLD:
            counts, edges = np.histogram(timestamps, bins=_RATE_BINS, range=time_span)
            widths = np.diff(edges)
            rate = counts / np.where(widths > 0, widths, np.nan)
            peak = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
            if peak > 0:
                # Scale into the row's own band so rows stay separable. Drawn
                # as a line rather than a filled area: a steady trigger-per-
                # frame line would otherwise fill its band completely and look
                # like the solid block this mode exists to avoid.
                baseline = row - 0.4
                normalised = baseline + 0.8 * (rate / peak)
                axis.step(
                    edges[:-1],
                    normalised,
                    where="post",
                    linewidth=1.0,
                    color=_RATE_COLOUR,
                )
            labels.append(f"{label}\n{timestamps.size} ev · rate (peak {peak:.0f}/s)")
        else:
            axis.eventplot(
                timestamps,
                lineoffsets=row,
                linelengths=0.66,
                linewidths=1.1,
                colors=[colour],
            )
            labels.append(f"{label}\n{timestamps.size} ev")

    axis.set_yticks(range(len(series)))
    axis.set_yticklabels(labels)
    axis.set_ylim(-0.6, len(series) - 0.4)
    # No y-label: the row labels already name each line, and a label here
    # only crowds them.
    axis.set_title("trigger events", loc="left")
    axis.grid(axis="y", visible=False)


def _plot_cumulative(axis, series: list[tuple[str, np.ndarray]]) -> None:
    for row, (label, timestamps) in enumerate(series):
        axis.plot(
            timestamps,
            np.arange(1, timestamps.size + 1),
            linewidth=1.6,
            color=_EVENT_COLOURS[row % len(_EVENT_COLOURS)],
            label=f"{label} ({timestamps.size})",
        )
    axis.set_ylabel("cumulative events")
    axis.set_title("cumulative count", loc="left")
    axis.legend(loc="upper left")


def _title(recording: Recording) -> str:
    # summary = recording.summary()
    parts = [recording.name]
    # if summary.get("si_version"):
    #     parts.append(f"SI {summary['si_version']}")
    # geometry = (
    #     f"{summary['n_pages']} pages, {summary['n_volumes']} volumes, "
    #     f"{summary['n_slices']} slice(s), {summary['n_channels']} channel(s)"
    # )
    # parts.append(geometry)
    # if summary.get("frame_rate_hz"):
    #     parts.append(f"{summary['frame_rate_hz']:.4g} Hz planes")
    # if summary.get("volume_rate_hz") and summary.get("volumetric"):
    #     parts.append(f"{summary['volume_rate_hz']:.4g} Hz volumes")
    # if summary.get("n_files", 1) > 1:
    #     parts.append(f"{summary['n_files']} files merged")
    return parts[0]


def build_overview_figure(recording: Recording) -> Figure:
    """Render the overview figure for `recording` and return it (unsaved)."""
    _apply_style()
    frame_times = _frame_times(recording)
    volume_times = _volume_times(recording)
    series = _event_series(recording)

    # The raster only needs room for the rows it actually has, so a file with
    # a single AUX line does not get a mostly-empty middle panel.
    raster_height = 0.8 + 0.5 * len(series)
    height_ratios = [2.4, raster_height, 2.0] if series else [2.4]
    figure, axes = plt.subplots(
        len(height_ratios),
        1,
        sharex=True,
        figsize=(11, 1.6 + 1.9 * sum(height_ratios) / 2.0),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)

    _plot_frame_intervals(axes[0], frame_times, volume_times)

    if series:
        if frame_times.size:
            span = (float(frame_times.min()), float(frame_times.max()))
        else:
            all_times = np.concatenate([timestamps for _label, timestamps in series])
            span = (float(all_times.min()), float(all_times.max()))
        if span[1] <= span[0]:
            span = (span[0], span[0] + 1.0)
        _plot_raster(axes[1], series, span)
        _plot_cumulative(axes[2], series)
    else:
        axes[0].text(
            0.5,
            0.9,
            "no AUX trigger or I2C events recorded in this file",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color=_MEDIAN_COLOUR,
        )

    # Mark the joins of a merged acquisition, so a discontinuity can be told
    # apart from a file boundary at a glance.
    if len(recording.pages_per_file) > 1 and frame_times.size:
        _mark_file_boundaries(recording, axes)

    axes[-1].set_xlabel("acquisition time (s)")
    sns.despine(fig=figure, left=True, bottom=True)
    figure.suptitle(_title(recording), y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return figure


def _mark_file_boundaries(recording: Recording, axes) -> None:
    frames = recording.frames
    channels = frames["channel"]
    first_channel = frames[channels == channels.min()]
    file_indices = first_channel["file_index"]
    timestamps = first_channel["frame_timestamp_s"]

    for boundary in np.unique(file_indices)[1:]:
        start = np.argmax(file_indices == boundary)
        boundary_time = timestamps[start]
        if not np.isfinite(boundary_time):
            continue
        for axis in axes:
            axis.axvline(boundary_time, color="0.45", linewidth=0.9, linestyle=":")


def save_overview_figure(
    recording: Recording,
    out_dir: str | Path,
    formats: str | Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 150,
    filename: str = "overview",
) -> list[Path]:
    """Render the overview figure once and save it in each requested format.

    Returns the paths written, in the order given. The figure is built a single
    time, so asking for PNG and PDF costs one render rather than two.

    PDF (and SVG) output keeps text as editable text rather than outlines - see
    `_apply_style` for the font settings that make that work.
    """
    if isinstance(formats, str):
        formats = (formats,)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    figure = build_overview_figure(recording)
    paths: list[Path] = []
    try:
        for image_format in formats:
            path = directory / f"{filename}.{image_format}"
            figure.savefig(path, dpi=dpi)
            paths.append(path)
    finally:
        plt.close(figure)
    return paths
