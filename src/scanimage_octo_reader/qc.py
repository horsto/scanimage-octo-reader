"""Quality-control checks on a swept recording.

These exist because ScanImage TIFFs fail in quiet ways: a dropped frame, a
truncated file, a clock hiccup or a mid-file format change all leave the file
perfectly readable while making downstream analysis wrong. Every check
therefore reports rather than raises, and severity is separated from
detection so a caller can decide what is fatal.

One subtlety runs through all of the timestamp checks: with multiple
channels, the pages of a single frame-scan share a frame number *and* a
timestamp. Duplicates across channels are expected, not a fault, so the
timing checks operate on one page per frame-scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scanimage_octo_reader.acquisition import Recording

__all__ = ["QCIssue", "QCReport", "check_recording"]

# Relative deviation of the frame interval from its median that is treated as
# a warning, and as an error. ScanImage's frame clock is FPGA-derived and
# normally jitters far below a percent, so 1% already means something is off,
# while 10% indicates dropped frames or a corrupt file.
_JITTER_WARN = 0.01
_JITTER_ERROR = 0.10


@dataclass(frozen=True)
class QCIssue:
    """One finding: `level` is ``'error'``, ``'warning'`` or ``'info'``."""

    level: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QCReport:
    """All findings for one recording."""

    issues: list[QCIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, **details: Any) -> None:
        self.issues.append(QCIssue(level=level, code=code, message=message, details=details))

    @property
    def errors(self) -> list[QCIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[QCIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def infos(self) -> list[QCIssue]:
        return [issue for issue in self.issues if issue.level == "info"]

    @property
    def ok(self) -> bool:
        """True when nothing was found that invalidates the recording."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "stats": self.stats,
            "issues": [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    **({"details": issue.details} if issue.details else {}),
                }
                for issue in self.issues
            ],
        }


def _first_channel_mask(frames: np.ndarray) -> np.ndarray:
    """Select one page per frame-scan, so channel duplicates don't skew timing checks."""
    channels = frames["channel"]
    if channels.size == 0:
        return np.zeros(0, dtype=bool)
    return channels == channels.min()


def check_recording(recording: Recording) -> QCReport:
    """Run every check against `recording` and return the collected findings."""
    report = QCReport()
    frames = recording.frames

    for warning in recording.warnings:
        report.add("warning", "read_warning", warning)
    if recording.geometry.warning:
        report.add(
            "warning",
            "geometry_fallback",
            f"could not confirm the volumetric layout: {recording.geometry.warning}",
        )

    if frames.size == 0:
        report.add("error", "no_pages", "the file contains no pages")
        return report

    _check_frame_numbers(recording, frames, report)
    _check_timestamps(recording, frames, report)
    _check_volume_timing(recording, frames, report)
    _check_volume_alignment(recording, report)
    _check_header_stability(recording, report)
    _check_hardware_flags(frames, report)
    _check_events(recording, report)
    return report


def _check_frame_numbers(recording: Recording, frames: np.ndarray, report: QCReport) -> None:
    numbers = frames["frame_number"]
    missing = int(np.count_nonzero(numbers < 0))
    if missing:
        report.add(
            "warning",
            "frame_number_missing",
            f"{missing} page(s) have no readable frameNumbers entry",
            n_pages=missing,
        )

    present = numbers[numbers >= 0]
    if present.size == 0:
        return

    unique = np.unique(present)
    report.stats["first_frame_number"] = int(unique[0])
    report.stats["last_frame_number"] = int(unique[-1])
    report.stats["n_unique_frame_numbers"] = int(unique.size)

    # With N channels each frame number legitimately appears N times.
    expected_repeats = recording.geometry.n_channels
    counts = np.bincount(np.searchsorted(unique, present))
    unexpected = int(np.count_nonzero(counts != expected_repeats))
    if unexpected:
        report.add(
            "warning",
            "frame_number_repeats",
            f"{unexpected} frame number(s) do not appear exactly {expected_repeats} time(s), "
            "which is what the channel count implies",
            n_frame_numbers=unexpected,
            expected_repeats=expected_repeats,
        )

    gaps = np.diff(unique)
    if np.any(gaps != 1):
        n_gaps = int(np.count_nonzero(gaps != 1))
        missing_frames = int(np.sum(gaps[gaps > 1] - 1))
        report.add(
            "error",
            "frame_number_gaps",
            f"the frame numbering has {n_gaps} gap(s), missing {missing_frames} frame(s) - "
            "frames were dropped, or the selected files are not a contiguous set",
            n_gaps=n_gaps,
            n_missing_frames=missing_frames,
        )


def _check_timestamps(recording: Recording, frames: np.ndarray, report: QCReport) -> None:
    mask = _first_channel_mask(frames)
    timestamps = frames["frame_timestamp_s"][mask]
    finite = timestamps[np.isfinite(timestamps)]

    n_nonfinite = int(timestamps.size - finite.size)
    if n_nonfinite:
        report.add(
            "warning",
            "timestamp_unreadable",
            f"{n_nonfinite} frame timestamp(s) could not be read",
            n_frames=n_nonfinite,
        )
    if finite.size == 0:
        return

    n_negative = int(np.count_nonzero(finite < 0))
    if n_negative:
        report.add(
            "error",
            "timestamp_negative",
            f"{n_negative} frame timestamp(s) are negative, which suggests a corrupt file",
            n_frames=n_negative,
        )

    if finite.size < 2:
        return

    intervals = np.diff(finite)
    median = float(np.median(intervals))
    report.stats["median_frame_interval_s"] = median
    report.stats["implied_frame_rate_hz"] = (1.0 / median) if median > 0 else None

    n_nonmonotonic = int(np.count_nonzero(intervals <= 0))
    if n_nonmonotonic:
        report.add(
            "error",
            "timestamp_not_monotonic",
            f"{n_nonmonotonic} frame timestamp(s) do not increase; frame or channel counts "
            "are probably being misinterpreted",
            n_frames=n_nonmonotonic,
        )

    if median <= 0:
        return

    # Jitter as a fraction of the median interval. RMS captures overall
    # stability; the extremes catch isolated hiccups that RMS would dilute.
    deviation = np.abs(intervals / median - 1.0)
    rms = float(np.sqrt(np.mean(deviation**2)))
    worst = float(deviation.max())
    report.stats["frame_interval_rms_jitter"] = rms
    report.stats["frame_interval_max_deviation"] = worst

    if worst >= _JITTER_ERROR:
        report.add(
            "error",
            "timestamp_jitter",
            f"the worst frame interval deviates {worst:.1%} from the median "
            f"({median * 1e3:.3f} ms); frames were likely dropped",
            rms_jitter=rms,
            max_deviation=worst,
        )
    elif worst >= _JITTER_WARN:
        report.add(
            "warning",
            "timestamp_jitter",
            f"the worst frame interval deviates {worst:.1%} from the median "
            f"({median * 1e3:.3f} ms)",
            rms_jitter=rms,
            max_deviation=worst,
        )


def _check_volume_timing(recording: Recording, frames: np.ndarray, report: QCReport) -> None:
    """Check the volume timeline, which is what per-cell activity is sampled on.

    For a volumetric acquisition the per-plane interval says little about the
    sampling rate of a given neuron: a cell is revisited once per *volume*.
    This derives the volume interval from the data (rather than trusting
    `hRoiManager.scanVolumeRate`) and cross-checks the two.
    """
    stride = recording.geometry.pages_per_volume
    n_volumes = frames.size // stride if stride else 0
    if n_volumes < 2:
        return

    # One timestamp per volume: its first page.
    starts = frames["frame_timestamp_s"][: n_volumes * stride : stride]
    finite = starts[np.isfinite(starts)]
    if finite.size < 2:
        return

    intervals = np.diff(finite)
    median = float(np.median(intervals))
    if median <= 0:
        report.add(
            "error",
            "volume_interval_nonpositive",
            "consecutive volumes do not advance in time; the volume layout is probably "
            "being misinterpreted",
        )
        return

    report.stats["median_volume_interval_s"] = median
    report.stats["implied_volume_rate_hz"] = 1.0 / median

    header_rate = recording.geometry.volume_rate_hz
    if header_rate:
        deviation = abs((1.0 / median) - header_rate) / header_rate
        report.stats["volume_rate_header_deviation"] = deviation
        if deviation > 0.01:
            report.add(
                "warning",
                "volume_rate_mismatch",
                f"the volume rate implied by the timestamps ({1.0 / median:.6g} Hz) differs "
                f"by {deviation:.1%} from the header's scanVolumeRate ({header_rate:.6g} Hz)",
                implied_hz=1.0 / median,
                header_hz=header_rate,
            )

    worst = float(np.abs(intervals / median - 1.0).max())
    report.stats["volume_interval_rms_jitter"] = float(
        np.sqrt(np.mean((intervals / median - 1.0) ** 2))
    )
    report.stats["volume_interval_max_deviation"] = worst
    if worst >= _JITTER_ERROR:
        report.add(
            "error",
            "volume_interval_jitter",
            f"the worst volume interval deviates {worst:.1%} from the median "
            f"({median * 1e3:.3f} ms); volumes were likely dropped",
            max_deviation=worst,
        )
    elif worst >= _JITTER_WARN:
        report.add(
            "warning",
            "volume_interval_jitter",
            f"the worst volume interval deviates {worst:.1%} from the median "
            f"({median * 1e3:.3f} ms)",
            max_deviation=worst,
        )


def _check_volume_alignment(recording: Recording, report: QCReport) -> None:
    geometry = recording.geometry
    stride = geometry.pages_per_volume
    if stride <= 1:
        return
    remainder = recording.n_pages % stride
    report.stats["pages_per_volume"] = stride
    report.stats["n_volumes"] = geometry.n_volumes(recording.n_pages)
    if remainder:
        report.add(
            "warning",
            "partial_volume",
            f"{recording.n_pages} pages is not a whole number of volumes "
            f"({stride} pages each); the last {remainder} page(s) form an incomplete volume",
            n_trailing_pages=remainder,
        )


def _check_header_stability(recording: Recording, report: QCReport) -> None:
    key_sets = recording.sweep.key_sets
    if len(key_sets) > 1:
        report.add(
            "warning",
            "page_header_drift",
            f"page headers use {len(key_sets)} different key sets across the recording; "
            "the frame table may be incomplete for some pages",
            n_key_sets=len(key_sets),
            page_counts=sorted(key_sets.values(), reverse=True),
        )
    if recording.sweep.unknown_keys:
        report.add(
            "info",
            "unknown_page_keys",
            "page headers contain key(s) without a dedicated converter: "
            + ", ".join(sorted(recording.sweep.unknown_keys)),
            keys=sorted(recording.sweep.unknown_keys),
        )

    if recording.sweep.n_recovered_pages:
        report.add(
            "warning",
            "recovered_pages",
            f"{recording.sweep.n_recovered_pages} page(s) were found by following the IFD "
            "chain past the end of what tifffile reported; they are included here, but "
            "other tools reading this file may silently miss them",
            n_pages=recording.sweep.n_recovered_pages,
        )

    end_flags = recording.frames["end_of_acquisition"]
    if end_flags.size and not np.any(end_flags == 1):
        report.add(
            "info",
            "no_end_of_acquisition",
            "no page is flagged as the end of the acquisition, so the recording may have "
            "been aborted or the file set may be incomplete",
        )


def _check_hardware_flags(frames: np.ndarray, report: QCReport) -> None:
    over_voltage = int(np.count_nonzero(frames["dc_over_voltage"] == 1))
    if over_voltage:
        report.add(
            "warning",
            "dc_over_voltage",
            f"{over_voltage} frame(s) were acquired with a DC over-voltage flag set, so the "
            "pixel data of those frames may be clipped",
            n_frames=over_voltage,
        )


def _check_events(recording: Recording, report: QCReport) -> None:
    for line, events in sorted(recording.aux.items()):
        timestamps = events["timestamp_s"]
        report.stats[f"n_aux{line}_events"] = int(timestamps.size)
        n_negative = int(np.count_nonzero(timestamps < 0))
        if n_negative:
            report.add(
                "warning",
                "aux_negative_timestamp",
                f"AUX line {line} has {n_negative} negative timestamp(s), which ScanImage "
                "uses as a sentinel rather than as data",
                line=line,
                n_events=n_negative,
            )
        if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
            report.add(
                "warning",
                "aux_not_monotonic",
                f"AUX line {line} timestamps are not in increasing order",
                line=line,
            )

    packets = recording.i2c
    report.stats["n_i2c_packets"] = len(packets)
    n_invalid = sum(1 for record in packets if not record.packet.is_valid_timestamp)
    if n_invalid:
        report.add(
            "warning",
            "i2c_invalid_timestamp",
            f"{n_invalid} I2C packet(s) carry a negative or non-finite timestamp; they are "
            "exported with valid=False rather than dropped",
            n_packets=n_invalid,
        )
