"""Tests against real ScanImage recordings.

These are opt-in: set ``SOCTO_TEST_DATA`` to a directory of ScanImage TIFFs
(the development set lives in ``~/Downloads/schuham_light_stim_tif``) to run
them. CI and a fresh checkout stay independent of multi-gigabyte files, while
the expectations below still pin down what was verified against real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scanimage_octo_reader import check_recording, read_recording

DATA_DIR = os.environ.get("SOCTO_TEST_DATA")

pytestmark = pytest.mark.skipif(
    not DATA_DIR or not Path(DATA_DIR).is_dir(),
    reason="set SOCTO_TEST_DATA to a directory of ScanImage TIFFs to run these",
)


def data_file(name: str) -> Path:
    path = Path(DATA_DIR) / name
    if not path.exists():
        pytest.skip(f"{name} is not present in SOCTO_TEST_DATA")
    return path


@pytest.fixture(scope="module")
def recording():
    """The light-stim recording, swept once and shared by the tests below."""
    return read_recording(data_file("LC_brain1__00001.tif"))


class TestLightStimRecording:
    """`LC_brain1__00001.tif`: 3-slice volumetric, one TTL every 10 s on AUX 0."""

    def test_geometry(self, recording):
        summary = recording.summary()
        assert summary["n_pages"] == 20000
        assert summary["n_channels"] == 1
        assert summary["volumetric"] is True
        # numSlices in this header is a stale 11; actualNumSlices is 3.
        assert summary["n_slices"] == 3
        assert summary["flyback_frames"] == 1
        assert summary["pages_per_volume"] == 4
        assert summary["n_volumes"] == 5000
        assert summary["si_version"] == "2022.1.0"
        assert summary["image_shape"] == [512, 512]
        assert summary["dtype"] == "int16"
        assert summary["bigtiff"] is True

    def test_light_stim_triggers(self, recording):
        assert set(recording.aux) == {0}
        events = recording.aux[0]
        assert events.size == 67
        intervals = events["timestamp_s"][1:] - events["timestamp_s"][:-1]
        assert float(intervals.min()) == pytest.approx(10.0, abs=0.01)
        assert float(intervals.max()) == pytest.approx(10.0, abs=0.01)

    def test_events_are_attributed_to_a_real_frame(self, recording):
        events = recording.aux[0]
        offsets = events["timestamp_s"] - events["frame_timestamp_s"]
        frame_period = 1.0 / recording.summary()["frame_rate_hz"]
        # Each trigger must fall within the frame it was logged in (a small
        # negative offset is possible when the trigger precedes the frame's
        # own timestamp by a scan line or two).
        assert offsets.min() > -frame_period
        assert offsets.max() < frame_period

    def test_no_i2c_in_this_recording(self, recording):
        assert recording.i2c == []

    def test_no_pages_are_lost(self, recording):
        assert recording.sweep.n_recovered_pages == 0
        assert int(recording.frames["frame_number"][-1]) == 20000

    def test_qc_passes(self, recording):
        report = check_recording(recording)
        assert report.ok, [issue.message for issue in report.errors]
        assert report.stats["frame_interval_rms_jitter"] < 1e-3

    def test_export_and_plot(self, recording, tmp_path):
        from scanimage_octo_reader.export import export_recording

        result = export_recording(recording, tmp_path)
        assert (result.directory / "metadata.json").exists()
        assert (result.directory / "aux_triggers" / "aux0.npy").exists()
        assert (result.directory / "plots" / "overview.png").stat().st_size > 0
        assert (result.directory / "plots" / "overview.pdf").stat().st_size > 0


class TestDenseTriggerRecording:
    """`brain1__00013.tif`: a trigger on every one of its 20 000 frames."""

    def test_dense_aux_line(self):
        recording = read_recording(data_file("brain1__00013.tif"))
        assert recording.aux[0].size == recording.n_pages == 20000
        assert check_recording(recording).ok

    def test_dense_plot_is_still_produced(self, tmp_path):
        from scanimage_octo_reader.plots import save_overview_figure

        recording = read_recording(data_file("brain1__00013.tif"))
        paths = save_overview_figure(recording, tmp_path)
        assert all(path.stat().st_size > 0 for path in paths)


class TestSplitAcquisition:
    """`brain1__00012_0000{1,2}.tif`: one acquisition across two files."""

    def test_merged_timeline_is_continuous(self):
        first = data_file("brain1__00012_00001.tif")
        data_file("brain1__00012_00002.tif")  # skip early if the sibling is absent

        single = read_recording(first)
        merged = read_recording(first, merge_acquisition=True)

        assert single.n_pages == 2000
        assert merged.n_pages == 4000
        assert len(merged.paths) == 2
        assert merged.name == "brain1__00012"
        # ScanImage numbers and timestamps run straight through the split.
        assert int(merged.frames["frame_number"][0]) == 1
        assert int(merged.frames["frame_number"][-1]) == 4000
        timestamps = merged.frames["frame_timestamp_s"]
        assert timestamps[2000] > timestamps[1999]
        assert check_recording(merged).ok
