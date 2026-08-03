"""Sweeping page headers into the frame table and event tables."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    FRAME_PERIOD_S,
    build_header,
    build_page_description,
    descriptions_for,
    write_tif,
)

from scanimage_octo_reader import read_recording
from scanimage_octo_reader.triggers import (
    decode_i2c_key_values,
    i2c_payload_matrix,
    i2c_table,
)


class TestFrameTable:
    def test_scalar_fields_are_read(self, single_plane_tif):
        frames = read_recording(single_plane_tif).frames
        assert frames.size == 20
        assert frames["frame_number"].tolist() == list(range(1, 21))
        assert frames["page_index"].tolist() == list(range(20))
        assert frames["frame_timestamp_s"][1] == pytest.approx(FRAME_PERIOD_S)
        # ScanImage's -1 sentinel must survive as a real value, not become NaN.
        assert frames["acq_trigger_timestamp_s"][0] == -1.0
        assert frames["end_of_acquisition"][-1] == 1

    def test_table_has_no_object_fields(self, single_plane_tif):
        """The table must be loadable without ``allow_pickle``."""
        frames = read_recording(single_plane_tif).frames
        assert all(frames.dtype[name].kind not in ("O", "V") for name in frames.dtype.names)

    def test_per_page_event_counts(self, single_plane_tif):
        frames = read_recording(single_plane_tif).frames
        assert frames["n_aux0"].sum() == 2
        assert frames["n_aux0"][3] == 1
        assert frames["n_aux1"].sum() == 0

    def test_two_channels_share_frame_numbers(self, two_channel_tif):
        recording = read_recording(two_channel_tif)
        frames = recording.frames
        assert recording.geometry.n_channels == 2
        assert frames["channel"][:4].tolist() == [0, 1, 0, 1]
        assert frames["frame_number"][:4].tolist() == [1, 1, 2, 2]

    def test_unknown_keys_are_kept_not_dropped(self, tmp_path):
        """A newer ScanImage version must not break the sweep."""
        path = tmp_path / "future__00001.tif"
        descriptions = [
            build_page_description(i + 1, i * FRAME_PERIOD_S, extra={"newFangledField": "7"})
            for i in range(4)
        ]
        write_tif(path, descriptions)
        sweep = read_recording(path).sweep
        assert sweep.unknown_keys == {"newFangledField": "7"}
        assert sweep.frames.size == 4


class TestAuxAttribution:
    def test_events_carry_their_frame_context(self, single_plane_tif):
        events = read_recording(single_plane_tif).aux[0]
        assert events.size == 2
        assert events["page_index"].tolist() == [3, 11]
        assert events["frame_number"].tolist() == [4, 12]
        # The trigger fell inside the frame it is attributed to.
        offsets = events["timestamp_s"] - events["frame_timestamp_s"]
        assert np.all(offsets >= 0)
        assert np.all(offsets < FRAME_PERIOD_S)

    def test_volume_and_slice_context(self, volumetric_tif):
        events = read_recording(volumetric_tif).aux[0]
        # Page 5 of a 4-page volume layout is volume 1, slice 1.
        assert events["volume_index"].tolist() == [1]
        assert events["slice_index"].tolist() == [1]

    def test_both_timebases_are_provided(self, volumetric_tif):
        """A trigger must be placeable on the plane *and* the volume timeline.

        Per-cell activity is sampled once per volume, so the offset that
        matters for aligning a stimulus to a trace is the one measured from
        the volume start, not from the plane that happened to be scanning.
        """
        recording = read_recording(volumetric_tif)
        events = recording.aux[0]
        volume_period = recording.geometry.pages_per_volume * FRAME_PERIOD_S

        # The event landed on page 5: slice 1 of volume 1, i.e. one plane into
        # a volume that starts at page 4.
        assert events["frame_timestamp_s"][0] == pytest.approx(5 * FRAME_PERIOD_S)
        assert events["volume_timestamp_s"][0] == pytest.approx(4 * FRAME_PERIOD_S)

        assert 0 <= events["offset_in_frame_s"][0] < FRAME_PERIOD_S
        assert FRAME_PERIOD_S <= events["offset_in_volume_s"][0] < volume_period
        assert events["timestamp_s"][0] == pytest.approx(
            events["volume_timestamp_s"][0] + events["offset_in_volume_s"][0]
        )

    def test_timebases_coincide_for_a_single_plane(self, single_plane_tif):
        events = read_recording(single_plane_tif).aux[0]
        assert events["volume_timestamp_s"].tolist() == pytest.approx(
            events["frame_timestamp_s"].tolist()
        )
        assert events["offset_in_volume_s"].tolist() == pytest.approx(
            events["offset_in_frame_s"].tolist()
        )

    def test_frame_table_carries_the_volume_timestamp(self, volumetric_tif):
        frames = read_recording(volumetric_tif).frames
        # Every page of a volume, flyback included, shares the volume's start.
        assert frames["volume_timestamp_s"][:4].tolist() == pytest.approx([0.0] * 4)
        assert frames["volume_timestamp_s"][4:8].tolist() == pytest.approx([4 * FRAME_PERIOD_S] * 4)

    def test_empty_lines_are_omitted_entirely(self, single_plane_tif):
        aux = read_recording(single_plane_tif).aux
        assert set(aux) == {0}

    def test_multiple_events_on_one_page(self, tmp_path):
        path = tmp_path / "multi__00001.tif"
        descriptions = descriptions_for(4, aux_events={2: [1]}, aux_text="[0.10 0.20 0.30]")
        write_tif(path, descriptions)
        events = read_recording(path).aux[2]
        assert events["timestamp_s"].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert events["page_index"].tolist() == [1, 1, 1]

    def test_comma_separated_aux_arrays(self, tmp_path):
        """Older ScanImage versions use commas instead of spaces."""
        path = tmp_path / "comma__00001.tif"
        write_tif(path, descriptions_for(4, aux_events={0: [1]}, aux_text="[0.11, 0.22]"))
        events = read_recording(path).aux[0]
        assert events["timestamp_s"].tolist() == pytest.approx([0.11, 0.22])

    def test_all_four_lines(self, tmp_path):
        path = tmp_path / "fourlines__00001.tif"
        write_tif(path, descriptions_for(6, aux_events={0: [0], 1: [1], 2: [2], 3: [3]}))
        aux = read_recording(path).aux
        assert set(aux) == {0, 1, 2, 3}
        assert all(events.size == 1 for events in aux.values())


class TestI2C:
    def test_text_flavour_end_to_end(self, tmp_path):
        path = tmp_path / "i2ctext__00001.tif"
        write_tif(
            path,
            descriptions_for(
                4,
                i2c_events={
                    1: "{{0.50, 'treadmill_9'} {0.55, 'lick_1'} }",
                    2: "{{0.90, 'treadmill_11'}}",
                },
            ),
        )
        recording = read_recording(path)
        assert len(recording.i2c) == 3
        assert recording.frames["n_i2c"].tolist() == [0, 2, 1, 0]

        table = i2c_table(recording.i2c)
        assert table["payload_kind"].tolist() == [b"text"] * 3
        assert table["page_index"].tolist() == [1, 1, 2]
        assert table["frame_number"].tolist() == [2, 2, 3]
        assert table["valid"].all()

    def test_byte_flavour_payload_matrix_is_padded(self, tmp_path):
        path = tmp_path / "i2cbytes__00001.tif"
        write_tif(
            path,
            descriptions_for(3, i2c_events={0: "{{0.1, [1 2 3]} {0.2, [9]} }"}),
        )
        recording = read_recording(path)
        table = i2c_table(recording.i2c)
        matrix = i2c_payload_matrix(recording.i2c)
        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.uint8
        assert matrix[0].tolist() == [1, 2, 3]
        # Shorter payloads are zero-padded; the true length is in the table.
        assert matrix[1].tolist() == [9, 0, 0]
        assert table["payload_length"].tolist() == [3, 1]

    def test_key_value_decode(self, tmp_path):
        path = tmp_path / "i2cdecode__00001.tif"
        write_tif(
            path,
            descriptions_for(
                4,
                i2c_events={
                    0: "{{0.1, 'treadmill_9'}}",
                    1: "{{0.2, 'treadmill_10'} {0.25, 'lick_1'} }",
                },
            ),
        )
        decoded, reason = decode_i2c_key_values(read_recording(path).i2c)
        assert reason is None
        assert set(decoded) == {"treadmill", "lick"}
        assert decoded["treadmill"]["value"].tolist() == [9.0, 10.0]
        assert decoded["lick"]["value"].tolist() == [1.0]

    def test_decode_refuses_partial_results(self, tmp_path):
        """A single non-conforming payload means the convention isn't in use."""
        path = tmp_path / "i2cmixed__00001.tif"
        write_tif(
            path,
            descriptions_for(3, i2c_events={0: "{{0.1, 'treadmill_9'} {0.2, 'plain'} }"}),
        )
        decoded, reason = decode_i2c_key_values(read_recording(path).i2c)
        assert decoded == {}
        assert "convention" in reason

    def test_decode_skipped_for_byte_payloads(self, tmp_path):
        path = tmp_path / "i2craw__00001.tif"
        write_tif(path, descriptions_for(3, i2c_events={0: "{{0.1, [1 2]}}"}))
        decoded, reason = decode_i2c_key_values(read_recording(path).i2c)
        assert decoded == {}
        assert "raw bytes" in reason

    def test_negative_timestamps_are_flagged_not_dropped(self, tmp_path):
        path = tmp_path / "i2cneg__00001.tif"
        write_tif(path, descriptions_for(3, i2c_events={0: "{{-1.0, 'a_1'} {0.5, 'a_2'} }"}))
        table = i2c_table(read_recording(path).i2c)
        assert table.size == 2
        assert table["valid"].tolist() == [False, True]


class TestPageRecovery:
    def test_all_pages_are_read(self, tmp_path):
        """tifffile can under-report SI-style page counts; nothing may be lost."""
        path = tmp_path / "count__00001.tif"
        write_tif(path, descriptions_for(30))
        recording = read_recording(path)
        assert recording.n_pages == 30
        assert recording.frames["frame_number"].tolist() == list(range(1, 31))


class TestTriggerSummary:
    def test_summary_reports_counts_and_regularity(self, tmp_path):
        path = tmp_path / "regular__00001.tif"
        # A trigger every 4th page is a regular train.
        write_tif(path, descriptions_for(20, aux_events={0: [0, 4, 8, 12, 16]}))
        summary = read_recording(path).trigger_summary()
        assert summary["aux"]["aux0"]["n_events"] == 5
        assert summary["aux"]["aux0"]["median_interval_s"] == pytest.approx(4 * FRAME_PERIOD_S)
        assert summary["aux_lines_empty"] == [1, 2, 3]
        assert summary["i2c"]["n_packets"] == 0

    def test_header_only_file_still_summarises(self, tmp_path):
        path = tmp_path / "quiet__00001.tif"
        write_tif(path, descriptions_for(5), header=build_header())
        summary = read_recording(path).trigger_summary()
        assert summary["aux"] == {}
        assert summary["aux_lines_present"] == []
