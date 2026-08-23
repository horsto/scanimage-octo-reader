"""Export products, QC detection, and the overview figure."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import FRAME_PERIOD_S, build_page_description, descriptions_for, write_tif

from scanimage_octo_reader import read_recording
from scanimage_octo_reader.export import build_metadata, export_recording, to_jsonable
from scanimage_octo_reader.qc import check_recording


class TestJsonSafety:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (math.inf, "Infinity"),
            (-math.inf, "-Infinity"),
            (float("nan"), "NaN"),
            (np.float32(1.5), 1.5),
            (np.int64(3), 3),
            (np.array([1, 2]), [1, 2]),
            ((1, 2), [1, 2]),
            (b"bytes", "bytes"),
        ],
    )
    def test_conversions(self, value, expected):
        assert to_jsonable(value) == expected

    def test_metadata_is_strictly_valid_json(self, single_plane_tif, tmp_path):
        """The header contains MATLAB `inf`, which plain JSON cannot express."""
        recording = read_recording(single_plane_tif)
        metadata = build_metadata(recording)
        # allow_nan=False rejects the non-standard Infinity/NaN literals.
        text = json.dumps(to_jsonable(metadata), allow_nan=False)
        assert '"Infinity"' in text
        assert json.loads(text)["tool"]["nonfinite_encoding"] == "string"

    def test_inf_survives_as_a_labelled_string(self, single_plane_tif):
        metadata = build_metadata(read_recording(single_plane_tif))
        nested = to_jsonable(metadata)["scanimage"]["FrameData"]["SI"]
        assert nested["hBeams"]["lengthConstants"] == "Infinity"


class TestExport:
    def test_full_export_writes_the_expected_tree(self, volumetric_tif, tmp_path):
        recording = read_recording(volumetric_tif)
        result = export_recording(recording, tmp_path)

        directory = tmp_path / "volume__00001"
        assert result.directory == directory
        written = set(result.relative_files())
        assert {"metadata.json", "frames.npy", "aux_triggers/aux0.npy", "manifest.json"} <= written
        assert any(name.startswith("plots/overview") for name in written)

    def test_outputs_load_without_pickle(self, volumetric_tif, tmp_path):
        """`np.load` must work with its safe defaults."""
        export_recording(read_recording(volumetric_tif), tmp_path)
        directory = tmp_path / "volume__00001"

        frames = np.load(directory / "frames.npy", allow_pickle=False)
        assert frames.size == 24
        assert "frame_timestamp_s" in frames.dtype.names

        events = np.load(directory / "aux_triggers" / "aux0.npy", allow_pickle=False)
        assert events["timestamp_s"].size == 1

    def test_manifest_records_sources_counts_and_qc(self, volumetric_tif, tmp_path):
        export_recording(read_recording(volumetric_tif), tmp_path)
        manifest = json.loads((tmp_path / "volume__00001" / "manifest.json").read_text())
        assert manifest["n_pages"] == 24
        assert manifest["aux_event_counts"] == {"aux0": 1}
        assert manifest["n_i2c_packets"] == 0
        assert manifest["source_files"][0].endswith("volume__00001.tif")
        assert manifest["qc"]["ok"] is True

    def test_i2c_products(self, tmp_path):
        path = tmp_path / "i2c__00001.tif"
        write_tif(
            path,
            descriptions_for(4, i2c_events={1: "{{0.5, 'treadmill_9'} {0.6, 'treadmill_10'} }"}),
        )
        export_recording(read_recording(path), tmp_path / "out", decode_i2c=True)
        directory = tmp_path / "out" / "i2c__00001"

        packets = np.load(directory / "i2c" / "packets.npy", allow_pickle=False)
        assert packets.size == 2
        payloads = np.load(directory / "i2c" / "payloads.npy", allow_pickle=False)
        assert payloads.dtype == np.uint8
        raw = json.loads((directory / "i2c" / "packets_raw.json").read_text())
        assert raw[0]["raw"] == "{0.5, 'treadmill_9'}"
        decoded = np.load(directory / "i2c" / "decoded_treadmill.npy", allow_pickle=False)
        assert decoded["value"].tolist() == [9.0, 10.0]

    def test_no_aux_or_i2c_directories_when_there_are_no_events(self, tmp_path):
        path = tmp_path / "quiet__00001.tif"
        write_tif(path, descriptions_for(4))
        export_recording(read_recording(path), tmp_path / "out")
        directory = tmp_path / "out" / "quiet__00001"
        assert not (directory / "aux_triggers").exists()
        assert not (directory / "i2c").exists()

    def test_existing_directory_is_protected(self, single_plane_tif, tmp_path):
        recording = read_recording(single_plane_tif)
        export_recording(recording, tmp_path)
        with pytest.raises(FileExistsError, match="overwrite"):
            export_recording(recording, tmp_path)
        # ... and can be replaced deliberately.
        export_recording(recording, tmp_path, overwrite=True)

    def test_metadata_only_export(self, single_plane_tif, tmp_path):
        result = export_recording(
            read_recording(single_plane_tif),
            tmp_path,
            write_frames=False,
            write_aux=False,
            write_i2c=False,
            make_plot=False,
        )
        assert set(result.relative_files()) == {"metadata.json", "manifest.json"}

    def test_flat_header_layout(self, single_plane_tif, tmp_path):
        export_recording(
            read_recording(single_plane_tif), tmp_path, flat_header=True, make_plot=False
        )
        metadata = json.loads((tmp_path / "plane__00001" / "metadata.json").read_text())
        assert metadata["scanimage"]["framedata_layout"] == "flat"
        assert "SI.hScan2D.name" in metadata["scanimage"]["FrameData"]

    def test_default_location_is_beside_the_source_tif(self, single_plane_tif):
        """Not the working directory: exports belong with the data."""
        result = export_recording(read_recording(single_plane_tif), make_plot=False)
        assert result.directory == single_plane_tif.parent / "plane__00001"
        assert (result.directory / "metadata.json").exists()

    def test_default_location_of_a_merged_acquisition(self, split_acquisition_tifs):
        first, _second = split_acquisition_tifs
        recording = read_recording(first, merge_acquisition=True)
        result = export_recording(recording, make_plot=False)
        assert result.directory == first.parent / "split__00012"


class TestQC:
    def test_clean_file_passes(self, single_plane_tif):
        report = check_recording(read_recording(single_plane_tif))
        assert report.ok
        assert report.stats["n_unique_frame_numbers"] == 20

    def test_frame_number_gap_is_an_error(self, tmp_path):
        path = tmp_path / "gap__00001.tif"
        # Frame 3 is missing: 1, 2, 4, 5 ...
        numbers = [1, 2, 4, 5, 6]
        descriptions = [
            build_page_description(number, index * FRAME_PERIOD_S)
            for index, number in enumerate(numbers)
        ]
        write_tif(path, descriptions)
        report = check_recording(read_recording(path))
        assert not report.ok
        codes = [issue.code for issue in report.errors]
        assert "frame_number_gaps" in codes

    def test_timestamp_jitter_is_detected(self, tmp_path):
        path = tmp_path / "jitter__00001.tif"
        # One frame arrives twice as late as the rest.
        timestamps = [0.0, 0.033, 0.066, 0.132, 0.165]
        descriptions = [
            build_page_description(index + 1, timestamp)
            for index, timestamp in enumerate(timestamps)
        ]
        write_tif(path, descriptions)
        report = check_recording(read_recording(path))
        assert "timestamp_jitter" in [issue.code for issue in report.errors]

    def test_non_monotonic_timestamps_are_an_error(self, tmp_path):
        path = tmp_path / "backwards__00001.tif"
        descriptions = [
            build_page_description(index + 1, timestamp)
            for index, timestamp in enumerate([0.0, 0.033, 0.02, 0.1])
        ]
        write_tif(path, descriptions)
        report = check_recording(read_recording(path))
        assert "timestamp_not_monotonic" in [issue.code for issue in report.errors]

    def test_partial_volume_warns(self, tmp_path):
        path = tmp_path / "partial__00001.tif"
        from conftest import build_header

        header = build_header(
            **{
                "SI.hStackManager.enable": True,
                "SI.hStackManager.actualNumSlices": 3,
                "SI.hStackManager.numFramesPerVolume": 3,
                "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
            }
        )
        # 10 pages is 2.5 volumes of 4 pages.
        write_tif(path, descriptions_for(10), header=header)
        report = check_recording(read_recording(path))
        assert "partial_volume" in [issue.code for issue in report.warnings]
        assert report.ok  # incomplete trailing volume is normal, not fatal

    def test_dc_over_voltage_warns(self, tmp_path):
        path = tmp_path / "overvolt__00001.tif"
        descriptions = descriptions_for(4)
        descriptions[2] = descriptions[2].replace("dcOverVoltage = 0", "dcOverVoltage = 1")
        write_tif(path, descriptions)
        report = check_recording(read_recording(path))
        assert "dc_over_voltage" in [issue.code for issue in report.warnings]

    def test_missing_end_flag_is_informational(self, tmp_path):
        path = tmp_path / "aborted__00001.tif"
        write_tif(path, descriptions_for(6, mark_end=False))
        report = check_recording(read_recording(path))
        assert "no_end_of_acquisition" in [issue.code for issue in report.infos]
        assert report.ok

    def test_report_serialises(self, single_plane_tif):
        report = check_recording(read_recording(single_plane_tif))
        payload = json.dumps(to_jsonable(report.as_dict()), allow_nan=False)
        assert json.loads(payload)["ok"] is True

    def test_volume_rate_is_reported_for_volumetric_data(self, volumetric_tif):
        """Per-cell activity is sampled per volume, so that rate must be stated."""
        report = check_recording(read_recording(volumetric_tif))
        # 4 pages per volume at a 30 Hz plane clock -> 7.5 Hz volumes.
        assert report.stats["median_volume_interval_s"] == pytest.approx(4 * FRAME_PERIOD_S)
        assert report.stats["implied_volume_rate_hz"] == pytest.approx(7.5)
        assert report.stats["implied_frame_rate_hz"] == pytest.approx(30.0)

    def test_volume_rate_equals_frame_rate_for_a_single_plane(self, single_plane_tif):
        report = check_recording(read_recording(single_plane_tif))
        assert report.stats["implied_volume_rate_hz"] == pytest.approx(
            report.stats["implied_frame_rate_hz"]
        )

    def test_truncated_header_losing_trigger_fields_is_an_error(self, tmp_path):
        """A flood on one AUX line can push later fields out of the header.

        ScanImage writes the frame header into a fixed-size buffer in a fixed
        key order, so too many timestamps on an early AUX line silently drop
        `auxTrigger2`, `auxTrigger3` and `I2CData`. Those events are then
        absent from the file altogether, which must not pass quietly.
        """
        path = tmp_path / "truncated__00001.tif"
        descriptions = descriptions_for(6)
        # Simulate the overflow: cut frame 2's header after auxTrigger1.
        descriptions[2] = descriptions[2].split("auxTrigger2")[0]
        write_tif(path, descriptions)

        report = check_recording(read_recording(path))
        assert not report.ok
        issue = next(i for i in report.errors if i.code == "truncated_page_header")
        assert issue.details["n_frames"] == 1
        assert issue.details["first_frames"] == [2]
        assert set(issue.details["lost_keys"]) == {"auxTrigger2", "auxTrigger3", "I2CData"}
        assert report.stats["n_truncated_page_headers"] == 1
        # Truncation explains the differing key sets, so it is not also
        # reported as generic drift.
        assert "page_header_drift" not in [i.code for i in report.issues]

    def test_intact_headers_report_no_truncation(self, single_plane_tif):
        report = check_recording(read_recording(single_plane_tif))
        assert "truncated_page_header" not in [i.code for i in report.issues]
        assert "n_truncated_page_headers" not in report.stats

    def test_volume_rate_mismatch_with_the_header_warns(self, tmp_path):
        from conftest import build_header

        # The header claims 30 Hz volumes while the data delivers 7.5 Hz.
        header = build_header(
            **{
                "SI.hStackManager.enable": True,
                "SI.hStackManager.actualNumSlices": 3,
                "SI.hStackManager.numFramesPerVolume": 3,
                "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                "SI.hRoiManager.scanVolumeRate": 30.0,
            }
        )
        path = tmp_path / "badrate__00001.tif"
        write_tif(path, descriptions_for(24), header=header)
        report = check_recording(read_recording(path))
        assert "volume_rate_mismatch" in [issue.code for issue in report.warnings]


class TestPlots:
    def test_figure_has_three_panels_when_there_are_events(self, single_plane_tif):
        from scanimage_octo_reader.plots import build_overview_figure

        figure = build_overview_figure(read_recording(single_plane_tif))
        try:
            assert len(figure.axes) == 3
        finally:
            figure.clear()

    def test_single_panel_when_nothing_fired(self, tmp_path):
        from scanimage_octo_reader.plots import build_overview_figure

        path = tmp_path / "quiet__00001.tif"
        write_tif(path, descriptions_for(8))
        figure = build_overview_figure(read_recording(path))
        try:
            assert len(figure.axes) == 1
        finally:
            figure.clear()

    def test_dense_triggers_render(self, tmp_path):
        """A trigger on every frame must not blow up or draw a solid block."""
        from scanimage_octo_reader.plots import DENSE_EVENT_THRESHOLD, save_overview_figure

        path = tmp_path / "dense__00001.tif"
        n_pages = DENSE_EVENT_THRESHOLD + 10
        write_tif(path, descriptions_for(n_pages, aux_events={0: list(range(n_pages))}))
        recording = read_recording(path)
        assert recording.aux[0].size > DENSE_EVENT_THRESHOLD
        paths = save_overview_figure(recording, tmp_path / "plots")
        assert all(path.stat().st_size > 0 for path in paths)

    def test_png_and_pdf_by_default(self, single_plane_tif, tmp_path):
        from scanimage_octo_reader.plots import save_overview_figure

        paths = save_overview_figure(read_recording(single_plane_tif), tmp_path)
        assert [path.name for path in paths] == ["overview.png", "overview.pdf"]
        assert all(path.stat().st_size > 0 for path in paths)

    @pytest.mark.parametrize("image_format", ["png", "pdf", "svg"])
    def test_individual_formats(self, single_plane_tif, tmp_path, image_format):
        from scanimage_octo_reader.plots import save_overview_figure

        paths = save_overview_figure(
            read_recording(single_plane_tif), tmp_path, formats=image_format
        )
        assert len(paths) == 1
        assert paths[0].suffix == f".{image_format}"
        assert paths[0].stat().st_size > 0

    def test_pdf_text_stays_editable(self, single_plane_tif, tmp_path):
        """Text must be embedded as TrueType, not converted to outlines.

        matplotlib defaults to Type 3 fonts, which most editors turn into
        paths, so labels cannot be re-typeset in a figure. With
        ``pdf.fonttype = 42`` it embeds the TrueType glyphs instead, which the
        PDF describes as a ``/Type0`` composite font with a
        ``/CIDFontType2`` descendant and a ``/FontFile2`` program - the
        combination checked here. (Verified to discriminate: the same figure
        written with fonttype 3 contains ``/Type3`` and none of these.)
        """
        from scanimage_octo_reader.plots import save_overview_figure

        (path,) = save_overview_figure(read_recording(single_plane_tif), tmp_path, formats="pdf")
        content = path.read_bytes()
        assert b"/CIDFontType2" in content
        assert b"/FontFile2" in content
        assert b"/Type3" not in content

    def test_svg_keeps_text_as_text(self, single_plane_tif, tmp_path):
        from scanimage_octo_reader.plots import save_overview_figure

        (path,) = save_overview_figure(read_recording(single_plane_tif), tmp_path, formats="svg")
        markup = path.read_text()
        assert "acquisition time (s)" in markup

    def test_merged_recording_plots(self, split_acquisition_tifs):
        from scanimage_octo_reader.plots import build_overview_figure

        first, _second = split_acquisition_tifs
        figure = build_overview_figure(read_recording(first, merge_acquisition=True))
        try:
            assert figure.axes
        finally:
            figure.clear()
