"""Split acquisitions: discovering siblings and merging them into one timeline."""

from __future__ import annotations

import pytest
from conftest import FRAME_PERIOD_S, build_header, descriptions_for, write_tif

from scanimage_octo_reader import read_recording
from scanimage_octo_reader.acquisition import find_acquisition_files, natural_sort


class TestFileDiscovery:
    def test_split_siblings_are_found_and_ordered(self, split_acquisition_tifs):
        first, second = split_acquisition_tifs
        assert find_acquisition_files(first) == [first, second]
        # Discovery works from any member of the set.
        assert find_acquisition_files(second) == [first, second]

    def test_single_file_naming_is_not_expanded(self, single_plane_tif):
        """`<base>_<acq>.tif` is a complete acquisition, never part of a set."""
        assert find_acquisition_files(single_plane_tif) == [single_plane_tif]

    def test_siblings_with_different_headers_are_excluded(self, tmp_path):
        """Matching names alone must not be enough to fabricate a timeline."""
        first = tmp_path / "mixed__00003_00001.tif"
        second = tmp_path / "mixed__00003_00002.tif"
        write_tif(first, descriptions_for(4))
        write_tif(
            second,
            descriptions_for(4),
            header=build_header(**{"SI.hRoiManager.scanZoomFactor": 99}),
        )
        assert find_acquisition_files(first) == [first]

    def test_natural_sort_orders_numerically(self):
        assert natural_sort(["f_10.tif", "f_2.tif", "f_1.tif"]) == [
            "f_1.tif",
            "f_2.tif",
            "f_10.tif",
        ]

    def test_file_index_10_sorts_after_9(self, tmp_path):
        paths = []
        for index in (1, 2, 9, 10, 11):
            path = tmp_path / f"many__00001_{index:05d}.tif"
            write_tif(path, descriptions_for(2))
            paths.append(path)
        assert find_acquisition_files(paths[0]) == paths


class TestMerging:
    def test_merge_is_off_by_default(self, split_acquisition_tifs):
        first, _second = split_acquisition_tifs
        recording = read_recording(first)
        assert recording.n_pages == 10
        assert len(recording.paths) == 1

    def test_merged_timeline_is_continuous(self, split_acquisition_tifs):
        first, second = split_acquisition_tifs
        recording = read_recording(first, merge_acquisition=True)

        assert recording.paths == [first, second]
        assert recording.n_pages == 20
        assert recording.pages_per_file == [10, 10]
        # ScanImage numbers split files continuously, so no renumbering happens.
        assert recording.frames["frame_number"].tolist() == list(range(1, 21))
        # Page indices are offset so they stay unique across the merge.
        assert recording.frames["page_index"].tolist() == list(range(20))
        assert recording.frames["file_index"].tolist() == [0] * 10 + [1] * 10

    def test_merged_timestamps_increase_across_the_boundary(self, split_acquisition_tifs):
        first, _second = split_acquisition_tifs
        timestamps = read_recording(first, merge_acquisition=True).frames["frame_timestamp_s"]
        assert timestamps[10] == pytest.approx(10 * FRAME_PERIOD_S)
        assert all(b > a for a, b in zip(timestamps[:-1], timestamps[1:]))

    def test_events_from_both_files_are_collected(self, split_acquisition_tifs):
        first, _second = split_acquisition_tifs
        events = read_recording(first, merge_acquisition=True).aux[0]
        assert events.size == 2
        assert events["page_index"].tolist() == [2, 14]
        assert events["file_index"].tolist() == [0, 1]

    def test_merged_name_drops_the_file_index(self, split_acquisition_tifs):
        first, _second = split_acquisition_tifs
        recording = read_recording(first, merge_acquisition=True)
        assert recording.name == "split__00012"

    def test_explicit_file_list_is_used_in_order(self, split_acquisition_tifs):
        first, second = split_acquisition_tifs
        recording = read_recording([first, second])
        assert recording.n_pages == 20
        assert recording.paths == [first, second]

    def test_truncated_header_is_located_in_the_right_file(self, tmp_path):
        """Key sets are numbered per file, so merging must remap them.

        Without the remap a truncated header in the second file would be
        attributed to the wrong frames, or missed entirely.
        """
        from scanimage_octo_reader import check_recording

        first = tmp_path / "trunc__00007_00001.tif"
        second = tmp_path / "trunc__00007_00002.tif"
        write_tif(first, descriptions_for(5, mark_end=False))

        tail = descriptions_for(5, first_frame_number=6, first_timestamp_s=5 * FRAME_PERIOD_S)
        # Frame 3 of the second file loses everything after auxTrigger1.
        tail[3] = tail[3].split("auxTrigger2")[0]
        write_tif(second, tail)

        recording = read_recording(first, merge_acquisition=True)
        assert recording.n_pages == 10
        assert recording.sweep.key_set_ids.size == 10

        report = check_recording(recording)
        issue = next(i for i in report.errors if i.code == "truncated_page_header")
        # Page 8 overall: 5 pages from the first file, then index 3.
        assert issue.details["first_frames"] == [8]
        assert issue.details["n_frames"] == 1

    def test_mismatched_explicit_list_warns(self, tmp_path):
        first = tmp_path / "a__00001_00001.tif"
        second = tmp_path / "b__00002_00001.tif"
        write_tif(first, descriptions_for(4))
        write_tif(
            second,
            descriptions_for(4),
            header=build_header(**{"SI.hRoiManager.scanZoomFactor": 42}),
        )
        recording = read_recording([first, second])
        assert any("different ScanImage header" in w for w in recording.warnings)
