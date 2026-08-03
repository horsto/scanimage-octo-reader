"""The global header and the page-layout rules derived from it."""

from __future__ import annotations

import tifffile

from scanimage_octo_reader.geometry import compute_geometry
from scanimage_octo_reader.header import ScanImageHeader, nest_dotted_keys, read_header


class TestNesting:
    def test_dotted_keys_become_a_tree(self):
        nested = nest_dotted_keys({"SI.hScan2D.name": "x", "SI.hScan2D.zoom": 3, "SI.top": 1})
        assert nested == {"SI": {"hScan2D": {"name": "x", "zoom": 3}, "top": 1}}

    def test_leaf_and_branch_collision_keeps_both(self):
        """ScanImage's key space is not strictly a tree; nothing may be dropped."""
        nested = nest_dotted_keys({"SI.a": 1, "SI.a.b": 2})
        assert nested["SI"]["a"] == {"_value": 1, "b": 2}

    def test_branch_then_leaf_collision_keeps_both(self):
        nested = nest_dotted_keys({"SI.a.b": 2, "SI.a": 1})
        assert nested["SI"]["a"] == {"b": 2, "_value": 1}


class TestHeaderLookup:
    def test_get_accepts_keys_with_and_without_prefix(self):
        header = ScanImageHeader(frame_data={"SI.hScan2D.name": "scanner"})
        assert header.get("SI.hScan2D.name") == "scanner"
        assert header.get("hScan2D.name") == "scanner"
        assert header.get("missing", "default") == "default"

    def test_si_version_string(self):
        header = ScanImageHeader(
            frame_data={
                "SI.VERSION_MAJOR": 2022,
                "SI.VERSION_MINOR": 1,
                "SI.VERSION_UPDATE": 0,
            }
        )
        assert header.si_version == "2022.1.0"

    def test_reads_from_the_software_tag_when_there_is_no_bigtiff_header(self, single_plane_tif):
        """Classic (non-BigTIFF) ScanImage files only carry `SI.*` in `Software`."""
        with tifffile.TiffFile(single_plane_tif) as tif:
            # No BigTIFF metadata header here, so `FrameData` has to come from
            # the `Software` tag fallback.
            assert not (tif.scanimage_metadata or {}).get("FrameData")
            header = read_header(tif)
        assert header.get("SI.hScan2D.name") == "Test_Scanner"
        assert header.si_version == "2022.1.0"


def header_with(**overrides) -> ScanImageHeader:
    base = {
        "SI.hChannels.channelSave": 1,
        "SI.hStackManager.enable": False,
        "SI.hFastZ.enable": False,
        "SI.hStackManager.numSlices": 11,
    }
    base.update(overrides)
    return ScanImageHeader(frame_data=base)


class TestGeometry:
    def test_single_plane_is_flat_without_a_warning(self):
        geometry = compute_geometry(header_with(), n_pages=10)
        assert geometry.volumetric is False
        assert geometry.pages_per_volume == 1
        assert geometry.n_slices == 1
        # A plain timeseries is not a fallback, so nothing to warn about.
        assert geometry.warning is None

    def test_stale_numslices_is_ignored(self):
        """The sample data has numSlices=11 but actualNumSlices=3."""
        geometry = compute_geometry(
            header_with(
                **{
                    "SI.hStackManager.enable": True,
                    "SI.hStackManager.actualNumSlices": 3,
                    "SI.hStackManager.numFramesPerVolume": 3,
                    "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                }
            ),
            n_pages=20000,
        )
        assert geometry.n_slices == 3
        assert geometry.flyback_frames == 1
        assert geometry.pages_per_volume == 4
        assert geometry.n_volumes(20000) == 5000

    def test_channels_are_the_fastest_axis(self):
        geometry = compute_geometry(header_with(**{"SI.hChannels.channelSave": [1, 2]}), n_pages=8)
        assert geometry.n_channels == 2
        page_map = geometry.page_map(8)
        assert page_map.channel.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_page_map_marks_flyback_and_slices(self):
        geometry = compute_geometry(
            header_with(
                **{
                    "SI.hStackManager.enable": True,
                    "SI.hStackManager.actualNumSlices": 3,
                    "SI.hStackManager.numFramesPerVolume": 3,
                    "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                }
            ),
            n_pages=8,
        )
        page_map = geometry.page_map(8)
        assert page_map.slice_index.tolist() == [0, 1, 2, -1, 0, 1, 2, -1]
        assert page_map.is_flyback.tolist() == [False, False, False, True] * 2
        assert page_map.volume_index.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_frames_per_slice_groups_repeats_inside_a_slice(self):
        """On disk the order is Z-outer, frame-repeat-inner."""
        geometry = compute_geometry(
            header_with(
                **{
                    "SI.hStackManager.enable": True,
                    "SI.hStackManager.actualNumSlices": 2,
                    "SI.hStackManager.framesPerSlice": 2,
                    "SI.hStackManager.numFramesPerVolume": 4,
                    "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                }
            ),
            n_pages=8,
        )
        page_map = geometry.page_map(8)
        assert page_map.slice_index.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
        assert page_map.frame_repeat_index.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_inconsistent_slice_counts_fall_back_with_an_explanation(self):
        geometry = compute_geometry(
            header_with(
                **{
                    "SI.hStackManager.enable": True,
                    "SI.hStackManager.actualNumSlices": 3,
                    "SI.hStackManager.framesPerSlice": 1,
                    # 3 x 1 != 5, so the layout cannot be trusted.
                    "SI.hStackManager.numFramesPerVolume": 5,
                    "SI.hStackManager.numFramesPerVolumeWithFlyback": 6,
                }
            ),
            n_pages=12,
        )
        assert geometry.volumetric is False
        assert "does not match" in geometry.warning

    def test_missing_slice_fields_fall_back(self):
        geometry = compute_geometry(header_with(**{"SI.hStackManager.enable": True}), n_pages=12)
        assert geometry.volumetric is False
        assert "slice-count fields are missing" in geometry.warning

    def test_stride_larger_than_the_file_falls_back(self):
        geometry = compute_geometry(
            header_with(
                **{
                    "SI.hStackManager.enable": True,
                    "SI.hStackManager.actualNumSlices": 30,
                    "SI.hStackManager.numFramesPerVolume": 30,
                    "SI.hStackManager.numFramesPerVolumeWithFlyback": 31,
                }
            ),
            n_pages=10,
        )
        assert geometry.volumetric is False
        assert "only has 10 page(s)" in geometry.warning
