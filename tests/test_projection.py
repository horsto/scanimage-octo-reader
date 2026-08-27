"""Per-volume plane projections.

Every fixture used here fills page N with the value N (see
`conftest.page_value_frames`), so each assertion is a hand-computable number:
a page taken from the wrong slice, the wrong channel or from flyback shows up
as a wrong value rather than as a plausible-looking image.

`valued_volumetric_tif` is 24 pages = 6 volumes x (3 slices + 1 flyback), one
channel, so volume ``v`` holds slices at pages ``4v, 4v+1, 4v+2`` and flyback
at ``4v+3``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from scanimage_octo_reader import read_recording
from scanimage_octo_reader.projection import (
    ProjectionResult,
    project_recording,
    volume_page_offsets,
)


def pages_of(path) -> np.ndarray:
    """Every page of `path` as a ``(n_pages, height, width)`` array."""
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


def values_of(path) -> np.ndarray:
    """The (uniform) value of every page of `path`, as a 1-D array.

    The fixtures fill each page with a single value, so a projection page is
    uniform too; collapsing it to one number per page keeps the expectations
    readable.
    """
    stack = pages_of(path)
    assert (stack == stack[:, :1, :1]).all(), "a projection page is not uniform"
    return stack[:, 0, 0]


def only(results: list[ProjectionResult]) -> ProjectionResult:
    assert len(results) == 1, [result.path.name for result in results]
    return results[0]


class TestProjectionValues:
    def test_mean_across_planes_excludes_flyback(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))

        # mean(4v, 4v+1, 4v+2) = 4v+1. Including the 4v+3 flyback page would
        # give 4v+1.5, and averaging whole volumes would give something else
        # again - the values pin down exactly which pages were read.
        assert result.n_pages == 6
        assert list(values_of(result.path)) == [1, 5, 9, 13, 17, 21]

    def test_max_and_std(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        results = project_recording(recording, tmp_path / "out", methods=["max", "std"])
        by_method = {result.method: result for result in results}

        assert list(values_of(by_method["max"].path)) == [2, 6, 10, 14, 18, 22]
        expected_std = float(np.std([0.0, 1.0, 2.0]))
        assert values_of(by_method["std"].path) == pytest.approx(expected_std)

    def test_plane_subset(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(
            project_recording(recording, tmp_path / "out", methods=["max"], planes=[0, 1])
        )
        # Without plane 2 the maximum of volume v is 4v+1, not 4v+2.
        assert list(values_of(result.path)) == [1, 5, 9, 13, 17, 21]
        assert result.planes == [0, 1]

    def test_single_plane_selection_is_a_plain_extraction(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"], planes=[2]))
        assert list(values_of(result.path)) == [2, 6, 10, 14, 18, 22]

    def test_every_method_writes_its_own_file(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        results = project_recording(recording, tmp_path / "out", methods=["mean", "max", "std"])
        directory = tmp_path / "out" / "vol__00001"
        assert {path.name for path in directory.iterdir()} == {
            "vol__00001_proj-mean.tif",
            "vol__00001_proj-max.tif",
            "vol__00001_proj-std.tif",
        }
        assert {result.method for result in results} == {"mean", "max", "std"}

    def test_repeated_methods_are_written_once(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        results = project_recording(recording, tmp_path / "out", methods=["mean", "mean"])
        assert len(results) == 1


class TestFrameRepeats:
    """``framesPerSlice > 1``: several frames per Z position within a volume.

    `valued_frame_repeat_tif` is 2 volumes of (2 planes x 3 frames + 1
    flyback), so volume 0's real pages are 0-5 (plane 0 = 0,1,2; plane 1 =
    3,4,5) and page 6 is flyback.
    """

    def test_repeats_are_pooled_in_with_the_planes(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        assert recording.geometry.frames_per_slice == 3

        result = only(
            project_recording(recording, tmp_path / "out", methods=["mean"], dtype="float32")
        )
        # One page per *volume*, not per frame-repeat: the repeats collapse.
        assert result.n_pages == 2
        assert list(values_of(result.path)) == [2.5, 9.5]

    def test_pooling_does_not_over_weight_any_plane(self, valued_frame_repeat_tif, tmp_path):
        """Every plane contributes equally many repeats, so a pooled mean is
        exactly the mean of the per-plane temporal means, and a pooled max the
        max of the per-plane maxima.

        This is about *grouping* only. It does not make `max` insensitive to
        the repeats: max is a biased statistic whose expectation grows with
        the number of samples pooled, so a `framesPerSlice > 1` recording
        yields a brighter max projection than the same scene with one frame
        per plane. See the README - this test pins the arithmetic, not the
        statistics.
        """
        recording = read_recording(valued_frame_repeat_tif)
        results = project_recording(
            recording, tmp_path / "out", methods=["mean", "max"], dtype="float32"
        )
        by_method = {result.method: result for result in results}

        # Volume 0 holds pages 0-5, volume 1 pages 7-12 (6 and 13 are flyback).
        volume_pages = [np.arange(6.0), np.arange(7.0, 13.0)]
        for volume, values in enumerate(volume_pages):
            plane_0, plane_1 = values[:3], values[3:]
            expected_mean = np.mean([plane_0.mean(), plane_1.mean()])
            assert values_of(by_method["mean"].path)[volume] == pytest.approx(expected_mean)
            assert values_of(by_method["max"].path)[volume] == pytest.approx(
                max(plane_0.max(), plane_1.max())
            )

    def test_a_plane_subset_takes_all_of_that_plane_s_repeats(
        self, valued_frame_repeat_tif, tmp_path
    ):
        recording = read_recording(valued_frame_repeat_tif)
        result = only(
            project_recording(
                recording, tmp_path / "out", methods=["mean"], planes=[0], dtype="float32"
            )
        )
        # Plane 0 alone is pages 0,1,2 of volume 0 and 7,8,9 of volume 1.
        assert list(values_of(result.path)) == [1.0, 8.0]


class TestRepeatModes:
    """`repeats='average'`: collapse each plane's repeats before projecting.

    On `valued_frame_repeat_tif`, volume 0's planes are [0,1,2] and [3,4,5],
    so averaging first leaves the method a stack of [1, 4]; volume 1's planes
    are [7,8,9] and [10,11,12], leaving [8, 11].
    """

    def test_mean_is_unchanged(self, valued_frame_repeat_tif, tmp_path):
        """Equal repeat counts per plane, so averaging first changes nothing."""
        recording = read_recording(valued_frame_repeat_tif)
        pooled = only(
            project_recording(recording, tmp_path / "pool", dtype="float32", repeats="pool")
        )
        averaged = only(
            project_recording(recording, tmp_path / "avg", dtype="float32", repeats="average")
        )
        assert list(values_of(averaged.path)) == list(values_of(pooled.path)) == [2.5, 9.5]

    def test_max_is_taken_over_the_averaged_planes(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        result = only(
            project_recording(
                recording,
                tmp_path / "out",
                methods=["max"],
                dtype="float32",
                repeats="average",
            )
        )
        # max(1, 4) and max(8, 11), not the pooled max(0..5)=5 and
        # max(7..12)=12 - the noise-inflated values this mode avoids.
        assert list(values_of(result.path)) == [4.0, 11.0]

    def test_std_becomes_a_between_plane_measure(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        result = only(
            project_recording(recording, tmp_path / "out", methods=["std"], repeats="average")
        )
        # std([1, 4]) = 1.5: the spread between plane averages, rather than
        # the pooled std([0..5]) = 1.71 that mixes in within-plane spread.
        assert values_of(result.path) == pytest.approx(1.5)

    def test_the_mode_is_recorded_in_the_provenance(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        result = only(project_recording(recording, tmp_path / "out", repeats="average"))
        assert result.repeats == "average"
        assert result.n_reduced_frames == 2  # one averaged frame per plane
        with tifffile.TiffFile(result.path) as tif:
            projection = json.loads(tif.pages[0].description)["projection"]
        assert projection["repeats"] == "average"
        assert projection["n_reduced_frames"] == 2
        assert projection["frames_per_slice"] == 3

    def test_it_is_a_no_op_without_frame_repeats(self, valued_volumetric_tif, tmp_path):
        """With one frame per Z position there is nothing to average."""
        recording = read_recording(valued_volumetric_tif)
        pooled = project_recording(
            recording, tmp_path / "pool", methods=["max", "std"], repeats="pool"
        )
        averaged = project_recording(
            recording, tmp_path / "avg", methods=["max", "std"], repeats="average"
        )
        for one, other in zip(pooled, averaged):
            assert np.array_equal(pages_of(one.path), pages_of(other.path))
            # Reported as pool, because that is the reduction performed.
            assert other.repeats == "pool"
            assert other.n_reduced_frames == 3

    def test_an_unknown_mode_is_refused(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        with pytest.raises(ValueError, match="unknown repeat mode"):
            project_recording(recording, tmp_path / "out", repeats="median")


class TestDegenerateStd:
    """A std over fewer than 2 frames is identically zero, so it is refused."""

    def test_single_plane_with_averaged_repeats(self, valued_frame_repeat_tif, tmp_path):
        recording = read_recording(valued_frame_repeat_tif)
        with pytest.raises(ValueError, match="at least 2 frames"):
            project_recording(
                recording,
                tmp_path / "out",
                methods=["std"],
                planes=[0],
                repeats="average",
            )

    def test_pooling_the_repeats_makes_the_same_selection_valid(
        self, valued_frame_repeat_tif, tmp_path
    ):
        """The 3 repeats are 3 real samples, so a pooled std is well defined."""
        recording = read_recording(valued_frame_repeat_tif)
        result = only(
            project_recording(
                recording, tmp_path / "out", methods=["std"], planes=[0], repeats="pool"
            )
        )
        assert values_of(result.path)[0] == pytest.approx(np.std([0.0, 1.0, 2.0]))

    def test_single_plane_without_repeats(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        with pytest.raises(ValueError, match="at least 2 frames"):
            project_recording(recording, tmp_path / "out", methods=["std"], planes=[1])

    def test_other_methods_are_unaffected(self, valued_volumetric_tif, tmp_path):
        """Only std is degenerate on one frame; mean and max still mean something."""
        recording = read_recording(valued_volumetric_tif)
        results = project_recording(
            recording, tmp_path / "out", methods=["mean", "max"], planes=[1]
        )
        assert len(results) == 2


class TestChannels:
    def test_one_file_per_saved_channel(self, valued_volumetric_two_channel_tif, tmp_path):
        recording = read_recording(valued_volumetric_two_channel_tif)
        results = project_recording(recording, tmp_path / "out", methods=["mean"])
        by_channel = {result.channel: result for result in results}

        assert set(by_channel) == {1, 2}
        # Channels vary fastest: volume v holds channel 1 at pages 8v+0,2,4
        # and channel 2 at 8v+1,3,5 - means of 8v+2 and 8v+3.
        assert list(values_of(by_channel[1].path)) == [2, 10, 18, 26, 34, 42]
        assert list(values_of(by_channel[2].path)) == [3, 11, 19, 27, 35, 43]
        assert by_channel[2].path.name == "volchan__00001_proj-mean_ch2.tif"

    def test_a_single_requested_channel(self, valued_volumetric_two_channel_tif, tmp_path):
        recording = read_recording(valued_volumetric_two_channel_tif)
        result = only(
            project_recording(recording, tmp_path / "out", methods=["mean"], channels=[2])
        )
        assert result.channel == 2
        assert list(values_of(result.path)) == [3, 11, 19, 27, 35, 43]

    def test_single_channel_recordings_get_no_suffix(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))
        assert result.channel is None
        assert result.path.name == "vol__00001_proj-mean.tif"

    def test_a_channel_that_was_not_saved_is_refused(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        with pytest.raises(ValueError, match="not saved"):
            project_recording(recording, tmp_path / "out", channels=[3])


class TestDtypes:
    def test_auto_keeps_the_source_dtype_for_mean_and_max(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        results = project_recording(recording, tmp_path / "out", methods=["mean", "max"])
        assert {result.dtype for result in results} == {"int16"}
        assert all(pages_of(result.path).dtype == np.int16 for result in results)

    def test_auto_uses_float32_for_std(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["std"]))
        # An std of 0.82 rounded onto the source's integer grid would be 1.
        assert result.dtype == "float32"
        assert pages_of(result.path).dtype == np.float32

    def test_explicit_float32(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(
            project_recording(
                recording, tmp_path / "out", methods=["mean"], planes=[0, 1], dtype="float32"
            )
        )
        # mean(4v, 4v+1) = 4v+0.5, which only float output can hold.
        assert list(values_of(result.path)) == [0.5, 4.5, 8.5, 12.5, 16.5, 20.5]

    def test_an_unknown_dtype_is_refused(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        with pytest.raises(ValueError, match="unknown output dtype"):
            project_recording(recording, tmp_path / "out", dtype="int7")


class TestVolumeSelection:
    def test_partial_trailing_volume_is_skipped(self, partial_volume_tif, tmp_path):
        recording = read_recording(partial_volume_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))
        # 22 pages = 5 whole volumes of 4 pages, plus 2 leftover pages.
        assert result.n_pages == 5
        assert list(values_of(result.path)) == [1, 5, 9, 13, 17]
        assert any("partial volume" in warning for warning in result.warnings)

    def test_limit_caps_the_output(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"], limit=2))
        assert result.n_pages == 2
        assert list(values_of(result.path)) == [1, 5]
        assert result.skipped_volumes == 4

    def test_a_limit_beyond_the_recording_is_harmless(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"], limit=100))
        assert result.n_pages == 6
        assert result.skipped_volumes == 0

    def test_progress_reports_every_volume(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        seen: list[int] = []
        project_recording(recording, tmp_path / "out", progress=seen.append)
        assert seen == [1, 2, 3, 4, 5, 6]


class TestRefusals:
    def test_a_single_plane_recording_has_nothing_to_project(self, single_plane_tif, tmp_path):
        recording = read_recording(single_plane_tif)
        with pytest.raises(ValueError, match="not a multi-plane recording"):
            project_recording(recording, tmp_path / "out")

    def test_a_plane_outside_the_stack(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        with pytest.raises(ValueError, match="outside 0-2"):
            project_recording(recording, tmp_path / "out", planes=[0, 7])

    def test_an_unknown_method(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        with pytest.raises(ValueError, match="unknown projection method"):
            project_recording(recording, tmp_path / "out", methods=["median"])

    def test_overwrite_is_required_to_replace_an_output(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        project_recording(recording, tmp_path / "out")
        with pytest.raises(FileExistsError):
            project_recording(recording, tmp_path / "out")
        project_recording(recording, tmp_path / "out", overwrite=True)


class TestOutputFile:
    def test_defaults_to_the_source_directory(self, valued_volumetric_tif):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording))
        assert result.path == valued_volumetric_tif.parent / "vol__00001" / (
            "vol__00001_proj-mean.tif"
        )
        assert result.path.exists()

    def test_provenance_is_written_into_the_first_page(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))

        with tifffile.TiffFile(result.path) as tif:
            description = json.loads(tif.pages[0].description)
            software = tif.pages[0].tags["Software"].value
            # Only the first page carries the metadata; the rest are plain
            # pages of a contiguous series.
            assert not tif.pages[1].description

        assert description["projection"]["method"] == "mean"
        assert description["projection"]["planes"] == [0, 1, 2]
        assert description["projection"]["plane_zs"] == [10.0, 20.0, 30.0]
        assert description["projection"]["flyback_pages_excluded"] == 1
        assert description["projection"]["volume_rate_hz"] == 7.5
        assert description["source"]["files"] == [str(valued_volumetric_tif.resolve())]
        # The source's ScanImage header travels with the projection, with the
        # plane layout rewritten to describe it (see the test below).
        assert "SI.hScan2D.name = 'Test_Scanner'" in software
        assert "SI.hStackManager.actualNumSlices = 1" in software
        assert "SI.hStackManager.actualNumSlices = 3" not in software

    def test_reopens_as_a_flat_single_plane_timeseries(self, valued_volumetric_tif, tmp_path):
        """The copied ScanImage header must not re-fold the output into volumes.

        Readers derive the page layout from the ``SI.*`` header, gating on
        ``hStackManager.enable`` / ``hFastZ.enable`` and ``actualNumSlices``
        (`geometry` here, and `napari-tiff` the same way). Handing on the
        source header verbatim made a projection reappear as a 3-plane
        volumetric stack of a third the length when reopened, so the layout
        fields are rewritten - see `projected_software_tag`.
        """
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))

        reopened = read_recording(result.path)
        assert reopened.n_pages == 6
        assert reopened.geometry.volumetric is False
        assert reopened.geometry.n_slices == 1
        assert reopened.geometry.n_channels == 1
        assert reopened.geometry.pages_per_volume == 1
        assert reopened.geometry.flyback_frames == 0
        # One page is one volume now, so the page rate is the volume rate.
        assert reopened.geometry.frame_rate_hz == pytest.approx(7.5)
        # Everything not about the layout is the source's own header.
        assert reopened.header.get("SI.hRoiManager.scanZoomFactor") == 3
        assert reopened.header.get("SI.hScan2D.name") == "Test_Scanner"
        # The projection sits at no single depth: the mean of 10, 20, 30.
        assert reopened.header.get("SI.hStackManager.zs") == pytest.approx(20.0)

    def test_each_channel_output_claims_only_its_own_channel(
        self, valued_volumetric_two_channel_tif, tmp_path
    ):
        recording = read_recording(valued_volumetric_two_channel_tif)
        results = project_recording(recording, tmp_path / "out", methods=["mean"])
        for result in results:
            reopened = read_recording(result.path)
            # Left at the source's [1, 2], a reader would split one channel
            # of pages into two.
            assert reopened.geometry.channels_saved == [result.channel]
            assert reopened.geometry.n_channels == 1
            assert reopened.n_pages == 6

    def test_pages_form_one_series(self, valued_volumetric_tif, tmp_path):
        recording = read_recording(valued_volumetric_tif)
        result = only(project_recording(recording, tmp_path / "out"))
        with tifffile.TiffFile(result.path) as tif:
            assert tif.series[0].shape == (6, 4, 4)
            assert not tif.is_bigtiff


class TestSplitAcquisitions:
    def test_pages_are_read_across_merged_files(self, tmp_path):
        from conftest import (
            VOLUMETRIC_HEADER_OVERRIDES,
            build_header,
            descriptions_for,
            page_value_frames,
            write_tif,
        )

        header = build_header(**VOLUMETRIC_HEADER_OVERRIDES)
        frames = page_value_frames(16)
        write_tif(
            tmp_path / "merged__00003_00001.tif",
            descriptions_for(8, mark_end=False),
            header=header,
            frames=frames[:8],
        )
        write_tif(
            tmp_path / "merged__00003_00002.tif",
            descriptions_for(8, first_frame_number=9),
            header=header,
            frames=frames[8:],
        )

        recording = read_recording(tmp_path / "merged__00003_00001.tif", merge_acquisition=True)
        assert recording.pages_per_file == [8, 8]
        result = only(project_recording(recording, tmp_path / "out", methods=["mean"]))
        # Page values run 0-15 across the two files, so the second file's
        # volumes must project to 9 and 13.
        assert list(values_of(result.path)) == [1, 5, 9, 13]


class TestPageOffsets:
    @pytest.mark.parametrize(
        "fixture",
        [
            "valued_volumetric_tif",
            "valued_volumetric_two_channel_tif",
            "valued_frame_repeat_tif",
        ],
    )
    def test_offsets_agree_with_the_page_map(self, fixture, request):
        """`volume_page_offsets` must select exactly what `page_map` says it does.

        The offsets exploit the fact that a volume's page layout is periodic;
        `geometry.page_map` derives the same assignment per page. Any drift
        between the two would silently project the wrong pages.
        """
        recording = read_recording(request.getfixturevalue(fixture))
        geometry = recording.geometry
        page_map = geometry.page_map(recording.n_pages)

        for position, channel in enumerate(geometry.channels_saved):
            offsets = volume_page_offsets(geometry, range(geometry.n_slices), position)
            expected = np.flatnonzero(
                (page_map.volume_index == 0) & (page_map.channel == position) & ~page_map.is_flyback
            )
            # One row per Z position, holding that plane's repeated frames.
            assert offsets.shape == (geometry.n_slices, geometry.frames_per_slice), channel
            assert list(offsets.ravel()) == list(expected), channel
            # Rows really are planes: page_map must agree on every entry.
            for plane, row in enumerate(offsets):
                assert list(page_map.slice_index[row]) == [plane] * len(row), channel
