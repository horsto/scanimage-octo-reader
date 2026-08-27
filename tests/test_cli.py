"""The ``socto`` command line interface."""

from __future__ import annotations

import json
import re

import pytest
import tifffile
from conftest import FRAME_PERIOD_S, build_page_description, descriptions_for, write_tif
from rich.console import Console
from typer.testing import CliRunner

from scanimage_octo_reader import cli
from scanimage_octo_reader.cli import app

# Rich renders help screens and tables to the width it detects, which under
# pytest comes from the environment and so differs between machines and CI
# runners. At narrow widths it truncates option names and table cells with an
# ellipsis, which broke an assertion on one CI runner while passing everywhere
# else. Two consoles are involved and each needs pinning: typer builds its own
# for help screens (it reads the environment at invoke time, hence `env` here)
# while `cli.console` renders our tables (pinned by the fixture below).
runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1"})

_CONSOLE_WIDTH = 200

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Render the CLI's own output at a fixed width, whatever the terminal is."""
    monkeypatch.setattr(cli, "console", Console(width=_CONSOLE_WIDTH, no_color=True))
    monkeypatch.setattr(
        cli, "error_console", Console(width=_CONSOLE_WIDTH, no_color=True, stderr=True)
    )


def plain(result) -> str:
    """CLI output with any residual styling removed."""
    return _ANSI_ESCAPE.sub("", result.output)


class TestHelp:
    def test_bare_command_shows_help_and_succeeds(self):
        """`socto` alone must be helpful, not an error."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage" in plain(result)
        for command in (
            "info",
            "metadata",
            "triggers",
            "plot",
            "export",
            "pages",
            "project",
            "check",
        ):
            assert command in plain(result)

    def test_subcommand_without_arguments_shows_its_help(self):
        """A subcommand with no files shows its own help.

        The exit code is click's usual usage-error 2 here - unlike a bare
        `socto`, an incomplete command really is a mistake - so this only
        pins down that the user gets the help text rather than a traceback.
        """
        result = runner.invoke(app, ["export"])
        assert "Usage" in plain(result)
        assert "--no-plots" in plain(result)

    def test_missing_file_is_reported(self, tmp_path):
        result = runner.invoke(app, ["info", str(tmp_path / "nope.tif")])
        assert result.exit_code != 0


class TestInfo:
    def test_summarises_without_writing(self, volumetric_tif, tmp_path):
        result = runner.invoke(app, ["info", str(volumetric_tif)])
        assert result.exit_code == 0, plain(result)
        assert "volume__00001" in plain(result)
        assert "Test_Scanner" in plain(result)
        # Nothing may be created next to the input.
        assert list(tmp_path.glob("volume__00001/**")) == []

    def test_reports_triggers(self, single_plane_tif):
        result = runner.invoke(app, ["info", str(single_plane_tif)])
        assert "AUX0" in plain(result)
        assert "2 events" in plain(result)

    def test_handles_several_files(self, single_plane_tif, volumetric_tif):
        result = runner.invoke(app, ["info", str(single_plane_tif), str(volumetric_tif)])
        assert result.exit_code == 0
        assert "plane__00001" in plain(result)
        assert "volume__00001" in plain(result)


class TestExportCommands:
    def test_export_writes_everything(self, volumetric_tif, tmp_path):
        result = runner.invoke(app, ["export", str(volumetric_tif), "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, plain(result)
        directory = tmp_path / "out" / "volume__00001"
        assert (directory / "metadata.json").exists()
        assert (directory / "frames.npy").exists()
        assert (directory / "aux_triggers" / "aux0.npy").exists()
        assert (directory / "plots" / "overview.png").exists()
        assert (directory / "plots" / "overview.pdf").exists()
        assert (directory / "manifest.json").exists()

    def test_export_without_plots(self, single_plane_tif, tmp_path):
        runner.invoke(
            app,
            ["export", str(single_plane_tif), "-o", str(tmp_path / "out"), "--no-plots"],
        )
        assert not (tmp_path / "out" / "plane__00001" / "plots").exists()

    def test_metadata_command_writes_json_only(self, single_plane_tif, tmp_path):
        result = runner.invoke(
            app, ["metadata", str(single_plane_tif), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, plain(result)
        directory = tmp_path / "out" / "plane__00001"
        assert {p.name for p in directory.iterdir()} == {"metadata.json", "manifest.json"}
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["summary"]["n_pages"] == 20

    def test_metadata_without_rois(self, single_plane_tif, tmp_path):
        runner.invoke(
            app,
            ["metadata", str(single_plane_tif), "-o", str(tmp_path / "out"), "--no-rois"],
        )
        metadata = json.loads((tmp_path / "out" / "plane__00001" / "metadata.json").read_text())
        assert "RoiGroups" not in metadata["scanimage"]

    def test_triggers_command_writes_tables_only(self, single_plane_tif, tmp_path):
        result = runner.invoke(
            app, ["triggers", str(single_plane_tif), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, plain(result)
        directory = tmp_path / "out" / "plane__00001"
        assert (directory / "aux_triggers" / "aux0.npy").exists()
        assert (directory / "frames.npy").exists()
        assert not (directory / "metadata.json").exists()

    def test_overwrite_is_required_to_replace_output(self, single_plane_tif, tmp_path):
        arguments = ["export", str(single_plane_tif), "-o", str(tmp_path / "out")]
        assert runner.invoke(app, arguments).exit_code == 0
        assert runner.invoke(app, arguments).exit_code != 0
        assert runner.invoke(app, [*arguments, "--overwrite"]).exit_code == 0

    def test_acquisition_flag_merges_siblings(self, split_acquisition_tifs, tmp_path):
        first, _second = split_acquisition_tifs
        result = runner.invoke(
            app, ["export", str(first), "-o", str(tmp_path / "out"), "--acquisition"]
        )
        assert result.exit_code == 0, plain(result)
        manifest = json.loads((tmp_path / "out" / "split__00012" / "manifest.json").read_text())
        assert manifest["n_pages"] == 20
        assert len(manifest["source_files"]) == 2


class TestPlotCommand:
    def test_writes_png_and_pdf_by_default(self, single_plane_tif, tmp_path):
        result = runner.invoke(app, ["plot", str(single_plane_tif), "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, plain(result)
        plots = tmp_path / "out" / "plane__00001" / "plots"
        assert (plots / "overview.png").exists()
        assert (plots / "overview.pdf").exists()

    def test_a_single_requested_format(self, single_plane_tif, tmp_path):
        result = runner.invoke(
            app, ["plot", str(single_plane_tif), "-o", str(tmp_path / "out"), "-f", "svg"]
        )
        assert result.exit_code == 0, plain(result)
        plots = tmp_path / "out" / "plane__00001" / "plots"
        assert {path.name for path in plots.iterdir()} == {"overview.svg"}

    def test_repeated_format_options(self, single_plane_tif, tmp_path):
        result = runner.invoke(
            app,
            [
                "plot",
                str(single_plane_tif),
                "-o",
                str(tmp_path / "out"),
                "-f",
                "pdf",
                "-f",
                "svg",
            ],
        )
        assert result.exit_code == 0, plain(result)
        plots = tmp_path / "out" / "plane__00001" / "plots"
        assert {path.name for path in plots.iterdir()} == {"overview.pdf", "overview.svg"}

    def test_rejects_an_unknown_format(self, single_plane_tif, tmp_path):
        result = runner.invoke(
            app,
            ["plot", str(single_plane_tif), "-o", str(tmp_path), "--format", "jpeg"],
        )
        assert result.exit_code == 2


class TestProjectCommand:
    def test_writes_one_page_per_volume(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app, ["project", str(valued_volumetric_tif), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, plain(result)
        path = tmp_path / "out" / "vol__00001" / "vol__00001_proj-mean.tif"
        assert path.exists()
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 6
        assert "6 page(s), mean of 3 plane(s)" in plain(result)

    def test_several_methods_and_a_plane_selection(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            [
                "project",
                str(valued_volumetric_tif),
                "-o",
                str(tmp_path / "out"),
                "-m",
                "max",
                "-m",
                "std",
                "--planes",
                "0,1",
            ],
        )
        assert result.exit_code == 0, plain(result)
        directory = tmp_path / "out" / "vol__00001"
        assert {path.name for path in directory.iterdir()} == {
            "vol__00001_proj-max.tif",
            "vol__00001_proj-std.tif",
        }

    def test_rejects_an_unknown_method(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            ["project", str(valued_volumetric_tif), "-o", str(tmp_path), "-m", "median"],
        )
        assert result.exit_code == 2

    def test_rejects_an_unreadable_plane_selection(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            ["project", str(valued_volumetric_tif), "-o", str(tmp_path), "--planes", "top"],
        )
        assert result.exit_code == 2

    def test_a_single_plane_recording_is_an_error(self, single_plane_tif, tmp_path):
        result = runner.invoke(app, ["project", str(single_plane_tif), "-o", str(tmp_path / "out")])
        assert result.exit_code == 1

    def test_overwrite_is_required_to_replace_output(self, valued_volumetric_tif, tmp_path):
        arguments = ["project", str(valued_volumetric_tif), "-o", str(tmp_path / "out")]
        assert runner.invoke(app, arguments).exit_code == 0
        assert runner.invoke(app, arguments).exit_code == 1
        assert runner.invoke(app, [*arguments, "--overwrite"]).exit_code == 0

    def test_repeats_average(self, valued_frame_repeat_tif, tmp_path):
        result = runner.invoke(
            app,
            [
                "project",
                str(valued_frame_repeat_tif),
                "-o",
                str(tmp_path / "out"),
                "-m",
                "max",
                "--repeats",
                "average",
                "--dtype",
                "float32",
            ],
        )
        assert result.exit_code == 0, plain(result)
        path = tmp_path / "out" / "repeats__00001" / "repeats__00001_proj-max.tif"
        with tifffile.TiffFile(path) as tif:
            # max over the two plane averages (1, 4), not the pooled max of 5.
            assert [float(page.asarray()[0, 0]) for page in tif.pages] == [4.0, 11.0]
        # Averaged repeats leave one frame per plane, so the frame count
        # matches the plane count and needs no extra annotation.
        assert "max of 2 plane(s) [0, 1]" in plain(result)

    def test_rejects_an_unknown_repeat_mode(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            ["project", str(valued_volumetric_tif), "-o", str(tmp_path), "--repeats", "median"],
        )
        assert result.exit_code == 2

    def test_a_degenerate_std_is_an_error(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            [
                "project",
                str(valued_volumetric_tif),
                "-o",
                str(tmp_path / "out"),
                "-m",
                "std",
                "--planes",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "at least 2 frames" in plain(result)

    def test_limit_and_dtype(self, valued_volumetric_tif, tmp_path):
        result = runner.invoke(
            app,
            [
                "project",
                str(valued_volumetric_tif),
                "-o",
                str(tmp_path / "out"),
                "--limit",
                "2",
                "--dtype",
                "float32",
            ],
        )
        assert result.exit_code == 0, plain(result)
        path = tmp_path / "out" / "vol__00001" / "vol__00001_proj-mean.tif"
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].dtype == "float32"


class TestPagesCommand:
    def test_table_output(self, single_plane_tif):
        result = runner.invoke(app, ["pages", str(single_plane_tif), "--stop", "3"])
        assert result.exit_code == 0, plain(result)
        assert "frame_number" in plain(result)

    def test_csv_output(self, single_plane_tif):
        result = runner.invoke(app, ["pages", str(single_plane_tif), "--stop", "3", "--csv"])
        assert result.exit_code == 0
        lines = [line for line in plain(result).splitlines() if line.strip()]
        assert lines[0].startswith("page_index,frame_number")
        assert len(lines) == 4  # header + 3 rows


class TestCheckCommand:
    def test_clean_file_exits_zero(self, single_plane_tif):
        result = runner.invoke(app, ["check", str(single_plane_tif)])
        assert result.exit_code == 0
        assert "OK" in plain(result)

    def test_broken_file_exits_nonzero(self, tmp_path):
        path = tmp_path / "gap__00001.tif"
        descriptions = [
            build_page_description(number, index * FRAME_PERIOD_S)
            for index, number in enumerate([1, 2, 6, 7])
        ]
        write_tif(path, descriptions)
        result = runner.invoke(app, ["check", str(path)])
        assert result.exit_code == 1
        assert "FAILED" in plain(result)
        assert "frame_number_gaps" in plain(result)

    def test_quiet_suppresses_informational_lines(self, tmp_path):
        path = tmp_path / "aborted__00001.tif"
        write_tif(path, descriptions_for(6, mark_end=False))
        verbose = runner.invoke(app, ["check", str(path)])
        quiet = runner.invoke(app, ["check", str(path), "--quiet"])
        assert "no_end_of_acquisition" in plain(verbose)
        assert "no_end_of_acquisition" not in plain(quiet)
