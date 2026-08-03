"""The ``socto`` command line interface.

Every subcommand accepts one or more files (shell globs work, since the shell
expands them). Bare ``socto``, or any subcommand without arguments, prints
help rather than an error - see `no_args_is_help`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from scanimage_octo_reader.acquisition import Recording, read_recording
from scanimage_octo_reader.qc import check_recording

app = typer.Typer(
    name="socto",
    help=(
        "Inspect ScanImage TIFF timeseries: export structured metadata, unpack AUX trigger "
        "and I2C data, plot the timeline, and check for dropped frames."
    ),
    # Handled in `_configure` instead, so that a bare `socto` prints help and
    # exits 0 rather than with click's usage-error code.
    no_args_is_help=False,
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)

# Shared option types, so every subcommand spells these the same way.
FilesArgument = Annotated[
    list[Path],
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        show_default=False,
        help="One or more ScanImage TIFF files.",
    ),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out",
        "-o",
        file_okay=False,
        show_default=False,
        help=(
            "Directory to write into; each recording gets its own subdirectory. "
            "Defaults to the directory holding the TIFF being processed."
        ),
    ),
]
AcquisitionOption = Annotated[
    bool,
    typer.Option(
        "--acquisition",
        "-a",
        help=(
            "Merge the sibling files of a split acquisition "
            "(<base>_<acq>_<index>.tif) into one continuous timeline."
        ),
    ),
]
OverwriteOption = Annotated[
    bool, typer.Option("--overwrite", help="Replace the contents of an existing output directory.")
]
QuietOption = Annotated[bool, typer.Option("--quiet", "-q", help="Only report errors.")]
FormatOption = Annotated[
    list[str] | None,
    typer.Option(
        "--format",
        "-f",
        show_default=False,
        help=(
            "Figure format(s): png, pdf or svg. Repeatable. Defaults to png and pdf; "
            "vector output keeps text editable."
        ),
    ),
]

_SUPPORTED_FORMATS = ("png", "pdf", "svg")


def _validated_formats(formats: list[str] | None) -> tuple[str, ...]:
    """Validate the requested figure formats, or fall back to the defaults."""
    from scanimage_octo_reader.plots import DEFAULT_FORMATS

    if not formats:
        return tuple(DEFAULT_FORMATS)
    unsupported = [item for item in formats if item not in _SUPPORTED_FORMATS]
    if unsupported:
        error_console.print(
            f"[red]error[/red] unsupported figure format(s): {', '.join(unsupported)}; "
            f"choose from {', '.join(_SUPPORTED_FORMATS)}"
        )
        raise typer.Exit(code=2)
    # Preserve the order given, minus duplicates.
    return tuple(dict.fromkeys(formats))


@app.callback(invoke_without_command=True)
def _configure(
    context: typer.Context,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show informational log messages.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if context.invoked_subcommand is None:
        # `socto` on its own is a request for orientation, not a mistake.
        console.print(context.get_help())
        raise typer.Exit()


def _load(
    files: list[Path], merge: bool, quiet: bool, show_progress: bool = True
) -> list[Recording]:
    """Read every input, sweeping page headers, with a progress display.

    With `merge`, each *given* file is expanded to its acquisition siblings,
    and files already covered by an earlier expansion are skipped so that
    passing a whole directory of split files does not read them repeatedly.
    """
    recordings: list[Recording] = []
    already_read: set[Path] = set()

    for path in files:
        if path.resolve() in already_read:
            continue
        if quiet or not show_progress:
            recording = read_recording(path, merge_acquisition=merge)
        else:
            with console.status(f"sweeping page headers of {path.name}...", spinner="dots"):
                recording = read_recording(path, merge_acquisition=merge)
        already_read.update(source.resolve() for source in recording.paths)
        recordings.append(recording)

        for warning in recording.warnings:
            error_console.print(f"[yellow]warning[/yellow] {path.name}: {warning}")
    return recordings


def _print_summary_table(recording: Recording) -> None:
    summary = recording.summary()
    triggers = recording.trigger_summary()

    table = Table(
        title=recording.name,
        title_style="bold",
        show_header=False,
        box=None,
        pad_edge=False,
    )
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value")

    def row(label: str, value: object) -> None:
        table.add_row(label, "-" if value is None else str(value))

    row("files", ", ".join(path.name for path in recording.paths))
    row("ScanImage", summary["si_version"])
    row("scanner", f"{summary['scanner_name']} ({summary['scanner_type']})")
    row("epoch", summary["epoch"])
    row("pages", summary["n_pages"])
    row("frame size", f"{summary['image_shape']} {summary['dtype']}")
    row("channels", f"{summary['n_channels']} {summary['channels_saved']}")
    if summary["volumetric"]:
        row(
            "volumes",
            f"{summary['n_volumes']} x {summary['n_slices']} slice(s), "
            f"{summary['pages_per_volume']} pages each "
            f"({summary['flyback_frames']} flyback, "
            f"{summary['frames_per_slice']} frame(s)/slice)",
        )
        row("z positions", summary["zs"])
    else:
        row("volumes", "single plane")
    row("frame rate", f"{summary['frame_rate_hz']} Hz")
    row("volume rate", f"{summary['volume_rate_hz']} Hz")
    row("duration", f"{summary['duration_s']:.3f} s" if summary["duration_s"] else None)
    row("zoom", summary["zoom"])
    row("mROI", summary["mroi_enabled"])

    aux = triggers["aux"]
    if aux:
        for name, stats in aux.items():
            detail = f"{stats['n_events']} events"
            if "median_interval_s" in stats:
                detail += f", median interval {stats['median_interval_s']:.6g} s"
            if "first_timestamp_s" in stats:
                detail += f", {stats['first_timestamp_s']:.4g}-{stats['last_timestamp_s']:.4g} s"
            row(name.upper(), detail)
    else:
        row("AUX", "no trigger events")

    i2c = triggers["i2c"]
    if i2c.get("n_packets"):
        detail = f"{i2c['n_packets']} packets ({i2c.get('payload_kind')})"
        if "decoded_keys" in i2c:
            detail += f", keys: {', '.join(sorted(i2c['decoded_keys']))}"
        row("I2C", detail)
    else:
        row("I2C", "no packets")

    console.print(table)
    console.print()


@app.command(no_args_is_help=True)
def info(
    files: FilesArgument,
    acquisition: AcquisitionOption = False,
    quiet: QuietOption = False,
) -> None:
    """Show a summary of each file. Writes nothing."""
    for recording in _load(files, acquisition, quiet):
        _print_summary_table(recording)


@app.command(no_args_is_help=True)
def metadata(
    files: FilesArgument,
    out: OutOption = None,
    flat: Annotated[
        bool,
        typer.Option("--flat", help="Keep the SI.* keys flat instead of nesting them."),
    ] = False,
    no_rois: Annotated[
        bool, typer.Option("--no-rois", help="Omit the mROI/scanfield RoiGroups section.")
    ] = False,
    acquisition: AcquisitionOption = False,
    overwrite: OverwriteOption = False,
    quiet: QuietOption = False,
) -> None:
    """Export structured global metadata as JSON."""
    from scanimage_octo_reader.export import export_recording

    for recording in _load(files, acquisition, quiet):
        result = export_recording(
            recording,
            out,
            write_frames=False,
            write_aux=False,
            write_i2c=False,
            make_plot=False,
            include_rois=not no_rois,
            flat_header=flat,
            overwrite=overwrite,
            run_qc=False,
        )
        _report_written(result, quiet)


@app.command(no_args_is_help=True)
def triggers(
    files: FilesArgument,
    out: OutOption = None,
    no_aux: Annotated[bool, typer.Option("--no-aux", help="Skip the AUX trigger tables.")] = False,
    no_i2c: Annotated[bool, typer.Option("--no-i2c", help="Skip the I2C tables.")] = False,
    decode_i2c: Annotated[
        bool,
        typer.Option(
            "--decode-i2c",
            help="Also decode text I2C payloads following the '<key>_<value>' convention.",
        ),
    ] = False,
    acquisition: AcquisitionOption = False,
    overwrite: OverwriteOption = False,
    quiet: QuietOption = False,
) -> None:
    """Export AUX trigger and I2C data, plus the per-frame table, as .npy files."""
    from scanimage_octo_reader.export import export_recording

    for recording in _load(files, acquisition, quiet):
        result = export_recording(
            recording,
            out,
            write_metadata=False,
            write_aux=not no_aux,
            write_i2c=not no_i2c,
            decode_i2c=decode_i2c,
            make_plot=False,
            overwrite=overwrite,
        )
        _report_written(result, quiet)


@app.command(no_args_is_help=True)
def plot(
    files: FilesArgument,
    out: OutOption = None,
    image_formats: FormatOption = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution for raster formats.")] = 150,
    acquisition: AcquisitionOption = False,
    quiet: QuietOption = False,
) -> None:
    """Plot the frame timeline and trigger overview for each file."""
    from scanimage_octo_reader.export import default_output_root
    from scanimage_octo_reader.plots import save_overview_figure

    formats = _validated_formats(image_formats)
    for recording in _load(files, acquisition, quiet):
        root = default_output_root(recording) if out is None else Path(out)
        paths = save_overview_figure(
            recording, root / recording.name / "plots", formats=formats, dpi=dpi
        )
        if not quiet:
            for path in paths:
                console.print(f"[green]wrote[/green] {path}")


@app.command(no_args_is_help=True)
def export(
    files: FilesArgument,
    out: OutOption = None,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip the overview figure.")] = False,
    decode_i2c: Annotated[
        bool,
        typer.Option("--decode-i2c", help="Also decode '<key>_<value>' text I2C payloads."),
    ] = False,
    flat: Annotated[
        bool, typer.Option("--flat", help="Keep the SI.* keys flat instead of nesting them.")
    ] = False,
    image_formats: FormatOption = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution for raster formats.")] = 150,
    acquisition: AcquisitionOption = False,
    overwrite: OverwriteOption = False,
    quiet: QuietOption = False,
) -> None:
    """Export everything: metadata, frame table, triggers, plots and manifest."""
    from scanimage_octo_reader.export import export_recording

    formats = _validated_formats(image_formats)
    for recording in _load(files, acquisition, quiet):
        result = export_recording(
            recording,
            out,
            decode_i2c=decode_i2c,
            flat_header=flat,
            make_plot=not no_plots,
            plot_formats=formats,
            dpi=dpi,
            overwrite=overwrite,
        )
        _report_written(result, quiet)
        if result.qc and not quiet:
            _print_qc_lines(result.qc)


@app.command(no_args_is_help=True)
def pages(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
            help="A single ScanImage TIFF file.",
        ),
    ],
    start: Annotated[int, typer.Option("--start", help="First page index to show.")] = 0,
    stop: Annotated[
        int, typer.Option("--stop", help="Last page index to show (exclusive); -1 for all.")
    ] = 20,
    as_csv: Annotated[
        bool, typer.Option("--csv", help="Emit CSV on stdout instead of a table.")
    ] = False,
) -> None:
    """Show the per-page frame table, for inspecting raw page headers."""
    recording = read_recording(file)
    frames = recording.frames
    end = frames.size if stop is None or stop < 0 else min(stop, frames.size)
    selection = frames[start:end]

    fields = list(frames.dtype.names)
    if as_csv:
        typer.echo(",".join(fields))
        for row in selection:
            typer.echo(",".join(str(row[name]) for name in fields))
        return

    table = Table(title=f"{recording.name}: pages {start}-{end - 1}", box=None)
    for name in fields:
        table.add_column(name, no_wrap=True, overflow="fold")
    for row in selection:
        table.add_row(*(str(row[name]) for name in fields))
    console.print(table)


@app.command(no_args_is_help=True)
def check(
    files: FilesArgument,
    acquisition: AcquisitionOption = False,
    quiet: QuietOption = False,
) -> None:
    """Run quality-control checks; exits non-zero if any recording has errors."""
    failed = False
    for recording in _load(files, acquisition, quiet):
        report = check_recording(recording)
        status = "[green]OK[/green]" if report.ok else "[red]FAILED[/red]"
        console.print(f"{status} {recording.name}")
        _print_qc_lines(report, show_info=not quiet)
        if not report.ok:
            failed = True
    if failed:
        raise typer.Exit(code=1)


def _print_qc_lines(report, show_info: bool = False) -> None:
    styles = {"error": "red", "warning": "yellow", "info": "blue"}
    for issue in report.issues:
        if issue.level == "info" and not show_info:
            continue
        style = styles.get(issue.level, "white")
        console.print(f"  [{style}]{issue.level}[/{style}] {issue.code}: {issue.message}")


def _report_written(result, quiet: bool) -> None:
    if quiet:
        return
    console.print(
        f"[green]wrote[/green] {result.directory} "
        f"({len(result.files)} file(s): {', '.join(result.relative_files())})"
    )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
