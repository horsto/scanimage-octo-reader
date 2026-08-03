"""Reading and structuring the *global* ScanImage header.

The global header is written once per file and holds the acquisition
configuration: ~425 flat, dotted ``SI.*`` keys in the TIFF ``Software`` tag,
plus an mROI/scanfield description as JSON in the ``Artist`` tag.

`tifffile` already parses both into `TiffFile.scanimage_metadata`
(``{'version', 'FrameData', 'RoiGroups'}``), so this module's job is to get
at that robustly and to turn the flat dotted key space into something
pleasant to read and to query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tifffile
from tifffile import TiffFile

logger = logging.getLogger(__name__)

__all__ = ["ScanImageHeader", "nest_dotted_keys", "read_header", "read_tiff_tags"]

# TIFF tags worth recording in exported metadata. The ScanImage payload
# tags (Software/Artist/ImageDescription) are deliberately excluded: their
# content is already exported, parsed, elsewhere.
_TAGS_OF_INTEREST = (
    "ImageWidth",
    "ImageLength",
    "BitsPerSample",
    "SampleFormat",
    "Compression",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "RowsPerStrip",
    "Orientation",
    "XResolution",
    "YResolution",
    "ResolutionUnit",
)


def nest_dotted_keys(flat: dict[str, Any], separator: str = ".") -> dict[str, Any]:
    """Expand a flat dotted key space into nested dicts.

    ``{'SI.hScan2D.name': 'x'}`` becomes
    ``{'SI': {'hScan2D': {'name': 'x'}}}``.

    ScanImage's key space is not strictly tree-shaped: a key can be both a
    leaf and a prefix of other keys (e.g. ``SI.hStackManager.zs`` alongside
    ``SI.hStackManager.zs.something``, or a component name colliding with a
    property). Rather than silently dropping either value, a colliding leaf
    is preserved under a ``'_value'`` entry inside the branch dict, and a
    warning is logged.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = str(key).split(separator)
        cursor = nested
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                if part in cursor:
                    logger.warning(
                        "header key %r collides with a leaf value at %r; "
                        "keeping the leaf under '_value'",
                        key,
                        part,
                    )
                    cursor[part] = {"_value": existing}
                else:
                    cursor[part] = {}
            cursor = cursor[part]
        leaf = parts[-1]
        if isinstance(cursor.get(leaf), dict):
            logger.warning(
                "header key %r collides with a nested branch; storing it under '_value'", key
            )
            cursor[leaf]["_value"] = value
        else:
            cursor[leaf] = value
    return nested


@dataclass
class ScanImageHeader:
    """The parsed global header of a single ScanImage TIFF."""

    frame_data: dict[str, Any] = field(default_factory=dict)
    roi_groups: dict[str, Any] | None = None
    version: int | None = None

    @property
    def nested(self) -> dict[str, Any]:
        """`frame_data` with its dotted keys expanded into nested dicts."""
        return nest_dotted_keys(self.frame_data)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a flat header key, with or without the leading ``'SI.'``."""
        if key in self.frame_data:
            return self.frame_data[key]
        if not key.startswith("SI."):
            return self.frame_data.get(f"SI.{key}", default)
        return default

    @property
    def si_version(self) -> str | None:
        """ScanImage version as ``'<major>.<minor>.<update>'``, when available."""
        parts = [
            self.get("SI.VERSION_MAJOR"),
            self.get("SI.VERSION_MINOR"),
            self.get("SI.VERSION_UPDATE"),
        ]
        present = [str(p) for p in parts if p is not None]
        return ".".join(present) if present else None


def read_header(tif: TiffFile) -> ScanImageHeader:
    """Read the global ScanImage header from an open `TiffFile`.

    Prefers `TiffFile.scanimage_metadata`, which is only populated for
    ScanImage BigTIFFs carrying the special metadata header. Falls back to
    parsing any page's ``Software`` tag, which holds the same ``SI.*`` text
    in every ScanImage TIFF regardless of flavour, and the ``Artist`` tag
    for the ROI groups (this mirrors `napari-tiff`'s
    ``get_scanimage_framedata``).
    """
    metadata = tif.scanimage_metadata or {}
    frame_data = metadata.get("FrameData") or {}
    roi_groups = metadata.get("RoiGroups")
    version = metadata.get("version")

    if not frame_data:
        frame_data = _framedata_from_software_tag(tif)
    if roi_groups is None:
        roi_groups = _roigroups_from_artist_tag(tif)

    return ScanImageHeader(frame_data=frame_data, roi_groups=roi_groups, version=version)


def _framedata_from_software_tag(tif: TiffFile) -> dict[str, Any]:
    try:
        software = tif.pages[0].tags["Software"].value
    except (KeyError, IndexError, AttributeError):
        return {}
    try:
        parsed = tifffile.matlabstr2py(software)
    except Exception as exc:  # noqa: BLE001 - malformed header must not be fatal
        logger.warning("could not parse the Software tag as a ScanImage header: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _roigroups_from_artist_tag(tif: TiffFile) -> dict[str, Any] | None:
    import json

    try:
        artist = tif.pages[0].tags["Artist"].value
    except (KeyError, IndexError, AttributeError):
        return None
    try:
        parsed = json.loads(artist.replace("\r", ""))
    except (ValueError, AttributeError) as exc:
        logger.warning("could not parse the Artist tag as ROI group JSON: %s", exc)
        return None
    return parsed.get("RoiGroups", parsed) if isinstance(parsed, dict) else None


def read_tiff_tags(tif: TiffFile) -> dict[str, Any]:
    """Return the container-level TIFF tags of the first page, as plain values.

    Enum-valued tags (compression, photometric, ...) are reduced to their
    names so the result survives JSON serialisation unchanged.
    """
    tags: dict[str, Any] = {}
    try:
        page = tif.pages[0]
    except IndexError:
        return tags

    for name in _TAGS_OF_INTEREST:
        tag = page.tags.get(name)
        if tag is None:
            continue
        value = tag.value
        if hasattr(value, "name"):  # tifffile enum (COMPRESSION, PHOTOMETRIC, ...)
            value = value.name
        elif isinstance(value, tuple):
            value = list(value)
        tags[name] = value
    return tags
