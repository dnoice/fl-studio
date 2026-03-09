"""FL Studio Toolkit - Release & Distribution Pipeline

Metadata tagging, lyrics management, licensing schemas, album packaging,
and export workflows for preparing music for distribution.

Quick start::

    from release import TrackMetadata, MetadataManager, Album, ExportPipeline

    meta = TrackMetadata(title="My Track", artist="Me", album="My Album")
    MetadataManager.write_tags("track.wav", meta)
    pipeline = ExportPipeline(album, config)
    result = pipeline.execute()  # -> dict with export results
"""

from release.album import Album, AlbumTrack, DiscInfo
from release.export_pipeline import ExportConfig, ExportPipeline
from release.licensing import (
    LicenseInfo,
    LicensingManager,
    MechanicalLicense,
    PublishingInfo,
    SyncLicense,
    WriterSplit,
)
from release.lyrics import Lyrics, LyricsManager, SyncedLine
from release.metadata import MetadataManager, TrackMetadata

__all__ = [
    "Album",
    "AlbumTrack",
    "DiscInfo",
    "ExportConfig",
    "ExportPipeline",
    "LicenseInfo",
    "LicensingManager",
    "MechanicalLicense",
    "PublishingInfo",
    "SyncLicense",
    "WriterSplit",
    "Lyrics",
    "LyricsManager",
    "SyncedLine",
    "MetadataManager",
    "TrackMetadata",
]
