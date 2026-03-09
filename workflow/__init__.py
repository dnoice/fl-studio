"""FL Studio Toolkit - Workflow Automation

Project management tools including FLP parsing, preset management,
sample organization, project backup, and render queue management.

Quick start::

    from workflow import FLPParser, PresetManager, ProjectBackup

    project = FLPParser.parse("my_song.flp")       # -> FLPProject
    presets = PresetManager("presets/").scan()       # -> list[PresetInfo]
    backup = ProjectBackup("project/").backup("backups/") # -> BackupResult
"""

from workflow.flp_parser import FLPChannel, FLPParser, FLPPattern, FLPProject
from workflow.preset_manager import PresetInfo, PresetManager
from workflow.project_backup import BackupResult, ProjectBackup
from workflow.render_queue import RenderFormat, RenderJob, RenderQueue
from workflow.sample_organizer import OrganizeResult, SampleInfo, SampleOrganizer

__all__ = [
    "FLPChannel",
    "FLPParser",
    "FLPPattern",
    "FLPProject",
    "PresetInfo",
    "PresetManager",
    "BackupResult",
    "ProjectBackup",
    "RenderFormat",
    "RenderJob",
    "RenderQueue",
    "OrganizeResult",
    "SampleInfo",
    "SampleOrganizer",
]

__version__ = "0.3.0"
