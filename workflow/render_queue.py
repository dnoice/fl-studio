"""Render Queue - Batch render management for FL Studio projects.

Manages a queue of FL Studio projects to render, with configurable
output settings, progress tracking, and job management.

Note: Actual rendering requires FL Studio's command-line interface.
This module manages the queue and invokes FL Studio for rendering.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class RenderFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class RenderStatus(Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenderJob:
    """A single render job configuration."""

    id: int
    project_path: str
    output_path: str
    format: str = "wav"
    bit_depth: int = 24
    sample_rate: int = 44100
    status: str = "pending"
    created_at: str = ""
    completed_at: str = ""
    error: str = ""
    duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class RenderQueue:
    """Batch render queue manager for FL Studio projects.

    FL Studio supports command-line rendering via:
        FL64.exe /R /Ewav "project.flp"

    This class manages a queue of render jobs.
    """

    FL_STUDIO_PATH = r"C:\Program Files\Image-Line\FL Studio 2025\FL64.exe"

    def __init__(self, queue_file: str | Path | None = None):
        """Initialize render queue.

        Args:
            queue_file: Path to persist queue state (JSON)
        """
        self._jobs: list[RenderJob] = []
        self._next_id = 1
        self._queue_file = Path(queue_file) if queue_file else None
        if self._queue_file and self._queue_file.exists():
            self._load()

    def add(
        self,
        project_path: str | Path,
        output_dir: str | Path | None = None,
        format: str = "wav",
        bit_depth: int = 24,
        sample_rate: int = 44100,
    ) -> RenderJob:
        """Add a project to the render queue.

        Args:
            project_path: Path to .flp file
            output_dir: Output directory (default: same as project)
            format: Output format
            bit_depth: Bit depth
            sample_rate: Sample rate

        Returns:
            The created RenderJob
        """
        project = Path(project_path)
        out_dir = Path(output_dir) if output_dir else project.parent

        output_path = out_dir / f"{project.stem}.{format}"

        job = RenderJob(
            id=self._next_id,
            project_path=str(project),
            output_path=str(output_path),
            format=format,
            bit_depth=bit_depth,
            sample_rate=sample_rate,
        )
        self._jobs.append(job)
        self._next_id += 1
        self._save()
        return job

    def add_batch(
        self,
        project_dir: str | Path,
        output_dir: str | Path | None = None,
        recursive: bool = True,
        **kwargs,
    ) -> list[RenderJob]:
        """Add all .flp files from a directory to the queue."""
        path = Path(project_dir)
        pattern = "**/*.flp" if recursive else "*.flp"
        jobs = []
        for flp in sorted(path.glob(pattern)):
            job = self.add(flp, output_dir, **kwargs)
            jobs.append(job)
        return jobs

    def remove(self, job_id: int) -> bool:
        """Remove a job from the queue."""
        self._jobs = [j for j in self._jobs if j.id != job_id]
        self._save()
        return True

    def clear(self, status: str | None = None) -> int:
        """Clear jobs from the queue.

        Args:
            status: Only clear jobs with this status (None = all)
        """
        if status:
            before = len(self._jobs)
            self._jobs = [j for j in self._jobs if j.status != status]
            removed = before - len(self._jobs)
        else:
            removed = len(self._jobs)
            self._jobs.clear()
        self._save()
        return removed

    def render_next(self, fl_path: str | None = None) -> RenderJob | None:
        """Render the next pending job.

        Args:
            fl_path: Path to FL64.exe (uses default if None)

        Returns:
            The rendered job, or None if queue is empty
        """
        fl_exe = fl_path or self.FL_STUDIO_PATH

        # Find next pending job
        job = next((j for j in self._jobs if j.status == "pending"), None)
        if not job:
            return None

        job.status = "rendering"
        self._save()

        try:
            # FL Studio command-line render
            format_flag = f"/E{job.format}"
            cmd = [fl_exe, "/R", format_flag, job.project_path]

            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            elapsed = (datetime.now() - start_time).total_seconds()
            job.duration_seconds = elapsed

            if result.returncode == 0:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
            else:
                job.status = "failed"
                job.error = result.stderr or f"Exit code: {result.returncode}"

        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "Render timed out (1 hour limit)"
        except FileNotFoundError:
            job.status = "failed"
            job.error = f"FL Studio not found at: {fl_exe}"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)

        self._save()
        return job

    def render_all(self, fl_path: str | None = None) -> list[RenderJob]:
        """Render all pending jobs sequentially."""
        rendered = []
        while True:
            job = self.render_next(fl_path)
            if job is None:
                break
            rendered.append(job)
        return rendered

    def list_jobs(self, status: str | None = None) -> list[RenderJob]:
        """List all jobs, optionally filtered by status."""
        if status:
            return [j for j in self._jobs if j.status == status]
        return list(self._jobs)

    def stats(self) -> dict:
        """Get queue statistics."""
        statuses: dict[str, int] = {}
        for j in self._jobs:
            statuses[j.status] = statuses.get(j.status, 0) + 1
        return {
            "total": len(self._jobs),
            **statuses,
        }

    def _save(self) -> None:
        """Persist queue state to disk."""
        if not self._queue_file:
            return
        self._queue_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "jobs": [
                {
                    "id": j.id,
                    "project_path": j.project_path,
                    "output_path": j.output_path,
                    "format": j.format,
                    "bit_depth": j.bit_depth,
                    "sample_rate": j.sample_rate,
                    "status": j.status,
                    "created_at": j.created_at,
                    "completed_at": j.completed_at,
                    "error": j.error,
                    "duration_seconds": j.duration_seconds,
                }
                for j in self._jobs
            ],
        }
        with open(self._queue_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load queue state from disk."""
        if not self._queue_file or not self._queue_file.exists():
            return
        with open(self._queue_file) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 1)
        self._jobs = [RenderJob(**j) for j in data.get("jobs", [])]
