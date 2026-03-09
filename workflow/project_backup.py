"""Project Backup - Incremental FL Studio project backup utility.

Backs up FL Studio project files and associated samples with
versioning, deduplication, and restore capabilities.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class BackupEntry:
    """A single backup snapshot."""

    timestamp: str
    source_path: str
    backup_path: str
    files: list[dict] = field(default_factory=list)  # [{name, size, hash}]
    total_size: int = 0
    notes: str = ""


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_path: str = ""
    files_backed_up: int = 0
    files_skipped: int = 0  # Already in backup (dedup)
    total_size: int = 0
    error: str | None = None


class ProjectBackup:
    """Incremental project backup with versioning."""

    MANIFEST_FILE = "backup_manifest.json"
    PROJECT_EXTENSIONS = {
        ".flp",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".aiff",
        ".fst",
        ".fxb",
        ".fxp",
        ".mid",
    }

    def __init__(self, backup_root: str | Path):
        """Initialize backup manager.

        Args:
            backup_root: Root directory for all backups
        """
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._manifest: list[BackupEntry] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load backup manifest from disk."""
        manifest_path = self.backup_root / self.MANIFEST_FILE
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                self._manifest = [BackupEntry(**entry) for entry in data]

    def _save_manifest(self) -> None:
        """Save backup manifest to disk."""
        manifest_path = self.backup_root / self.MANIFEST_FILE
        data = []
        for entry in self._manifest:
            data.append(
                {
                    "timestamp": entry.timestamp,
                    "source_path": entry.source_path,
                    "backup_path": entry.backup_path,
                    "files": entry.files,
                    "total_size": entry.total_size,
                    "notes": entry.notes,
                }
            )
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _file_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def backup(
        self, project_dir: str | Path, notes: str = "", include_samples: bool = True
    ) -> BackupResult:
        """Create a backup of an FL Studio project directory.

        Args:
            project_dir: Directory containing the project files
            notes: Optional notes for this backup
            include_samples: Include audio samples in backup

        Returns:
            BackupResult with summary
        """
        project_path = Path(project_dir)
        if not project_path.exists():
            return BackupResult(False, error=f"Directory not found: {project_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{project_path.name}_{timestamp}"
        backup_dir = self.backup_root / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Collect existing file hashes for dedup
        existing_hashes = set()
        for entry in self._manifest:
            for file_rec in entry.files:
                existing_hashes.add(file_rec.get("hash", ""))

        # Find files to back up
        extensions = self.PROJECT_EXTENSIONS if include_samples else {".flp", ".fst"}
        files_info = []
        files_backed_up = 0
        files_skipped = 0
        total_size = 0

        try:
            for f in sorted(project_path.rglob("*")):
                if f.is_file() and f.suffix.lower() in extensions:
                    file_hash = self._file_hash(f)
                    file_size = f.stat().st_size
                    rel_path = f.relative_to(project_path)

                    file_info = {
                        "name": str(rel_path),
                        "size": file_size,
                        "hash": file_hash,
                    }

                    if file_hash in existing_hashes:
                        files_skipped += 1
                        file_info["status"] = "dedup"
                    else:
                        # Copy file to backup
                        dest = backup_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(f), str(dest))
                        files_backed_up += 1
                        total_size += file_size
                        file_info["status"] = "backed_up"

                    files_info.append(file_info)

            # Create backup entry
            entry = BackupEntry(
                timestamp=timestamp,
                source_path=str(project_path),
                backup_path=str(backup_dir),
                files=files_info,
                total_size=total_size,
                notes=notes,
            )
            self._manifest.append(entry)
            self._save_manifest()

            return BackupResult(
                success=True,
                backup_path=str(backup_dir),
                files_backed_up=files_backed_up,
                files_skipped=files_skipped,
                total_size=total_size,
            )

        except Exception as e:
            return BackupResult(False, error=str(e))

    def restore(self, backup_index: int = -1, restore_dir: str | Path | None = None) -> bool:
        """Restore a backup.

        Args:
            backup_index: Index in manifest (-1 = latest)
            restore_dir: Where to restore (None = original location)
        """
        if not self._manifest:
            return False

        entry = self._manifest[backup_index]
        backup_path = Path(entry.backup_path)
        dest = Path(restore_dir) if restore_dir else Path(entry.source_path)
        dest.mkdir(parents=True, exist_ok=True)

        for file_info in entry.files:
            src = backup_path / file_info["name"]
            dst = dest / file_info["name"]
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        return True

    def list_backups(self) -> list[dict]:
        """List all backups."""
        return [
            {
                "index": i,
                "timestamp": entry.timestamp,
                "source": entry.source_path,
                "files": len(entry.files),
                "size_mb": entry.total_size / (1024 * 1024),
                "notes": entry.notes,
            }
            for i, entry in enumerate(self._manifest)
        ]

    def cleanup(self, keep_latest: int = 5) -> int:
        """Remove old backups, keeping only the latest N.

        Returns:
            Number of backups removed
        """
        if len(self._manifest) <= keep_latest:
            return 0

        to_remove = self._manifest[:-keep_latest]
        removed = 0

        for entry in to_remove:
            backup_path = Path(entry.backup_path)
            if backup_path.exists():
                shutil.rmtree(str(backup_path))
                removed += 1

        self._manifest = self._manifest[-keep_latest:]
        self._save_manifest()
        return removed
