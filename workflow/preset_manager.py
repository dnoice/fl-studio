"""Preset Manager - Search, index, and organize FL Studio presets.

Scans FL Studio's preset directories for .fst files and provides
search, tagging, and organization capabilities.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PresetInfo:
    """Information about a single preset file."""

    name: str
    filepath: str
    plugin: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    size_bytes: int = 0
    modified: float = 0.0

    @property
    def filename(self) -> str:
        return Path(self.filepath).name

    def matches(self, query: str) -> bool:
        """Check if preset matches a search query (case-insensitive)."""
        q = query.lower()
        return (
            q in self.name.lower()
            or q in self.plugin.lower()
            or q in self.category.lower()
            or any(q in tag.lower() for tag in self.tags)
        )


class PresetManager:
    """FL Studio preset search and management."""

    DEFAULT_PRESET_DIRS = [
        r"C:\Program Files\Image-Line\FL Studio 2025\Data\Patches",
        r"C:\Users\{user}\Documents\Image-Line\FL Studio\Presets",
    ]

    def __init__(self, preset_dirs: list[str] | None = None):
        """Initialize with preset directories to scan.

        Args:
            preset_dirs: List of directories to scan. Uses defaults if None.
        """
        if preset_dirs:
            self.preset_dirs = [Path(d) for d in preset_dirs]
        else:
            self.preset_dirs = []
            for d in self.DEFAULT_PRESET_DIRS:
                expanded = d.replace("{user}", os.environ.get("USERNAME", "User"))
                path = Path(expanded)
                if path.exists():
                    self.preset_dirs.append(path)

        self._index: list[PresetInfo] = []
        self._tags_db: dict[str, list[str]] = {}  # filepath -> tags
        self._tags_file: Path | None = None

    def scan(self, extensions: tuple[str, ...] = (".fst", ".fxb", ".fxp")) -> int:
        """Scan preset directories and build the index.

        Returns:
            Number of presets found
        """
        self._index.clear()

        for preset_dir in self.preset_dirs:
            if not preset_dir.exists():
                continue

            for f in preset_dir.rglob("*"):
                if f.suffix.lower() in extensions and f.is_file() and not f.is_symlink():
                    # Determine plugin name from directory structure
                    try:
                        rel = f.relative_to(preset_dir)
                        parts = rel.parts
                        plugin = parts[0] if len(parts) > 1 else ""
                        category = parts[1] if len(parts) > 2 else ""
                    except ValueError:
                        plugin = ""
                        category = ""

                    tags = self._tags_db.get(str(f), [])

                    preset = PresetInfo(
                        name=f.stem,
                        filepath=str(f),
                        plugin=plugin,
                        category=category,
                        tags=tags,
                        size_bytes=f.stat().st_size,
                        modified=f.stat().st_mtime,
                    )
                    self._index.append(preset)

        return len(self._index)

    def search(
        self,
        query: str = "",
        plugin: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[PresetInfo]:
        """Search presets by query, plugin, category, or tags.

        Args:
            query: Search text (matches name, plugin, category, tags)
            plugin: Filter by plugin name
            category: Filter by category
            tags: Filter by tags (preset must have all listed tags)
            limit: Maximum results
        """
        results = self._index

        if query:
            results = [p for p in results if p.matches(query)]

        if plugin:
            plugin_lower = plugin.lower()
            results = [p for p in results if plugin_lower in p.plugin.lower()]

        if category:
            cat_lower = category.lower()
            results = [p for p in results if cat_lower in p.category.lower()]

        if tags:
            tags_lower = [t.lower() for t in tags]
            results = [
                p for p in results if all(any(t in pt.lower() for pt in p.tags) for t in tags_lower)
            ]

        return results[:limit]

    def list_plugins(self) -> list[str]:
        """Get sorted list of all plugin names found."""
        return sorted(set(p.plugin for p in self._index if p.plugin))

    def list_categories(self, plugin: str | None = None) -> list[str]:
        """Get sorted list of categories, optionally filtered by plugin."""
        if plugin:
            presets = [p for p in self._index if p.plugin.lower() == plugin.lower()]
        else:
            presets = self._index
        return sorted(set(p.category for p in presets if p.category))

    def add_tag(self, filepath: str, tag: str) -> None:
        """Add a tag to a preset."""
        if filepath not in self._tags_db:
            self._tags_db[filepath] = []
        if tag not in self._tags_db[filepath]:
            self._tags_db[filepath].append(tag)

        # Update in-memory index
        for p in self._index:
            if p.filepath == filepath:
                if tag not in p.tags:
                    p.tags.append(tag)
                break

    def remove_tag(self, filepath: str, tag: str) -> None:
        """Remove a tag from a preset."""
        if filepath in self._tags_db:
            self._tags_db[filepath] = [t for t in self._tags_db[filepath] if t != tag]

        for p in self._index:
            if p.filepath == filepath:
                p.tags = [t for t in p.tags if t != tag]
                break

    def save_tags(self, filepath: str | Path) -> None:
        """Save tags database to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self._tags_db, f, indent=2)

    def load_tags(self, filepath: str | Path) -> None:
        """Load tags database from a JSON file."""
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath) as f:
                self._tags_db = json.load(f)
            # Apply to index
            for p in self._index:
                if p.filepath in self._tags_db:
                    p.tags = self._tags_db[p.filepath]

    def stats(self) -> dict:
        """Get statistics about the preset library."""
        return {
            "total_presets": len(self._index),
            "plugins": len(self.list_plugins()),
            "categories": len(self.list_categories()),
            "tagged": sum(1 for p in self._index if p.tags),
            "total_size_mb": sum(p.size_bytes for p in self._index) / (1024 * 1024),
        }
