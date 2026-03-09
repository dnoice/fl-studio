"""Sample Organizer - Organize audio sample libraries.

Categorizes samples by type (kick, snare, hi-hat, etc.) using
filename analysis and audio characteristics, then organizes them
into a structured directory hierarchy.
"""

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# Keyword-based sample classification
CATEGORY_KEYWORDS = {
    "kick": ["kick", "kck", "bd", "bass drum", "bassdrum", "bombo"],
    "snare": ["snare", "snr", "sd"],
    "clap": ["clap", "clp", "handclap"],
    "hi-hat": [
        "hihat",
        "hi-hat",
        "hh",
        "hat",
        "closed hat",
        "open hat",
        "closedhat",
        "openhat",
        "chh",
        "ohh",
    ],
    "cymbal": ["cymbal", "crash", "ride", "splash", "china"],
    "tom": ["tom", "floor tom", "rack tom"],
    "percussion": [
        "perc",
        "percussion",
        "shaker",
        "tambourine",
        "conga",
        "bongo",
        "cowbell",
        "triangle",
        "guiro",
        "clave",
        "rimshot",
        "rim",
    ],
    "bass": ["bass", "sub", "808"],
    "lead": ["lead", "ld", "synth lead"],
    "pad": ["pad", "atmosphere", "ambient"],
    "keys": ["piano", "keys", "organ", "rhodes", "wurli", "epiano"],
    "guitar": ["guitar", "gtr", "acoustic guitar", "electric guitar"],
    "strings": ["strings", "violin", "viola", "cello", "orchestral"],
    "brass": ["brass", "trumpet", "horn", "trombone", "sax"],
    "vocal": ["vocal", "vox", "voice", "choir", "acapella"],
    "fx": [
        "fx",
        "sfx",
        "effect",
        "riser",
        "downlifter",
        "uplifter",
        "sweep",
        "noise",
        "impact",
        "whoosh",
        "transition",
    ],
    "loop": ["loop", "break", "breakbeat", "drum loop"],
    "one-shot": ["one-shot", "oneshot", "hit", "stab", "shot"],
}


@dataclass
class SampleInfo:
    """Information about a single audio sample."""

    filename: str
    filepath: str
    category: str = "uncategorized"
    subcategory: str = ""
    duration_ms: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    peak_db: float = -120.0
    estimated_bpm: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class OrganizeResult:
    """Result of a sample organization operation."""

    total_files: int = 0
    categorized: int = 0
    uncategorized: int = 0
    moved: int = 0
    errors: list[str] = field(default_factory=list)
    categories: dict[str, int] = field(default_factory=dict)


class SampleOrganizer:
    """Organize and categorize audio sample libraries."""

    AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg"}

    @staticmethod
    def analyze_sample(filepath: str | Path) -> SampleInfo:
        """Analyze a single audio sample."""
        filepath = Path(filepath)

        info = SampleInfo(
            filename=filepath.name,
            filepath=str(filepath),
        )

        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            info.sample_rate = sr
            info.channels = audio.shape[1] if audio.ndim > 1 else 1
            info.duration_ms = len(audio) / sr * 1000 if sr > 0 else 0.0

            if len(audio) > 0:
                peak = np.max(np.abs(audio))
                info.peak_db = float(20 * np.log10(max(peak, 1e-10)))
        except (sf.SoundFileError, OSError, RuntimeError):
            pass  # Unsupported or corrupted audio file

        # Categorize by filename
        info.category = SampleOrganizer._categorize_by_name(filepath.stem)

        return info

    @staticmethod
    def _categorize_by_name(filename: str) -> str:
        """Categorize a sample by its filename."""
        name_lower = filename.lower()
        # Remove common separators and normalize
        name_clean = re.sub(r"[_\-.\d]+", " ", name_lower)

        best_category = "uncategorized"
        best_score = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in name_clean or keyword in name_lower:
                    # Prefer longer keyword matches
                    score = len(keyword)
                    if score > best_score:
                        best_score = score
                        best_category = category

        return best_category

    @staticmethod
    def scan_directory(directory: str | Path, recursive: bool = True) -> list[SampleInfo]:
        """Scan a directory for audio samples and analyze them.

        Args:
            directory: Directory to scan
            recursive: Search subdirectories
        """
        path = Path(directory)
        results = []

        pattern = "**/*" if recursive else "*"
        for f in sorted(path.glob(pattern)):
            if f.suffix.lower() in SampleOrganizer.AUDIO_EXTENSIONS and f.is_file():
                info = SampleOrganizer.analyze_sample(f)
                results.append(info)

        return results

    @staticmethod
    def organize(
        source_dir: str | Path,
        output_dir: str | Path,
        recursive: bool = True,
        copy: bool = True,
        dry_run: bool = False,
    ) -> OrganizeResult:
        """Organize samples into categorized subdirectories.

        Args:
            source_dir: Source directory with unorganized samples
            output_dir: Output directory for organized structure
            recursive: Scan subdirectories
            copy: Copy files (True) or move them (False)
            dry_run: If True, don't actually move/copy files

        Returns:
            OrganizeResult with summary
        """
        samples = SampleOrganizer.scan_directory(source_dir, recursive)
        output_path = Path(output_dir)
        result = OrganizeResult(total_files=len(samples))

        for sample in samples:
            if sample.category != "uncategorized":
                result.categorized += 1
            else:
                result.uncategorized += 1

            # Track category counts
            result.categories[sample.category] = result.categories.get(sample.category, 0) + 1

            # Build output path
            cat_dir = output_path / sample.category
            dest = cat_dir / sample.filename

            if not dry_run:
                try:
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    src = Path(sample.filepath)

                    # Handle filename conflicts
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = cat_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    if copy:
                        shutil.copy2(str(src), str(dest))
                    else:
                        shutil.move(str(src), str(dest))
                    result.moved += 1
                except Exception as e:
                    result.errors.append(f"{sample.filename}: {e}")
            else:
                result.moved += 1

        return result

    @staticmethod
    def generate_report(samples: list[SampleInfo]) -> str:
        """Generate a text report of sample analysis."""
        if not samples:
            return "No samples found."

        categories: dict[str, list[SampleInfo]] = {}
        total_duration: float = 0.0
        for s in samples:
            if s.category not in categories:
                categories[s.category] = []
            categories[s.category].append(s)
            total_duration += s.duration_ms

        lines = [
            "=== Sample Library Report ===",
            f"Total Samples: {len(samples)}",
            f"Total Duration: {total_duration/1000/60:.1f} minutes",
            f"Categories: {len(categories)}",
            "",
        ]

        for cat, cat_samples in sorted(categories.items()):
            durations = [s.duration_ms for s in cat_samples]
            lines.append(f"--- {cat} ({len(cat_samples)} samples) ---")
            lines.append(f"  Duration range: {min(durations):.0f}ms - {max(durations):.0f}ms")
            if len(durations) > 0:
                lines.append(f"  Avg duration: {sum(durations)/len(durations):.0f}ms")
            lines.append("")

        return "\n".join(lines)
