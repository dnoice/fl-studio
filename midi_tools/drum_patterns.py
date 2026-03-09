"""Drum Pattern Library - Genre-based drum pattern generation.

Provides pattern templates for various genres with configurable
velocity, swing, fill generation, and humanization.
"""

import copy
import random
from dataclasses import dataclass, field

# General MIDI drum map (standard)
GM_DRUMS = {
    "kick": 36,
    "rim": 37,
    "snare": 38,
    "clap": 39,
    "closed_hat": 42,
    "open_hat": 46,
    "pedal_hat": 44,
    "low_tom": 41,
    "mid_tom": 47,
    "high_tom": 50,
    "crash": 49,
    "ride": 51,
    "ride_bell": 53,
    "tambourine": 54,
    "cowbell": 56,
    "shaker": 70,
    "perc_1": 60,
    "perc_2": 61,
}

# FPC default mapping (FL Studio's FPC plugin)
FPC_DRUMS = {
    "kick": 36,
    "snare": 38,
    "clap": 39,
    "closed_hat": 42,
    "open_hat": 46,
    "low_tom": 41,
    "mid_tom": 47,
    "high_tom": 50,
    "crash": 49,
    "ride": 51,
    "rim": 37,
    "shaker": 70,
}


@dataclass
class DrumHit:
    """A single drum hit."""

    instrument: str  # Key from drum map (e.g., 'kick', 'snare')
    step: int  # Step position (0-based)
    velocity: float  # 0.0 - 1.0
    offset: float = 0.0  # Timing offset in fractions of a step (-0.5 to 0.5)


@dataclass
class DrumPattern:
    """A complete drum pattern."""

    name: str
    hits: list[DrumHit] = field(default_factory=list)
    steps: int = 16  # Total steps in pattern
    step_size: float = 0.25  # Step size in beats (0.25 = 16th note)
    swing: float = 0.0  # Swing amount 0.0-1.0
    genre: str = ""

    @property
    def total_beats(self) -> float:
        return self.steps * self.step_size

    @property
    def instruments_used(self) -> set[str]:
        return {h.instrument for h in self.hits}

    def get_hits_for(self, instrument: str) -> list[DrumHit]:
        """Get all hits for a specific instrument."""
        return [h for h in self.hits if h.instrument == instrument]

    def add_hit(
        self, instrument: str, step: int, velocity: float = 0.8, offset: float = 0.0
    ) -> None:
        """Add a single hit to the pattern."""
        self.hits.append(DrumHit(instrument, step % self.steps, velocity, offset))

    def remove_instrument(self, instrument: str) -> None:
        """Remove all hits for an instrument."""
        self.hits = [h for h in self.hits if h.instrument != instrument]

    def apply_swing(self, amount: float | None = None) -> "DrumPattern":
        """Apply swing to odd-numbered steps."""
        swing = amount if amount is not None else self.swing
        result = copy.deepcopy(self)
        for hit in result.hits:
            if hit.step % 2 == 1:
                hit.offset += swing * 0.5
        result.swing = swing
        return result

    def humanize(self, timing: float = 0.02, velocity: float = 0.05) -> "DrumPattern":
        """Add subtle randomization to timing and velocity.

        Args:
            timing: Max timing offset in fractions of a step
            velocity: Max velocity variation
        """
        result = copy.deepcopy(self)
        for hit in result.hits:
            hit.offset += random.uniform(-timing, timing)
            hit.velocity = max(0.01, min(1.0, hit.velocity + random.uniform(-velocity, velocity)))
        return result

    def double(self) -> "DrumPattern":
        """Double the pattern length by repeating."""
        result = copy.deepcopy(self)
        original_steps = self.steps
        result.steps *= 2
        for hit in list(self.hits):
            result.hits.append(
                DrumHit(hit.instrument, hit.step + original_steps, hit.velocity, hit.offset)
            )
        return result

    def half(self) -> "DrumPattern":
        """Take only the first half of the pattern."""
        result = copy.deepcopy(self)
        half_steps = self.steps // 2
        result.steps = half_steps
        result.hits = [h for h in result.hits if h.step < half_steps]
        return result

    def merge(self, other: "DrumPattern") -> "DrumPattern":
        """Merge another pattern's hits into this one."""
        result = copy.deepcopy(self)
        for hit in other.hits:
            if hit.step < result.steps:
                result.hits.append(copy.deepcopy(hit))
        return result

    def to_grid(self, instrument: str) -> list[float]:
        """Get a velocity grid for an instrument (0.0 = no hit)."""
        grid = [0.0] * self.steps
        for hit in self.hits:
            if hit.instrument == instrument:
                grid[hit.step] = hit.velocity
        return grid

    @classmethod
    def from_grid(
        cls, name: str, grids: dict[str, list[float]], step_size: float = 0.25
    ) -> "DrumPattern":
        """Create pattern from velocity grids.

        Args:
            name: Pattern name
            grids: Dict mapping instrument name -> velocity list
                   e.g., {'kick': [1, 0, 0, 0, 1, 0, 0, 0, ...]}
            step_size: Step size in beats
        """
        hits = []
        steps = 0
        for instrument, grid in grids.items():
            steps = max(steps, len(grid))
            for i, vel in enumerate(grid):
                if vel > 0:
                    hits.append(DrumHit(instrument, i, vel))
        return cls(name=name, hits=hits, steps=steps, step_size=step_size)


# Shorthand: 'x' = full hit, 'o' = soft hit, '-' = rest, 'X' = accent
def _parse_pattern(
    pattern_str: str, accent_vel: float = 1.0, normal_vel: float = 0.8, soft_vel: float = 0.5
) -> list[float]:
    """Parse a pattern string into velocity list."""
    result = []
    for ch in pattern_str:
        if ch == "X":
            result.append(accent_vel)
        elif ch == "x":
            result.append(normal_vel)
        elif ch == "o":
            result.append(soft_vel)
        elif ch == "-":
            result.append(0.0)
    return result


class DrumPatternLibrary:
    """Collection of genre-specific drum pattern presets."""

    # ─── Hip-Hop / Trap ───

    @staticmethod
    def trap_basic() -> DrumPattern:
        return DrumPattern.from_grid(
            "Trap Basic",
            {
                "kick": _parse_pattern("X---x---X-----x-"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "open_hat": _parse_pattern("--x---x---x---x-"),
            },
        )

    @staticmethod
    def trap_hihat_rolls() -> DrumPattern:
        """Trap with rapid hi-hat rolls."""
        return DrumPattern.from_grid(
            "Trap Hi-Hat Rolls",
            {
                "kick": _parse_pattern("X---x-----x-x---"),
                "snare": _parse_pattern("----X-------X---"),
                "clap": _parse_pattern("----x-------x---"),
                "closed_hat": _parse_pattern("xxXxxoxxXxxoxxXx"),
                "open_hat": _parse_pattern("------x-------x-"),
            },
        )

    @staticmethod
    def boom_bap() -> DrumPattern:
        return DrumPattern.from_grid(
            "Boom Bap",
            {
                "kick": _parse_pattern("X-----x-x---x---"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "open_hat": _parse_pattern("--x-----------x-"),
            },
        )

    @staticmethod
    def lofi_hip_hop() -> DrumPattern:
        return DrumPattern.from_grid(
            "Lo-Fi Hip Hop",
            {
                "kick": _parse_pattern("X-----o---x-----"),
                "snare": _parse_pattern("----x-------X---"),
                "closed_hat": _parse_pattern("xox-xox-xox-xox-"),
                "shaker": _parse_pattern("--o---o---o---o-"),
            },
        )

    # ─── House / EDM ───

    @staticmethod
    def house_basic() -> DrumPattern:
        """Four-on-the-floor house beat."""
        return DrumPattern.from_grid(
            "House Basic",
            {
                "kick": _parse_pattern("X---X---X---X---"),
                "clap": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("-x-x-x-x-x-x-x-"),
                "open_hat": _parse_pattern("--x-------x-----"),
            },
        )

    @staticmethod
    def deep_house() -> DrumPattern:
        return DrumPattern.from_grid(
            "Deep House",
            {
                "kick": _parse_pattern("X---X---X---X---"),
                "snare": _parse_pattern("----x-------x---"),
                "closed_hat": _parse_pattern("xox-xox-xox-xox-"),
                "open_hat": _parse_pattern("------x-------x-"),
                "shaker": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    @staticmethod
    def tech_house() -> DrumPattern:
        return DrumPattern.from_grid(
            "Tech House",
            {
                "kick": _parse_pattern("X---X---X---X---"),
                "clap": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("xxo-xxo-xxo-xxo-"),
                "rim": _parse_pattern("-x----x--x----x-"),
                "shaker": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    # ─── Techno ───

    @staticmethod
    def techno_basic() -> DrumPattern:
        return DrumPattern.from_grid(
            "Techno Basic",
            {
                "kick": _parse_pattern("X---X---X---X---"),
                "clap": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("--x---x---x---x-"),
                "ride": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    @staticmethod
    def industrial_techno() -> DrumPattern:
        return DrumPattern.from_grid(
            "Industrial Techno",
            {
                "kick": _parse_pattern("X--xX---X--xX---"),
                "clap": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "rim": _parse_pattern("--x---x---x---x-"),
                "perc_1": _parse_pattern("x-------x-------"),
            },
        )

    # ─── Drum & Bass ───

    @staticmethod
    def dnb_basic() -> DrumPattern:
        """Standard DnB two-step."""
        return DrumPattern.from_grid(
            "DnB Basic",
            {
                "kick": _parse_pattern("X---------x-----"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    @staticmethod
    def jungle() -> DrumPattern:
        return DrumPattern.from_grid(
            "Jungle",
            {
                "kick": _parse_pattern("X-----x---x-----"),
                "snare": _parse_pattern("----X--x----X-x-"),
                "closed_hat": _parse_pattern("xxo-xxo-xxo-xxo-"),
                "open_hat": _parse_pattern("--x-------x-----"),
            },
        )

    # ─── Rock / Pop ───

    @staticmethod
    def rock_basic() -> DrumPattern:
        return DrumPattern.from_grid(
            "Rock Basic",
            {
                "kick": _parse_pattern("X---x---X---x---"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    @staticmethod
    def pop_beat() -> DrumPattern:
        return DrumPattern.from_grid(
            "Pop Beat",
            {
                "kick": _parse_pattern("X---x---X-x-x---"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "open_hat": _parse_pattern("------x---------"),
            },
        )

    # ─── Latin / World ───

    @staticmethod
    def reggaeton() -> DrumPattern:
        """Dembow rhythm."""
        return DrumPattern.from_grid(
            "Reggaeton",
            {
                "kick": _parse_pattern("X---x--xX---x--x"),
                "snare": _parse_pattern("---X---x---X---x"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
            },
        )

    @staticmethod
    def bossa_nova() -> DrumPattern:
        return DrumPattern.from_grid(
            "Bossa Nova",
            {
                "kick": _parse_pattern("X--x--x---x--x--"),
                "rim": _parse_pattern("--x--x--x--x--x-"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "shaker": _parse_pattern("xoxoxoxoxoxoxoxo"),
            },
        )

    # ─── Electronic Sub-genres ───

    @staticmethod
    def dubstep_halftime() -> DrumPattern:
        return DrumPattern.from_grid(
            "Dubstep Halftime",
            {
                "kick": _parse_pattern("X-----------x---"),
                "snare": _parse_pattern("--------X-------"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "open_hat": _parse_pattern("------x---------"),
            },
        )

    @staticmethod
    def uk_garage() -> DrumPattern:
        """2-step garage pattern."""
        return DrumPattern.from_grid(
            "UK Garage",
            {
                "kick": _parse_pattern("X-----x-x-----x-"),
                "snare": _parse_pattern("----X-------X---"),
                "closed_hat": _parse_pattern("xox-xox-xox-xox-"),
                "open_hat": _parse_pattern("--x-------x-----"),
            },
        )

    @staticmethod
    def breakbeat() -> DrumPattern:
        return DrumPattern.from_grid(
            "Breakbeat",
            {
                "kick": _parse_pattern("X-----x---x-----"),
                "snare": _parse_pattern("----X-----x-X---"),
                "closed_hat": _parse_pattern("x-x-x-x-x-x-x-x"),
                "open_hat": _parse_pattern("------x---------"),
            },
        )

    # ─── Fill Generation ───

    @staticmethod
    def generate_fill(
        base_pattern: DrumPattern, intensity: float = 0.7, fill_bars: int = 1
    ) -> DrumPattern:
        """Generate a fill pattern based on a base pattern.

        Args:
            base_pattern: The pattern to base the fill on
            intensity: Fill intensity 0.0-1.0 (more hits at higher values)
            fill_bars: Number of bars for the fill
        """
        result = copy.deepcopy(base_pattern)
        steps = result.steps

        # Add snare/tom fills in the last quarter
        fill_start = int(steps * 0.75)
        fill_instruments = ["snare", "high_tom", "mid_tom", "low_tom"]

        for step in range(fill_start, steps):
            progress = (step - fill_start) / (steps - fill_start)
            if random.random() < intensity * (0.5 + 0.5 * progress):
                inst = random.choice(
                    fill_instruments[: max(1, int(len(fill_instruments) * progress))]
                )
                vel = 0.6 + 0.4 * progress
                result.add_hit(inst, step, vel)

        # Add crash on beat 1 (convention: fill leads into crash)
        result.name = f"{base_pattern.name} Fill"
        return result

    @staticmethod
    def list_patterns() -> list[str]:
        """List all available preset pattern names."""
        return [
            "trap_basic",
            "trap_hihat_rolls",
            "boom_bap",
            "lofi_hip_hop",
            "house_basic",
            "deep_house",
            "tech_house",
            "techno_basic",
            "industrial_techno",
            "dnb_basic",
            "jungle",
            "rock_basic",
            "pop_beat",
            "reggaeton",
            "bossa_nova",
            "dubstep_halftime",
            "uk_garage",
            "breakbeat",
        ]

    @classmethod
    def get_pattern(cls, name: str) -> DrumPattern:
        """Get a pattern by name."""
        patterns = {
            "trap_basic": cls.trap_basic,
            "trap_hihat_rolls": cls.trap_hihat_rolls,
            "boom_bap": cls.boom_bap,
            "lofi_hip_hop": cls.lofi_hip_hop,
            "house_basic": cls.house_basic,
            "deep_house": cls.deep_house,
            "tech_house": cls.tech_house,
            "techno_basic": cls.techno_basic,
            "industrial_techno": cls.industrial_techno,
            "dnb_basic": cls.dnb_basic,
            "jungle": cls.jungle,
            "rock_basic": cls.rock_basic,
            "pop_beat": cls.pop_beat,
            "reggaeton": cls.reggaeton,
            "bossa_nova": cls.bossa_nova,
            "dubstep_halftime": cls.dubstep_halftime,
            "uk_garage": cls.uk_garage,
            "breakbeat": cls.breakbeat,
        }
        if name not in patterns:
            raise KeyError(f"Unknown pattern: {name}. Available: {list(patterns.keys())}")
        return patterns[name]()
