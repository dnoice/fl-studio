"""Licensing Manager - Copyright, publishing splits, and licensing schemas.

Comprehensive licensing management for music releases, covering
songwriter/producer splits, mechanical licenses, sync licenses,
performance rights organization (PRO) registration, and
distribution agreements.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path


class RightsType(Enum):
    """Types of music rights."""

    SONGWRITING = "songwriting"  # Lyrics + melody composition
    PUBLISHING = "publishing"  # Publishing rights
    MASTER = "master"  # Sound recording ownership
    PERFORMANCE = "performance"  # Live/broadcast performance
    MECHANICAL = "mechanical"  # Reproduction rights
    SYNC = "sync"  # Synchronization (film/TV/ads)
    DIGITAL = "digital"  # Digital distribution
    NEIGHBORING = "neighboring"  # Performer's rights in recording


class PRO(Enum):
    """Performance Rights Organizations."""

    ASCAP = "ASCAP"
    BMI = "BMI"
    SESAC = "SESAC"
    GMR = "GMR"
    SOCAN = "SOCAN"  # Canada
    PRS = "PRS"  # UK
    GEMA = "GEMA"  # Germany
    SACEM = "SACEM"  # France
    JASRAC = "JASRAC"  # Japan
    APRA_AMCOS = "APRA AMCOS"  # Australia
    OTHER = "Other"


class LicenseType(Enum):
    """Common license types for music."""

    ALL_RIGHTS_RESERVED = "All Rights Reserved"
    CC_BY = "CC BY 4.0"
    CC_BY_SA = "CC BY-SA 4.0"
    CC_BY_NC = "CC BY-NC 4.0"
    CC_BY_NC_SA = "CC BY-NC-SA 4.0"
    CC_BY_ND = "CC BY-ND 4.0"
    CC_BY_NC_ND = "CC BY-NC-ND 4.0"
    CC0 = "CC0 (Public Domain)"
    CUSTOM = "Custom License"


@dataclass
class WriterSplit:
    """A single writer/contributor's ownership split."""

    name: str
    role: str = "songwriter"  # songwriter, composer, lyricist, producer, arranger
    share_percent: float = 0.0  # 0-100
    pro: str = ""  # PRO affiliation (ASCAP, BMI, etc.)
    ipi_number: str = ""  # Interested Parties Information number
    publisher: str = ""
    publisher_share: float = 0.0  # Publisher's cut of this writer's share
    email: str = ""
    notes: str = ""

    def validate(self) -> list[str]:
        issues = []
        if not self.name:
            issues.append("Writer name is required")
        if self.share_percent < 0 or self.share_percent > 100:
            issues.append(f"Invalid share: {self.share_percent}%")
        if self.publisher_share < 0 or self.publisher_share > 100:
            issues.append(f"Invalid publisher share: {self.publisher_share}%")
        return issues


@dataclass
class PublishingInfo:
    """Publishing information for a song."""

    publisher_name: str = ""
    publisher_pro: str = ""
    publisher_ipi: str = ""
    admin_share: float = 0.0  # Publisher's admin percentage
    territory: str = "Worldwide"
    sub_publishers: dict[str, str] = field(default_factory=dict)  # territory -> sub-publisher


@dataclass
class LicenseInfo:
    """Copyright and license information for a track or album."""

    copyright_owner: str = ""
    copyright_year: int = 0
    copyright_notice: str = ""  # e.g., "(C) 2025 Artist Name"
    phonographic_copyright: str = ""  # e.g., "(P) 2025 Label Name"
    license_type: str = "All Rights Reserved"
    custom_license_text: str = ""
    territory: str = "Worldwide"
    exclusive: bool = True
    duration_years: int = 0  # 0 = perpetual

    @property
    def copyright_string(self) -> str:
        """Generate standard copyright notice."""
        if self.copyright_notice:
            return self.copyright_notice
        year = self.copyright_year or date.today().year
        return f"\u00a9 {year} {self.copyright_owner}"

    @property
    def phonographic_string(self) -> str:
        """Generate phonographic copyright notice."""
        if self.phonographic_copyright:
            return self.phonographic_copyright
        year = self.copyright_year or date.today().year
        return f"\u2117 {year} {self.copyright_owner}"


@dataclass
class MechanicalLicense:
    """Mechanical license for reproduction of a composition."""

    song_title: str = ""
    songwriter: str = ""
    publisher: str = ""
    licensee: str = ""  # Who is licensed to reproduce
    license_number: str = ""
    rate_type: str = "statutory"  # statutory, negotiated
    rate_per_unit: float = 0.0
    territory: str = "United States"
    medium: str = "digital"  # physical, digital, both
    term_start: str = ""  # ISO date
    term_end: str = ""  # ISO date or "perpetual"
    max_units: int = 0  # 0 = unlimited
    advance_paid: float = 0.0
    notes: str = ""

    # Current US statutory mechanical rate (2024-2027)
    STATUTORY_RATE_US = 0.12  # 12 cents per unit (songs <= 5 min)
    STATUTORY_RATE_LONG = 0.0231  # 2.31 cents per minute (songs > 5 min)


@dataclass
class SyncLicense:
    """Synchronization license for use in visual media."""

    song_title: str = ""
    master_owner: str = ""
    publisher: str = ""
    licensee: str = ""
    project_name: str = ""
    project_type: str = ""  # film, TV, commercial, video game, web
    usage: str = ""  # background, featured, theme, trailer
    duration_s: int = 0  # How much of the song is used
    territory: str = "Worldwide"
    term: str = "perpetual"
    fee: float = 0.0
    master_fee: float = 0.0
    publishing_fee: float = 0.0
    exclusivity: str = "non-exclusive"
    notes: str = ""


@dataclass
class SongRegistration:
    """Complete registration record for a song with all rights holders."""

    title: str = ""
    alternate_titles: list[str] = field(default_factory=list)
    iswc: str = ""  # International Standard Musical Work Code
    writers: list[WriterSplit] = field(default_factory=list)
    publishing: PublishingInfo = field(default_factory=PublishingInfo)
    license_info: LicenseInfo = field(default_factory=LicenseInfo)
    mechanical_licenses: list[MechanicalLicense] = field(default_factory=list)
    sync_licenses: list[SyncLicense] = field(default_factory=list)
    isrc: str = ""
    language: str = "EN"
    original_release_date: str = ""
    genre: str = ""
    notes: str = ""

    def total_writer_share(self) -> float:
        """Sum of all writer shares (should equal 100%)."""
        return sum(w.share_percent for w in self.writers)

    def validate(self) -> list[str]:
        """Validate the song registration."""
        issues = []

        if not self.title:
            issues.append("ERROR: Song title is required")

        # Writer splits
        total = self.total_writer_share()
        if self.writers and abs(total - 100.0) > 0.01:
            issues.append(f"ERROR: Writer shares total {total:.1f}% (must be 100%)")

        if not self.writers:
            issues.append("WARN: No writers/composers registered")

        for writer in self.writers:
            for issue in writer.validate():
                issues.append(f"Writer '{writer.name}': {issue}")

        if not self.license_info.copyright_owner:
            issues.append("WARN: Copyright owner not set")

        if self.iswc and not self._validate_iswc(self.iswc):
            issues.append(f"ERROR: Invalid ISWC format: {self.iswc}")

        return issues

    @staticmethod
    def _validate_iswc(iswc: str) -> bool:
        """Validate ISWC format: T-NNN.NNN.NNN-C."""
        clean = iswc.replace("-", "").replace(".", "")
        return len(clean) == 11 and clean[0] == "T" and clean[1:].isdigit()


class LicensingManager:
    """Manage licensing for an entire catalog of songs.

    Tracks writer splits, publishing deals, mechanical and sync licenses,
    and generates registration documents.
    """

    def __init__(self):
        self._registrations: dict[str, SongRegistration] = {}

    def register_song(self, registration: SongRegistration) -> str:
        """Register a song with its rights information.

        Returns:
            Registration key.
        """
        key = registration.title.strip()
        self._registrations[key] = registration
        return key

    def get_song(self, title: str) -> SongRegistration | None:
        return self._registrations.get(title)

    @property
    def all_songs(self) -> list[SongRegistration]:
        return list(self._registrations.values())

    def quick_register(
        self, title: str, writers: list[dict], copyright_owner: str = "", isrc: str = ""
    ) -> SongRegistration:
        """Quick registration with minimal info.

        Args:
            title: Song title.
            writers: List of dicts with keys: name, role, share_percent, pro.
            copyright_owner: Copyright holder name.
            isrc: ISRC code.

        Returns:
            SongRegistration object.

        Example:
            manager.quick_register("My Song", [
                {"name": "John Doe", "role": "songwriter", "share_percent": 50, "pro": "ASCAP"},
                {"name": "Jane Smith", "role": "producer", "share_percent": 50, "pro": "BMI"},
            ], copyright_owner="John Doe")
        """
        reg = SongRegistration(title=title, isrc=isrc)
        reg.license_info.copyright_owner = copyright_owner or (
            writers[0]["name"] if writers else ""
        )
        reg.license_info.copyright_year = date.today().year

        for w in writers:
            reg.writers.append(
                WriterSplit(
                    name=w.get("name", ""),
                    role=w.get("role", "songwriter"),
                    share_percent=w.get("share_percent", 0.0),
                    pro=w.get("pro", ""),
                    publisher=w.get("publisher", ""),
                    ipi_number=w.get("ipi_number", ""),
                )
            )

        self._registrations[title] = reg
        return reg

    def add_mechanical_license(self, song_title: str, license: MechanicalLicense) -> None:
        """Add a mechanical license to a registered song."""
        reg = self._registrations.get(song_title)
        if reg:
            reg.mechanical_licenses.append(license)

    def add_sync_license(self, song_title: str, license: SyncLicense) -> None:
        """Add a sync license to a registered song."""
        reg = self._registrations.get(song_title)
        if reg:
            reg.sync_licenses.append(license)

    # ─── Financial ───

    def calculate_royalty_split(self, song_title: str, total_revenue: float) -> dict[str, float]:
        """Calculate royalty distribution for a song.

        Args:
            song_title: Registered song title.
            total_revenue: Total revenue to split.

        Returns:
            Dict mapping writer names to their payment amounts.
        """
        reg = self._registrations.get(song_title)
        if not reg or not reg.writers:
            return {}

        splits: dict[str, float] = {}
        for writer in reg.writers:
            writer_revenue = total_revenue * (writer.share_percent / 100.0)

            # Account for publisher share
            if writer.publisher_share > 0:
                pub_cut = writer_revenue * (writer.publisher_share / 100.0)
                writer_net = writer_revenue - pub_cut
                pub_key = f"{writer.publisher} (pub for {writer.name})"
                splits[pub_key] = splits.get(pub_key, 0) + pub_cut
                splits[writer.name] = splits.get(writer.name, 0) + writer_net
            else:
                splits[writer.name] = splits.get(writer.name, 0) + writer_revenue

        return {k: round(v, 2) for k, v in splits.items()}

    # ─── Validation ───

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all registrations.

        Returns:
            Dict mapping song titles to their validation issues.
        """
        results = {}
        for title, reg in self._registrations.items():
            issues = reg.validate()
            if issues:
                results[title] = issues
        return results

    def find_unregistered_writers(self) -> list[str]:
        """Find writers without PRO affiliation."""
        unregistered = []
        seen = set()
        for reg in self._registrations.values():
            for writer in reg.writers:
                if writer.name not in seen and not writer.pro:
                    unregistered.append(writer.name)
                    seen.add(writer.name)
        return unregistered

    # ─── Export / Reports ───

    def save_database(self, path: str) -> None:
        """Save all registrations to JSON."""
        data = {}
        for title, reg in self._registrations.items():
            data[title] = {
                "title": reg.title,
                "alternate_titles": reg.alternate_titles,
                "iswc": reg.iswc,
                "isrc": reg.isrc,
                "language": reg.language,
                "genre": reg.genre,
                "original_release_date": reg.original_release_date,
                "writers": [
                    {
                        "name": w.name,
                        "role": w.role,
                        "share_percent": w.share_percent,
                        "pro": w.pro,
                        "ipi_number": w.ipi_number,
                        "publisher": w.publisher,
                        "publisher_share": w.publisher_share,
                        "email": w.email,
                    }
                    for w in reg.writers
                ],
                "publishing": {
                    "publisher_name": reg.publishing.publisher_name,
                    "publisher_pro": reg.publishing.publisher_pro,
                    "publisher_ipi": reg.publishing.publisher_ipi,
                    "admin_share": reg.publishing.admin_share,
                    "territory": reg.publishing.territory,
                },
                "license": {
                    "copyright_owner": reg.license_info.copyright_owner,
                    "copyright_year": reg.license_info.copyright_year,
                    "license_type": reg.license_info.license_type,
                    "territory": reg.license_info.territory,
                    "exclusive": reg.license_info.exclusive,
                },
                "mechanical_licenses": [
                    {
                        "licensee": ml.licensee,
                        "rate_type": ml.rate_type,
                        "rate_per_unit": ml.rate_per_unit,
                        "territory": ml.territory,
                        "medium": ml.medium,
                    }
                    for ml in reg.mechanical_licenses
                ],
                "sync_licenses": [
                    {
                        "licensee": sl.licensee,
                        "project_name": sl.project_name,
                        "project_type": sl.project_type,
                        "usage": sl.usage,
                        "fee": sl.fee,
                        "territory": sl.territory,
                    }
                    for sl in reg.sync_licenses
                ],
            }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load_database(cls, path: str) -> "LicensingManager":
        """Load registrations from JSON."""
        manager = cls()
        data = json.loads(Path(path).read_text())

        for title, reg_data in data.items():
            reg = SongRegistration(
                title=reg_data.get("title", title),
                iswc=reg_data.get("iswc", ""),
                isrc=reg_data.get("isrc", ""),
                language=reg_data.get("language", "EN"),
                genre=reg_data.get("genre", ""),
                original_release_date=reg_data.get("original_release_date", ""),
                alternate_titles=reg_data.get("alternate_titles", []),
            )

            for w_data in reg_data.get("writers", []):
                reg.writers.append(
                    WriterSplit(
                        name=w_data.get("name", ""),
                        role=w_data.get("role", "songwriter"),
                        share_percent=w_data.get("share_percent", 0),
                        pro=w_data.get("pro", ""),
                        ipi_number=w_data.get("ipi_number", ""),
                        publisher=w_data.get("publisher", ""),
                        publisher_share=w_data.get("publisher_share", 0),
                        email=w_data.get("email", ""),
                    )
                )

            pub_data = reg_data.get("publishing", {})
            reg.publishing = PublishingInfo(
                publisher_name=pub_data.get("publisher_name", ""),
                publisher_pro=pub_data.get("publisher_pro", ""),
                publisher_ipi=pub_data.get("publisher_ipi", ""),
                admin_share=pub_data.get("admin_share", 0),
                territory=pub_data.get("territory", "Worldwide"),
            )

            lic_data = reg_data.get("license", {})
            reg.license_info = LicenseInfo(
                copyright_owner=lic_data.get("copyright_owner", ""),
                copyright_year=lic_data.get("copyright_year", 0),
                license_type=lic_data.get("license_type", "All Rights Reserved"),
                territory=lic_data.get("territory", "Worldwide"),
                exclusive=lic_data.get("exclusive", True),
            )

            manager._registrations[title] = reg

        return manager

    def generate_split_sheet(self, song_title: str) -> str:
        """Generate a text split sheet for a song.

        A split sheet is a legal document that identifies each
        contributor's ownership percentage.
        """
        reg = self._registrations.get(song_title)
        if not reg:
            return f"Song '{song_title}' not found."

        lines = [
            "=" * 60,
            "SPLIT SHEET / SONGWRITER AGREEMENT",
            "=" * 60,
            "",
            f"Song Title:     {reg.title}",
        ]
        if reg.alternate_titles:
            lines.append(f"Alt Titles:     {', '.join(reg.alternate_titles)}")
        if reg.iswc:
            lines.append(f"ISWC:           {reg.iswc}")
        if reg.isrc:
            lines.append(f"ISRC:           {reg.isrc}")
        lines.append(f"Date:           {date.today().isoformat()}")
        lines.append("")

        lines.append("WRITERS / CONTRIBUTORS:")
        lines.append("-" * 60)
        lines.append(f"{'Name':<25} {'Role':<15} {'Share':<8} {'PRO':<8} {'Publisher'}")
        lines.append("-" * 60)

        for w in reg.writers:
            pub = w.publisher or "Self-published"
            lines.append(f"{w.name:<25} {w.role:<15} {w.share_percent:>5.1f}%  {w.pro:<8} {pub}")
            if w.ipi_number:
                lines.append(f"  IPI: {w.ipi_number}")

        total = reg.total_writer_share()
        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<25} {'':15} {total:>5.1f}%")

        if abs(total - 100.0) > 0.01:
            lines.append("  *** WARNING: Shares do not total 100% ***")

        lines.extend(
            [
                "",
                "COPYRIGHT:",
                f"  {reg.license_info.copyright_string}",
                f"  {reg.license_info.phonographic_string}",
                f"  License: {reg.license_info.license_type}",
                f"  Territory: {reg.license_info.territory}",
                "",
                "SIGNATURES:",
                "",
            ]
        )

        for w in reg.writers:
            lines.extend(
                [
                    f"  {w.name}",
                    "  Signature: _________________________ Date: ___________",
                    "",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_pro_registration(self, song_title: str) -> str:
        """Generate PRO registration information."""
        reg = self._registrations.get(song_title)
        if not reg:
            return f"Song '{song_title}' not found."

        lines = [
            "=== PRO Registration Info ===",
            f"Song: {reg.title}",
            f"ISWC: {reg.iswc or 'Pending'}",
            "",
        ]

        # Group writers by PRO
        by_pro: dict[str, list[WriterSplit]] = {}
        for w in reg.writers:
            pro = w.pro or "Unaffiliated"
            by_pro.setdefault(pro, []).append(w)

        for pro, writers in by_pro.items():
            lines.append(f"--- {pro} ---")
            for w in writers:
                lines.append(f"  {w.name} ({w.role}): {w.share_percent}%")
                if w.ipi_number:
                    lines.append(f"    IPI: {w.ipi_number}")
                if w.publisher:
                    lines.append(f"    Publisher: {w.publisher}")
            lines.append("")

        return "\n".join(lines)

    def catalog_summary(self) -> str:
        """Generate a summary of the entire catalog."""
        lines = [
            "=== Catalog Summary ===",
            f"Total Songs: {len(self._registrations)}",
            "",
        ]

        # Collect stats
        all_writers = set()
        all_pros = set()
        total_sync = 0
        total_mech = 0

        for reg in self._registrations.values():
            for w in reg.writers:
                all_writers.add(w.name)
                if w.pro:
                    all_pros.add(w.pro)
            total_sync += len(reg.sync_licenses)
            total_mech += len(reg.mechanical_licenses)

        lines.append(f"Unique Writers: {len(all_writers)}")
        lines.append(f"PROs: {', '.join(sorted(all_pros)) or 'None'}")
        lines.append(f"Active Sync Licenses: {total_sync}")
        lines.append(f"Active Mechanical Licenses: {total_mech}")
        lines.append("")

        for title, reg in self._registrations.items():
            writers_str = ", ".join(f"{w.name} ({w.share_percent}%)" for w in reg.writers)
            lines.append(f"  {title}")
            lines.append(f"    Writers: {writers_str}")
            lines.append(f"    {reg.license_info.copyright_string}")

        return "\n".join(lines)
