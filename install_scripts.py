"""Install FL Studio scripts to the correct directories.

Copies piano roll scripts and Edison scripts from this project
into FL Studio's script directories.

Usage:
    python install_scripts.py [--fl-path "C:\\Program Files\\Image-Line\\FL Studio 2025"]
"""

import shutil
import sys
from pathlib import Path

DEFAULT_FL_PATH = r"C:\Program Files\Image-Line\FL Studio 2025"


def install_scripts(fl_path: str = DEFAULT_FL_PATH, dry_run: bool = False):
    """Install FL Studio scripts to their correct locations."""

    fl_root = Path(fl_path).resolve()
    if not fl_root.exists():
        print(f"  [ERROR] FL Studio path does not exist: {fl_root}")
        return
    project_root = Path(__file__).parent

    # Script directories
    targets = {
        "piano_roll": fl_root / "System" / "Config" / "Piano roll scripts",
        "edison": fl_root / "System" / "Config" / "Audio scripts",
    }

    sources = {
        "piano_roll": project_root / "fl_scripts" / "piano_roll",
        "edison": project_root / "fl_scripts" / "edison",
    }

    installed = 0
    errors = []

    for script_type in ("piano_roll", "edison"):
        src_dir = sources[script_type]
        dst_dir = targets[script_type]

        if not src_dir.exists():
            print(f"  [SKIP] Source not found: {src_dir}")
            continue

        if not dst_dir.exists():
            print(f"  [WARN] Target directory not found: {dst_dir}")
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Installing {script_type} scripts ---")
        print(f"  From: {src_dir}")
        print(f"  To:   {dst_dir}")

        for script_file in src_dir.glob("*.pyscript"):
            dst = dst_dir / script_file.name
            print(f"  {'[DRY RUN] ' if dry_run else ''}Copying: {script_file.name}")

            if not dry_run:
                try:
                    shutil.copy2(str(script_file), str(dst))
                    installed += 1
                except Exception as e:
                    errors.append(f"{script_file.name}: {e}")
            else:
                installed += 1

    # MIDI controller scripts
    midi_src = project_root / "fl_scripts" / "midi_controller"
    midi_dst = fl_root / "System" / "Hardware specific"

    if midi_src.exists():
        print("\n--- Installing MIDI controller scripts ---")
        for controller_dir in midi_src.iterdir():
            if controller_dir.is_dir():
                dst = midi_dst / controller_dir.name
                print(f"  {'[DRY RUN] ' if dry_run else ''}Copying: {controller_dir.name}/")
                if not dry_run:
                    try:
                        if dst.exists():
                            shutil.rmtree(str(dst))
                        shutil.copytree(str(controller_dir), str(dst))
                        installed += 1
                    except Exception as e:
                        errors.append(f"{controller_dir.name}: {e}")

    print(f"\n{'='*40}")
    print(f"Installed: {installed} items")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err}")
    else:
        print("No errors.")

    if not dry_run:
        print("\nRestart FL Studio to load the new scripts.")
        print("Piano roll scripts: Tools > Piano roll > Scripts")
        print("Edison scripts: Tools menu in Edison")


if __name__ == "__main__":
    fl_path = DEFAULT_FL_PATH
    dry_run = False

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--fl-path" and i < len(sys.argv) - 1:
            fl_path = sys.argv[i + 1]
        elif arg == "--dry-run":
            dry_run = True

    print(f"FL Studio path: {fl_path}")
    print(f"Dry run: {dry_run}\n")
    install_scripts(fl_path, dry_run)
