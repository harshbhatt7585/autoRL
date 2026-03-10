from __future__ import annotations

from pathlib import Path

REQUIRED_ENV_FILES: tuple[str, ...] = ("env.py", "train.py")


def missing_required_files(package_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ENV_FILES if not (package_dir / name).is_file()]


def is_complete_env_package(package_dir: Path) -> bool:
    return not missing_required_files(package_dir)
