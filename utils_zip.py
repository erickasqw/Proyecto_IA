"""Utilities for locating and safely extracting the FER datasets."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data_raw"

DATASET_ARCHIVES = {
    "fer": ("archive_7_fer.zip", "archive (7).zip"),
    "ckplus": ("archive_8_ckplus.zip", "archive (8).zip"),
    "raf": ("archive_9_raf.zip", "archive (9).zip"),
}


def _find_archive(names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = BASE_DIR / name
        if path.is_file():
            return path
    return None


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if os.path.commonpath((str(destination), str(member_path))) != str(destination):
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zip_file.extractall(destination)


def extract_datasets(force: bool = False) -> dict[str, Path]:
    """Extract all available archives once and return their dataset directories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    missing: list[str] = []

    for dataset_name, archive_names in DATASET_ARCHIVES.items():
        archive = _find_archive(archive_names)
        if archive is None:
            missing.append(" or ".join(archive_names))
            continue

        destination = RAW_DIR / dataset_name
        marker = destination / ".extracted"
        if force and destination.exists():
            shutil.rmtree(destination)
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, destination)
            marker.touch()
        extracted[dataset_name] = destination

    if missing:
        raise FileNotFoundError(
            "No se encontraron estos archivos ZIP: " + ", ".join(missing)
        )
    return extracted


if __name__ == "__main__":
    print({name: str(path) for name, path in extract_datasets().items()})