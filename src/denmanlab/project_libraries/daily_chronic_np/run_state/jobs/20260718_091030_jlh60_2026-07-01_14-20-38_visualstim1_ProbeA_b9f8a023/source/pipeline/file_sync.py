from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncResult:
    source: Path
    destination: Path
    copied_files: int
    skipped_files: int
    copied_bytes: int


def copy_tree_update_only(
    source: str | Path,
    destination: str | Path,
    *,
    exclude_dir_prefixes: tuple[str, ...] = (),
    dry_run: bool = False,
) -> SyncResult:
    source_root = Path(source)
    destination_root = Path(destination)
    if not source_root.exists():
        raise FileNotFoundError(f"source folder does not exist: {source_root}")
    copied_files = 0
    skipped_files = 0
    copied_bytes = 0
    for source_file in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source_root)
        if _excluded(relative, exclude_dir_prefixes):
            continue
        destination_file = destination_root / relative
        size = source_file.stat().st_size
        if _target_is_current(source_file, destination_file):
            skipped_files += 1
            continue
        copied_files += 1
        copied_bytes += size
        if dry_run:
            continue
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
    return SyncResult(source_root, destination_root, copied_files, skipped_files, copied_bytes)


def _excluded(relative: Path, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return False
    return any(part.startswith(prefixes) for part in relative.parts)


def _target_is_current(source_file: Path, target_file: Path) -> bool:
    if not target_file.exists():
        return False
    source_stat = source_file.stat()
    target_stat = target_file.stat()
    if source_stat.st_size != target_stat.st_size:
        return False
    return target_stat.st_mtime >= source_stat.st_mtime - 1.0
