from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class SyncResult:
    source: Path
    destination: Path
    copied_files: int
    skipped_files: int
    copied_bytes: int


@dataclass(frozen=True)
class SyncProgress:
    relative_path: Path
    file_index: int
    total_files: int
    file_bytes_copied: int
    file_bytes_total: int
    completed_bytes: int
    total_bytes: int
    status: str

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 100.0
        return min(100.0, 100.0 * self.completed_bytes / self.total_bytes)


def copy_tree_update_only(
    source: str | Path,
    destination: str | Path,
    *,
    exclude_dir_prefixes: tuple[str, ...] = (),
    dry_run: bool = False,
    progress_callback: Callable[[SyncProgress], None] | None = None,
    copy_chunk_bytes: int = 64 * 1024 * 1024,
) -> SyncResult:
    source_root = Path(source)
    destination_root = Path(destination)
    if not source_root.exists():
        raise FileNotFoundError(f"source folder does not exist: {source_root}")
    source_files = [
        path
        for path in sorted(path for path in source_root.rglob("*") if path.is_file())
        if not _excluded(path.relative_to(source_root), exclude_dir_prefixes)
    ]
    file_sizes = [path.stat().st_size for path in source_files]
    total_bytes = sum(file_sizes)
    completed_bytes = 0
    copied_files = 0
    skipped_files = 0
    copied_bytes = 0
    for file_index, (source_file, size) in enumerate(zip(source_files, file_sizes), start=1):
        relative = source_file.relative_to(source_root)
        destination_file = destination_root / relative
        if _target_is_current(source_file, destination_file):
            skipped_files += 1
            completed_bytes += size
            _report_progress(
                progress_callback,
                relative=relative,
                file_index=file_index,
                total_files=len(source_files),
                file_bytes_copied=size,
                file_bytes_total=size,
                completed_bytes=completed_bytes,
                total_bytes=total_bytes,
                status="skipped",
            )
            continue
        copied_files += 1
        copied_bytes += size
        if dry_run:
            completed_bytes += size
            _report_progress(
                progress_callback,
                relative=relative,
                file_index=file_index,
                total_files=len(source_files),
                file_bytes_copied=size,
                file_bytes_total=size,
                completed_bytes=completed_bytes,
                total_bytes=total_bytes,
                status="dry_run",
            )
            continue
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if progress_callback is None:
            shutil.copy2(source_file, destination_file)
            completed_bytes += size
            continue
        file_bytes_copied = 0
        with source_file.open("rb") as source_handle, destination_file.open("wb") as destination_handle:
            while True:
                chunk = source_handle.read(max(1024 * 1024, int(copy_chunk_bytes)))
                if not chunk:
                    break
                destination_handle.write(chunk)
                file_bytes_copied += len(chunk)
                _report_progress(
                    progress_callback,
                    relative=relative,
                    file_index=file_index,
                    total_files=len(source_files),
                    file_bytes_copied=file_bytes_copied,
                    file_bytes_total=size,
                    completed_bytes=completed_bytes + file_bytes_copied,
                    total_bytes=total_bytes,
                    status="copying",
                )
        shutil.copystat(source_file, destination_file)
        completed_bytes += size
        _report_progress(
            progress_callback,
            relative=relative,
            file_index=file_index,
            total_files=len(source_files),
            file_bytes_copied=size,
            file_bytes_total=size,
            completed_bytes=completed_bytes,
            total_bytes=total_bytes,
            status="complete",
        )
    return SyncResult(source_root, destination_root, copied_files, skipped_files, copied_bytes)


def _report_progress(
    callback: Callable[[SyncProgress], None] | None,
    *,
    relative: Path,
    file_index: int,
    total_files: int,
    file_bytes_copied: int,
    file_bytes_total: int,
    completed_bytes: int,
    total_bytes: int,
    status: str,
) -> None:
    if callback is None:
        return
    callback(
        SyncProgress(
            relative_path=relative,
            file_index=file_index,
            total_files=total_files,
            file_bytes_copied=file_bytes_copied,
            file_bytes_total=file_bytes_total,
            completed_bytes=completed_bytes,
            total_bytes=total_bytes,
            status=status,
        )
    )


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
