from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .file_sync import SyncProgress, copy_tree_update_only
from .registry import read_registry, update_row


@dataclass(frozen=True)
class LocalRecordingArchiveResult:
    ok: bool
    source_folder: Path
    archive_folder: Path
    copied_files: int
    skipped_files: int
    copied_bytes: int
    error: str = ""

    @property
    def status(self) -> str:
        return "verified" if self.ok else "failed"


def archive_recording_for_row(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    progress_callback: Callable[[SyncProgress], None] | None = None,
) -> LocalRecordingArchiveResult:
    source = Path(row.get("local_raw_folder") or row.get("raw_folder") or "")
    archive_folder = _archive_folder_for_source(source, config)
    if archive_folder is None:
        return LocalRecordingArchiveResult(False, source, Path(""), 0, 0, 0, "local recording archive root is not configured")
    if not source.exists():
        return LocalRecordingArchiveResult(False, source, archive_folder, 0, 0, 0, f"source folder does not exist: {source}")
    try:
        result = copy_tree_update_only(
            source,
            archive_folder,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )
        return LocalRecordingArchiveResult(
            True,
            source,
            archive_folder,
            result.copied_files,
            result.skipped_files,
            result.copied_bytes,
        )
    except Exception as exc:
        return LocalRecordingArchiveResult(False, source, archive_folder, 0, 0, 0, str(exc))


def archive_recording_for_session(
    config: dict[str, Any],
    *,
    session_id: str,
    dry_run: bool = False,
    update_registry: bool = False,
    progress_callback: Callable[[SyncProgress], None] | None = None,
) -> LocalRecordingArchiveResult:
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        raise KeyError(session_id)
    result = archive_recording_for_row(
        selected,
        config,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    if update_registry:
        _update_recording_rows(config, rows, selected, registry_fields_for_result(result))
    return result


def registry_fields_for_result(result: LocalRecordingArchiveResult) -> dict[str, str]:
    return {
        "local_recording_archive_status": result.status,
        "local_recording_archive_folder": str(result.archive_folder),
        "local_recording_archive_checked_at": datetime.now(timezone.utc).isoformat(),
        "local_recording_archive_error": result.error,
        "local_recording_archive_files": str(result.copied_files + result.skipped_files),
        "local_recording_archive_bytes": str(result.copied_bytes),
    }


def _archive_folder_for_source(source: Path, config: dict[str, Any]) -> Path | None:
    archive_root = config.get("backup", {}).get("local_recording_archive", {}).get("root")
    if not archive_root:
        return None
    local_raw_root = Path(config.get("staging", {}).get("local_raw_root") or config["project"]["raw_root"])
    try:
        relative = source.resolve().relative_to(local_raw_root.resolve())
    except ValueError:
        relative = Path(source.name)
    return Path(archive_root) / relative


def _update_recording_rows(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    selected: dict[str, str],
    fields: dict[str, str],
) -> None:
    selected_server = str(Path(selected.get("server_raw_folder") or selected.get("raw_folder") or "").resolve())
    selected_raw = str(Path(selected.get("raw_folder") or "").resolve())
    for row in rows:
        row_server = str(Path(row.get("server_raw_folder") or row.get("raw_folder") or "").resolve())
        row_raw = str(Path(row.get("raw_folder") or "").resolve())
        if row_server != selected_server and row_raw != selected_raw:
            continue
        row.update(fields)
        update_row(config["project"]["registry_csv"], row)


def print_archive_result(result: LocalRecordingArchiveResult) -> None:
    print(f"source: {result.source_folder}")
    print(f"archive: {result.archive_folder}")
    print(f"status: {result.status}")
    print(f"copied_files: {result.copied_files}")
    print(f"skipped_files: {result.skipped_files}")
    print(f"copied_bytes: {result.copied_bytes}")
    if result.error:
        print(f"message: {result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a full local recording folder to local backup storage.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = archive_recording_for_session(
        config,
        session_id=args.session_id,
        dry_run=args.dry_run,
        update_registry=args.update_registry,
    )
    print_archive_result(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
