from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backup import backup_folder_for_raw
from .config import load_config
from .file_sync import copy_tree_update_only
from .registry import read_registry, update_row


@dataclass(frozen=True)
class DerivedBackupResult:
    ok: bool
    raw_folder: Path
    backup_folder: Path
    source_folders: list[Path]
    copied_files: int
    skipped_files: int
    copied_bytes: int
    error: str = ""
    local_archive_folder: Path | None = None
    local_archive_status: str = ""
    local_archive_error: str = ""

    @property
    def status(self) -> str:
        return "verified" if self.ok else "failed"


def backup_derived_outputs_for_raw(
    raw_folder: str | Path,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> DerivedBackupResult:
    source_root = Path(raw_folder)
    target_root = backup_folder_for_raw(source_root, config)
    if not source_root.exists():
        return DerivedBackupResult(False, source_root, target_root, [], 0, 0, 0, f"raw folder does not exist: {source_root}")
    if not target_root.exists():
        return DerivedBackupResult(False, source_root, target_root, [], 0, 0, 0, f"backup folder does not exist: {target_root}")

    source_folders = find_derived_output_folders(source_root, config)
    if not source_folders:
        return DerivedBackupResult(True, source_root, target_root, [], 0, 0, 0)

    copied_files, skipped_files, copied_bytes, errors = _copy_derived_folders(
        source_root,
        target_root,
        source_folders,
        dry_run=dry_run,
    )
    local_archive_folder = _local_archive_folder_for_raw(source_root, config)
    local_archive_status = ""
    local_archive_error = ""
    if local_archive_folder is not None:
        local_archive_folder.mkdir(parents=True, exist_ok=True)
        _, _, _, local_errors = _copy_derived_folders(
            source_root,
            local_archive_folder,
            source_folders,
            dry_run=dry_run,
        )
        local_archive_status = "verified" if not local_errors else "failed"
        local_archive_error = "; ".join(local_errors)

    ok = not errors and local_archive_status != "failed"
    return DerivedBackupResult(
        ok,
        source_root,
        target_root,
        source_folders,
        copied_files,
        skipped_files,
        copied_bytes,
        "; ".join(errors),
        local_archive_folder,
        local_archive_status,
        local_archive_error,
    )


def backup_derived_outputs_for_row(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> DerivedBackupResult:
    return backup_derived_outputs_for_raw(row["raw_folder"], config, dry_run=dry_run)


def find_derived_output_folders(raw_folder: str | Path, config: dict[str, Any]) -> list[Path]:
    root = Path(raw_folder)
    derived_config = config.get("backup", {}).get("derived_outputs", {})
    exact_names = set(derived_config.get("include_dir_names") or ["nwb"])
    prefixes = tuple(derived_config.get("include_dir_prefixes") or ["spikeinterface_output"])

    selected: list[Path] = []
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts)):
        name = path.name
        if name not in exact_names and not name.startswith(prefixes):
            continue
        if any(_is_relative_to(path, parent) for parent in selected):
            continue
        selected.append(path)
    return sorted(selected)


def registry_fields_for_result(result: DerivedBackupResult) -> dict[str, str]:
    fields = {
        "derived_backup_status": result.status,
        "derived_backup_folder": str(result.backup_folder),
        "derived_backup_checked_at": datetime.now(timezone.utc).isoformat(),
        "derived_backup_error": result.error,
        "derived_backup_files": str(result.copied_files + result.skipped_files),
        "derived_backup_bytes": str(result.copied_bytes),
    }
    if result.local_archive_folder is not None:
        fields.update(
            {
                "derived_local_backup_status": result.local_archive_status,
                "derived_local_backup_folder": str(result.local_archive_folder),
                "derived_local_backup_checked_at": datetime.now(timezone.utc).isoformat(),
                "derived_local_backup_error": result.local_archive_error,
            }
        )
    return fields

def print_derived_backup_result(result: DerivedBackupResult, *, verbose: bool = False) -> None:
    print(f"source: {result.raw_folder}")
    print(f"backup: {result.backup_folder}")
    print(f"status: {result.status}")
    print(f"source_folders: {len(result.source_folders)}")
    print(f"copied_files: {result.copied_files}")
    print(f"skipped_files: {result.skipped_files}")
    print(f"copied_bytes: {result.copied_bytes}")
    if result.error:
        print(f"message: {result.error}")
    if result.local_archive_folder is not None:
        print(f"local_archive: {result.local_archive_folder}")
        print(f"local_archive_status: {result.local_archive_status}")
        if result.local_archive_error:
            print(f"local_archive_error: {result.local_archive_error}")
    if verbose:
        for folder in result.source_folders:
            print(f"derived_folder: {folder}")


def backup_derived_outputs_for_recording_session(
    config: dict[str, Any],
    *,
    session_id: str,
    dry_run: bool = False,
    update_registry: bool = False,
) -> DerivedBackupResult:
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        raise KeyError(session_id)
    result = backup_derived_outputs_for_row(selected, config, dry_run=dry_run)
    if update_registry:
        _update_registry_rows_for_raw(config, rows, selected.get("raw_folder", ""), result)
    return result


def _copy_derived_folders(
    source_root: Path,
    target_root: Path,
    source_folders: list[Path],
    *,
    dry_run: bool = False,
) -> tuple[int, int, int, list[str]]:
    copied_files = 0
    skipped_files = 0
    copied_bytes = 0
    errors: list[str] = []
    for source_folder in source_folders:
        try:
            relative = source_folder.resolve().relative_to(source_root.resolve())
        except ValueError:
            errors.append(f"derived folder is outside raw folder: {source_folder}")
            continue
        target_folder = target_root / relative
        result = copy_tree_update_only(source_folder, target_folder, dry_run=dry_run)
        copied_files += result.copied_files
        skipped_files += result.skipped_files
        copied_bytes += result.copied_bytes
    return copied_files, skipped_files, copied_bytes, errors


def _local_archive_folder_for_raw(raw_folder: Path, config: dict[str, Any]) -> Path | None:
    archive_root = config.get("backup", {}).get("derived_outputs", {}).get("local_archive_root")
    if not archive_root:
        return None
    raw_root = Path(config["project"]["raw_root"])
    try:
        relative = raw_folder.resolve().relative_to(raw_root.resolve())
    except ValueError:
        relative = Path(raw_folder.name)
    return Path(archive_root) / relative
def _update_registry_rows_for_raw(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    raw_folder: str,
    result: DerivedBackupResult,
) -> None:
    fields = registry_fields_for_result(result)
    raw_resolved = str(Path(raw_folder).resolve())
    for row in rows:
        if str(Path(row.get("raw_folder", "")).resolve()) != raw_resolved:
            continue
        row.update(fields)
        update_row(config["project"]["registry_csv"], row)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up derived pipeline outputs to the configured backup server.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", help="Any probe session_id from the target recording.")
    parser.add_argument("--raw-folder")
    parser.add_argument("--all", action="store_true", help="Back up derived outputs for every unique raw folder in the registry.")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_registry(config["project"]["registry_csv"])
    if args.all:
        raw_folders = sorted({row["raw_folder"] for row in rows if row.get("raw_folder")})
    elif args.session_id:
        selected = next((row for row in rows if row.get("session_id") == args.session_id), None)
        if selected is None:
            raise SystemExit(f"No registry row for session_id: {args.session_id}")
        raw_folders = [selected["raw_folder"]]
    elif args.raw_folder:
        raw_folders = [args.raw_folder]
    else:
        raise SystemExit("Pass --session-id SESSION_ID, --raw-folder RAW_FOLDER, or --all")

    by_raw: dict[str, DerivedBackupResult] = {}
    failed = False
    for raw_folder in raw_folders:
        result = backup_derived_outputs_for_raw(raw_folder, config, dry_run=args.dry_run)
        print_derived_backup_result(result, verbose=args.verbose)
        by_raw[str(Path(raw_folder).resolve())] = result
        failed = failed or not result.ok

    if args.update_registry:
        for raw_folder, result in by_raw.items():
            _update_registry_rows_for_raw(config, rows, raw_folder, result)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()





