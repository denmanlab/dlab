from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backup import backup_folder_for_raw, verify_raw_backup
from .config import load_config
from .file_sync import SyncResult, copy_tree_update_only
from .registry import read_registry, update_row


def backup_raw_for_folder(raw_folder: str | Path, config: dict[str, Any], *, dry_run: bool = False) -> tuple[SyncResult, dict[str, str]]:
    source = Path(raw_folder)
    destination = backup_folder_for_raw(source, config)
    exclude_prefixes = tuple(config.get("backup", {}).get("exclude_dir_prefixes", ["spikeinterface_output", "nwb"]))
    result = copy_tree_update_only(source, destination, exclude_dir_prefixes=exclude_prefixes, dry_run=dry_run)
    verification = verify_raw_backup(source, config) if not dry_run else None
    status = verification.status if verification is not None else "dry_run"
    error = verification.error if verification is not None else ""
    fields = {
        "backup_status": status,
        "backup_folder": str(destination),
        "backup_checked_at": datetime.now(timezone.utc).isoformat(),
        "backup_error": error,
    }
    return result, fields


def backup_raw_for_session(config: dict[str, Any], *, session_id: str, dry_run: bool = False, update_registry: bool = False) -> SyncResult:
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        raise KeyError(session_id)
    result, fields = backup_raw_for_folder(selected["raw_folder"], config, dry_run=dry_run)
    if update_registry:
        _update_recording_rows(config, rows, selected.get("raw_folder", ""), fields)
    return result


def backup_all_raw(config: dict[str, Any], *, dry_run: bool = False, update_registry: bool = False) -> list[SyncResult]:
    rows = read_registry(config["project"]["registry_csv"])
    raw_folders = sorted({row["raw_folder"] for row in rows if row.get("raw_folder") and Path(row["raw_folder"]).exists()})
    results: list[SyncResult] = []
    for raw_folder in raw_folders:
        result, fields = backup_raw_for_folder(raw_folder, config, dry_run=dry_run)
        results.append(result)
        if update_registry:
            _update_recording_rows(config, rows, raw_folder, fields)
    return results


def _update_recording_rows(config: dict[str, Any], rows: list[dict[str, str]], raw_folder: str, fields: dict[str, str]) -> None:
    raw_resolved = str(Path(raw_folder).resolve())
    for row in rows:
        if str(Path(row.get("raw_folder", "")).resolve()) != raw_resolved:
            continue
        row.update(fields)
        update_row(config["project"]["registry_csv"], row)


def print_raw_backup_result(result: SyncResult) -> None:
    print(f"source: {result.source}")
    print(f"backup: {result.destination}")
    print(f"copied_files: {result.copied_files}")
    print(f"skipped_files: {result.skipped_files}")
    print(f"copied_bytes: {result.copied_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy raw acquisition recordings to the configured Synology/server backup root.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id")
    parser.add_argument("--raw-folder")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.session_id:
        result = backup_raw_for_session(config, session_id=args.session_id, dry_run=args.dry_run, update_registry=args.update_registry)
        print_raw_backup_result(result)
    elif args.raw_folder:
        result, _ = backup_raw_for_folder(args.raw_folder, config, dry_run=args.dry_run)
        print_raw_backup_result(result)
    elif args.all:
        for result in backup_all_raw(config, dry_run=args.dry_run, update_registry=args.update_registry):
            print_raw_backup_result(result)
    else:
        raise SystemExit("Pass --session-id SESSION_ID, --raw-folder RAW_FOLDER, or --all")


if __name__ == "__main__":
    main()

