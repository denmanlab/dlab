from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .file_sync import SyncResult, copy_tree_update_only
from .machine import local_path_for_server_session
from .registry import read_registry, update_row


def stage_recording_for_row(row: dict[str, Any], config: dict[str, Any], *, dry_run: bool = False) -> tuple[SyncResult, dict[str, str]]:
    source = Path(row.get("server_raw_folder") or row.get("raw_folder") or "")
    destination = Path(row.get("local_raw_folder") or row.get("raw_folder") or local_path_for_server_session(source, config))
    exclude_prefixes = tuple(config.get("staging", {}).get("exclude_dir_prefixes") or ["spikeinterface_output", "nwb"])
    result = copy_tree_update_only(source, destination, exclude_dir_prefixes=exclude_prefixes, dry_run=dry_run)
    fields = {
        "raw_folder": str(destination),
        "server_raw_folder": str(source),
        "local_raw_folder": str(destination),
        "local_stage_status": "staged",
        "local_stage_folder": str(destination),
        "local_stage_checked_at": datetime.now(timezone.utc).isoformat(),
        "local_stage_error": "",
    }
    return result, fields


def stage_recording_for_session(config: dict[str, Any], *, session_id: str, dry_run: bool = False, update_registry: bool = False) -> SyncResult:
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        raise KeyError(session_id)
    result, fields = stage_recording_for_row(selected, config, dry_run=dry_run)
    if update_registry:
        _update_recording_rows(config, rows, selected, fields)
    return result


def stage_pending_recordings(config: dict[str, Any], *, dry_run: bool = False, update_registry: bool = False) -> list[SyncResult]:
    rows = read_registry(config["project"]["registry_csv"])
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") not in {"registered", "failed"}:
            continue
        key = row.get("server_raw_folder") or row.get("raw_folder")
        if key and row.get("local_stage_status") != "staged":
            selected.setdefault(key, row)
    results: list[SyncResult] = []
    for row in selected.values():
        result, fields = stage_recording_for_row(row, config, dry_run=dry_run)
        results.append(result)
        if update_registry:
            _update_recording_rows(config, rows, row, fields)
    return results


def _update_recording_rows(config: dict[str, Any], rows: list[dict[str, str]], selected: dict[str, str], fields: dict[str, str]) -> None:
    selected_server = str(Path(selected.get("server_raw_folder") or selected.get("raw_folder") or "").resolve())
    selected_raw = str(Path(selected.get("raw_folder") or "").resolve())
    for row in rows:
        row_server = str(Path(row.get("server_raw_folder") or row.get("raw_folder") or "").resolve())
        row_raw = str(Path(row.get("raw_folder") or "").resolve())
        if row_server != selected_server and row_raw != selected_raw:
            continue
        row.update(fields)
        update_row(config["project"]["registry_csv"], row)


def print_stage_result(result: SyncResult) -> None:
    print(f"source: {result.source}")
    print(f"destination: {result.destination}")
    print(f"copied_files: {result.copied_files}")
    print(f"skipped_files: {result.skipped_files}")
    print(f"copied_bytes: {result.copied_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage recordings from Synology/server storage to local analysis storage.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id")
    parser.add_argument("--all-pending", action="store_true")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.session_id:
        result = stage_recording_for_session(config, session_id=args.session_id, dry_run=args.dry_run, update_registry=args.update_registry)
        print_stage_result(result)
    elif args.all_pending:
        results = stage_pending_recordings(config, dry_run=args.dry_run, update_registry=args.update_registry)
        for result in results:
            print_stage_result(result)
    else:
        raise SystemExit("Pass --session-id SESSION_ID or --all-pending")


if __name__ == "__main__":
    main()
