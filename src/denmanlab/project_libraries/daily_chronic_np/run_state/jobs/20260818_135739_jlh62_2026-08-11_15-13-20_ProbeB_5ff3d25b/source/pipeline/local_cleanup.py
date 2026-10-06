from __future__ import annotations

import argparse
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .registry import read_registry, update_row


ACTIVE_STATUSES = {"preprocessing", "sorting", "qc_running", "running", "queued"}
ALLOWED_FINAL_STATUSES = {"complete", "skipped"}


@dataclass(frozen=True)
class LocalCleanupResult:
    eligible: bool
    deleted: bool
    source_folder: Path
    synology_folder: Path
    local_archive_folder: Path
    checked_files: int
    checked_bytes: int
    errors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.deleted:
            return "offloaded"
        return "verified" if self.eligible else "blocked"


def assess_recording_cleanup(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    source_folder: str | Path,
) -> LocalCleanupResult:
    source = Path(source_folder)
    matching = _rows_for_source(rows, source)
    synology = _synology_folder(matching, source, config)
    local_archive = _local_archive_folder(source, matching, config)
    errors: list[str] = []

    if not matching:
        errors.append("recording is not present in the registry")
    if not source.exists():
        errors.append(f"local recording folder does not exist: {source}")
    _validate_local_source(source, config, errors)
    statuses = {row.get("status", "") for row in matching}
    if not any(status == "complete" for status in statuses):
        errors.append("recording has no completed probe/block")
    unfinished = sorted(status for status in statuses if status not in ALLOWED_FINAL_STATUSES)
    if unfinished:
        errors.append(f"recording has unfinished rows: {', '.join(unfinished)}")
    if any(
        row.get("status") in ACTIVE_STATUSES
        or row.get("queue_status") in ACTIVE_STATUSES
        or row.get("preprocess_status") == "running"
        or row.get("sort_status") == "running"
        or row.get("qc_status") == "running"
        for row in matching
    ):
        errors.append("recording has an active or queued pipeline job")
    if not synology:
        errors.append("Synology recording folder is not configured")
    if not local_archive:
        errors.append("D: full-recording archive folder is not configured")

    checked_files = 0
    checked_bytes = 0
    if not errors:
        source_files = _file_size_map(source)
        checked_files = len(source_files)
        checked_bytes = sum(source_files.values())
        if not source_files:
            errors.append("local recording folder contains no files")
        else:
            errors.extend(_tree_mismatches(source_files, synology, "Synology"))
            errors.extend(_tree_mismatches(source_files, local_archive, "D: archive"))

    return LocalCleanupResult(
        eligible=not errors,
        deleted=False,
        source_folder=source,
        synology_folder=synology,
        local_archive_folder=local_archive,
        checked_files=checked_files,
        checked_bytes=checked_bytes,
        errors=tuple(errors),
    )


def cleanup_recording_for_session(
    config: dict[str, Any],
    *,
    session_id: str,
    delete_local: bool = False,
    update_registry: bool = False,
) -> LocalCleanupResult:
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        raise KeyError(session_id)
    source = selected.get("local_raw_folder") or selected.get("raw_folder") or ""
    result = assess_recording_cleanup(rows, config, source_folder=source)
    if delete_local:
        if not result.eligible:
            raise RuntimeError("Local cleanup verification failed: " + "; ".join(result.errors))
        _delete_verified_source(result.source_folder, config)
        result = LocalCleanupResult(
            eligible=True,
            deleted=True,
            source_folder=result.source_folder,
            synology_folder=result.synology_folder,
            local_archive_folder=result.local_archive_folder,
            checked_files=result.checked_files,
            checked_bytes=result.checked_bytes,
        )
    if update_registry:
        _update_recording_rows(rows, config, result)
    return result


def cleanup_all_eligible(
    config: dict[str, Any],
    *,
    delete_local: bool = False,
    update_registry: bool = False,
) -> list[LocalCleanupResult]:
    rows = read_registry(config["project"]["registry_csv"])
    sources = sorted(
        {
            row.get("local_raw_folder") or row.get("raw_folder") or ""
            for row in rows
            if row.get("local_raw_folder") or row.get("raw_folder")
        }
    )
    results: list[LocalCleanupResult] = []
    for source in sources:
        if not Path(source).exists():
            continue
        matching = _rows_for_source(rows, Path(source))
        representative = next((row for row in matching if row.get("session_id")), None)
        if representative is None:
            continue
        results.append(
            cleanup_recording_for_session(
                config,
                session_id=representative["session_id"],
                delete_local=delete_local,
                update_registry=update_registry,
            )
        )
    return results


def _rows_for_source(rows: list[dict[str, str]], source: Path) -> list[dict[str, str]]:
    resolved = _resolved(source)
    return [
        row
        for row in rows
        if _resolved(Path(row.get("local_raw_folder") or row.get("raw_folder") or "")) == resolved
    ]


def _synology_folder(
    rows: list[dict[str, str]],
    source: Path,
    config: dict[str, Any],
) -> Path:
    configured = {str(Path(row["server_raw_folder"])) for row in rows if row.get("server_raw_folder")}
    if len(configured) == 1:
        return Path(configured.pop())
    server_root = config.get("staging", {}).get("server_raw_root") or config.get("backup", {}).get("root")
    local_root = config.get("staging", {}).get("local_raw_root") or config.get("project", {}).get("raw_root")
    if not server_root or not local_root:
        return Path("")
    try:
        relative = source.resolve().relative_to(Path(local_root).resolve())
    except ValueError:
        relative = Path(source.name)
    return Path(server_root) / relative


def _local_archive_folder(
    source: Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> Path:
    configured = {
        str(Path(row["local_recording_archive_folder"]))
        for row in rows
        if row.get("local_recording_archive_folder")
    }
    if len(configured) == 1:
        return Path(configured.pop())
    archive_root = config.get("backup", {}).get("local_recording_archive", {}).get("root")
    local_root = config.get("staging", {}).get("local_raw_root") or config.get("project", {}).get("raw_root")
    if not archive_root or not local_root:
        return Path("")
    try:
        relative = source.resolve().relative_to(Path(local_root).resolve())
    except ValueError:
        relative = Path(source.name)
    return Path(archive_root) / relative


def _validate_local_source(source: Path, config: dict[str, Any], errors: list[str]) -> None:
    local_root_value = (
        config.get("staging", {}).get("local_raw_root")
        or config.get("project", {}).get("raw_root")
    )
    if not local_root_value:
        errors.append("local raw root is not configured")
        return
    local_root = Path(local_root_value)
    try:
        relative = source.resolve().relative_to(local_root.resolve())
    except ValueError:
        errors.append(f"local recording folder is outside the configured local raw root: {source}")
        return
    if not relative.parts:
        errors.append("refusing to clean the local raw root itself")


def _file_size_map(folder: Path) -> dict[Path, int]:
    return {path.relative_to(folder): path.stat().st_size for path in folder.rglob("*") if path.is_file()}


def _tree_mismatches(source_files: dict[Path, int], destination: Path, label: str) -> list[str]:
    if not destination.exists():
        return [f"{label} folder does not exist: {destination}"]
    missing = 0
    mismatched = 0
    examples: list[str] = []
    for relative, source_size in source_files.items():
        target = destination / relative
        if not target.is_file():
            missing += 1
            if len(examples) < 5:
                examples.append(f"missing {relative}")
            continue
        target_size = target.stat().st_size
        if target_size != source_size:
            mismatched += 1
            if len(examples) < 5:
                examples.append(f"size mismatch {relative}: local={source_size} destination={target_size}")
    if not missing and not mismatched:
        return []
    return [
        f"{label} verification failed: {missing} missing, {mismatched} size mismatch(es); "
        + "; ".join(examples)
    ]


def _delete_verified_source(source: Path, config: dict[str, Any]) -> None:
    errors: list[str] = []
    _validate_local_source(source, config, errors)
    if errors:
        raise RuntimeError("; ".join(errors))
    shutil.rmtree(_extended_windows_path(source), onerror=_handle_rmtree_error)


def _extended_windows_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def _handle_rmtree_error(function, path: str, exc_info) -> None:
    error = exc_info[1]
    if isinstance(error, FileNotFoundError):
        return
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except FileNotFoundError:
        return


def _update_recording_rows(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    result: LocalCleanupResult,
) -> None:
    checked_at = datetime.now(timezone.utc).isoformat()
    fields = {
        "local_cleanup_status": result.status,
        "local_cleanup_checked_at": checked_at,
        "local_cleanup_files": str(result.checked_files),
        "local_cleanup_bytes": str(result.checked_bytes),
        "local_cleanup_error": "; ".join(result.errors),
    }
    if result.deleted:
        fields.update(
            {
                "local_stage_status": "offloaded",
                "local_stage_checked_at": checked_at,
                "local_stage_error": "",
            }
        )
    source = _resolved(result.source_folder)
    for row in rows:
        row_source = Path(row.get("local_raw_folder") or row.get("raw_folder") or "")
        if _resolved(row_source) != source:
            continue
        updated = dict(row)
        updated.update(fields)
        update_row(config["project"]["registry_csv"], updated)


def _resolved(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except OSError:
        return str(path.absolute()).lower()


def print_cleanup_result(result: LocalCleanupResult) -> None:
    print(f"source: {result.source_folder}")
    print(f"synology: {result.synology_folder}")
    print(f"local_archive: {result.local_archive_folder}")
    print(f"status: {result.status}")
    print(f"checked_files: {result.checked_files}")
    print(f"checked_bytes: {result.checked_bytes}")
    for error in result.errors:
        print(f"blocked: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Synology and D: copies, then optionally remove a staged local recording."
    )
    parser.add_argument("--config", default="pipeline/config_analysis.yaml")
    parser.add_argument("--session-id")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--update-registry", action="store_true")
    args = parser.parse_args()
    if bool(args.session_id) == bool(args.all_eligible):
        raise SystemExit("Choose exactly one of --session-id SESSION_ID or --all-eligible")

    config = load_config(args.config)
    if args.all_eligible:
        results = cleanup_all_eligible(
            config,
            delete_local=args.delete_local,
            update_registry=args.update_registry,
        )
    else:
        results = [
            cleanup_recording_for_session(
                config,
                session_id=args.session_id,
                delete_local=args.delete_local,
                update_registry=args.update_registry,
            )
        ]
    blocked = False
    for result in results:
        print_cleanup_result(result)
        blocked = blocked or not result.eligible
    if blocked and args.delete_local:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
