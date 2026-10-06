from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path


REGISTRY_COLUMNS = [
    "session_id",
    "recording_id",
    "recording_block",
    "mouse_id",
    "animal_id",
    "recording_date",
    "task",
    "raw_folder",
    "server_raw_folder",
    "local_raw_folder",
    "behavior_session_folder",
    "behavior_events_csv",
    "open_ephys_experiment_name",
    "open_ephys_block_index",
    "open_ephys_record_node",
    "open_ephys_recording_folder",
    "stream_name",
    "stream_id",
    "probe_label",
    "probe_serial",
    "probe_type",
    "probeinterface_json",
    "probeinterface_hash",
    "n_channels",
    "sampling_frequency",
    "duration_seconds",
    "channel_ids_hash",
    "geometry_hash",
    "site_hash",
    "preprocessing_config_hash",
    "sorter_name",
    "sorter_version",
    "spikeinterface_version",
    "kilosort_version",
    "status",
    "raw_status",
    "backup_status",
    "backup_folder",
    "backup_checked_at",
    "backup_error",
    "local_stage_status",
    "local_stage_folder",
    "local_stage_checked_at",
    "local_stage_error",
    "derived_backup_status",
    "derived_backup_folder",
    "derived_backup_checked_at",
    "derived_backup_error",
    "derived_backup_files",
    "derived_backup_bytes",
    "derived_local_backup_status",
    "derived_local_backup_folder",
    "derived_local_backup_checked_at",
    "derived_local_backup_error",
    "local_recording_archive_status",
    "local_recording_archive_folder",
    "local_recording_archive_checked_at",
    "local_recording_archive_error",
    "local_recording_archive_files",
    "local_recording_archive_bytes",
    "nwb_backup_status",
    "nwb_backup_folder",
    "nwb_backup_path",
    "nwb_backup_checked_at",
    "nwb_backup_error",
    "nwb_source_resolution_json",
    "preprocess_status",
    "sort_status",
    "qc_status",
    "phy_export_status",
    "unitmatch_status",
    "queue_status",
    "queue_position",
    "active_profile",
    "profile_version",
    "active_config_hash",
    "active_source_hash",
    "current_step",
    "active_job_id",
    "interrupted_at",
    "processed_folder",
    "preprocessed_folder",
    "sorter_output_folder",
    "sorting_analyzer_folder",
    "quality_metrics_csv",
    "last_updated",
    "error_message",
    "notes",
]

def read_registry(path: str | Path) -> list[dict[str, str]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8", newline="") as f:
        return [_normalize_read_row(row) for row in csv.DictReader(f)]


def find_row(path: str | Path, *, session_id: str | None = None, raw_folder: str | None = None, probe_label: str | None = None) -> dict[str, str]:
    rows = read_registry(path)
    matches = []
    for row in rows:
        if session_id and row.get("session_id") != session_id:
            continue
        if raw_folder and Path(row.get("raw_folder", "")).resolve() != Path(raw_folder).resolve():
            continue
        if probe_label and row.get("probe_label") != probe_label:
            continue
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registry row, found {len(matches)}")
    return matches[0]


PROCESSING_STATE_COLUMNS = {
    "status",
    "preprocess_status",
    "sort_status",
    "qc_status",
    "phy_export_status",
    "unitmatch_status",
    "queue_status",
    "queue_position",
    "active_profile",
    "profile_version",
    "active_config_hash",
    "active_source_hash",
    "current_step",
    "active_job_id",
    "interrupted_at",
    "error_message",
}


def upsert_rows(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    preserve_completed: bool = False,
    preserve_processing_state: bool = False,
) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_registry(registry_path)
    by_key = {(row.get("session_id", ""), row.get("stream_id", ""), row.get("probe_label", "")): row for row in existing}

    for row in rows:
        normalized = {column: "" for column in REGISTRY_COLUMNS}
        normalized.update({key: "" if value is None else str(value) for key, value in row.items()})
        if not normalized.get("mouse_id") and normalized.get("animal_id"):
            normalized["mouse_id"] = normalized["animal_id"]
        if not normalized.get("animal_id") and normalized.get("mouse_id"):
            normalized["animal_id"] = normalized["mouse_id"]
        key = (
            normalized.get("session_id", ""),
            normalized.get("stream_id", ""),
            normalized.get("probe_label", ""),
        )
        if preserve_completed and by_key.get(key, {}).get("status") == "complete":
            continue
        if preserve_processing_state and key in by_key:
            previous = by_key[key]
            for column in PROCESSING_STATE_COLUMNS:
                normalized[column] = previous.get(column, "")
        by_key[key] = normalized

    ordered = list(by_key.values())
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{registry_path.name}.",
        suffix=".tmp",
        dir=registry_path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
            writer.writeheader()
            writer.writerows(ordered)
        _replace_registry_file(temp_name, registry_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def update_row(path: str | Path, row: dict[str, object]) -> None:
    upsert_rows(path, [row])


def _replace_registry_file(temp_name: str, registry_path: Path) -> None:
    delay = 0.05
    for attempt in range(10):
        try:
            os.replace(temp_name, registry_path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def _normalize_read_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if not normalized.get("mouse_id") and normalized.get("animal_id"):
        normalized["mouse_id"] = normalized["animal_id"]
    if not normalized.get("animal_id") and normalized.get("mouse_id"):
        normalized["animal_id"] = normalized["mouse_id"]
    return normalized



