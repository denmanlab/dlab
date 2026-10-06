from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def recording_identity(
    *,
    session_folder: str,
    record_node: str,
    experiment: str,
    recording: str,
) -> tuple[str, str]:
    key = "::".join(
        (
            str(session_folder).strip(),
            str(record_node).strip(),
            str(experiment).strip(),
            str(recording).strip(),
        )
    )
    digest = hashlib.sha1(key.lower().encode("utf-8")).hexdigest()[:16]
    label = " / ".join(part for part in (experiment, recording) if part)
    return f"rec_{digest}", label


def recording_identity_for_row(row: dict[str, Any]) -> tuple[str, str]:
    existing = str(row.get("recording_id") or "").strip()
    existing_label = str(row.get("recording_block") or "").strip()
    if existing:
        return existing, existing_label or _fallback_block_label(row)

    raw_folder = Path(str(row.get("server_raw_folder") or row.get("raw_folder") or ""))
    session_folder = raw_folder.name or str(row.get("mouse_id") or row.get("session_id") or "recording")
    record_node = str(row.get("open_ephys_record_node") or "").strip()
    experiment = str(row.get("open_ephys_experiment_name") or "").strip()
    recording = str(row.get("open_ephys_block_index") or "").strip()

    if not record_node or not experiment or not recording:
        inferred = _infer_from_output_path(str(row.get("processed_folder") or ""))
        record_node = record_node or inferred.get("record_node", "Record Node")
        experiment = experiment or inferred.get("experiment", "experiment1")
        recording = recording or inferred.get("recording", "recording1")

    return recording_identity(
        session_folder=session_folder,
        record_node=record_node,
        experiment=experiment,
        recording=recording,
    )


def recording_group_payload(row: dict[str, Any]) -> dict[str, Any]:
    recording_id, block = recording_identity_for_row(row)
    duration = _safe_float(row.get("duration_seconds"))
    inferred = _infer_from_output_path(str(row.get("processed_folder") or ""))
    return {
        "recording_id": recording_id,
        "recording_block": block,
        "record_node": str(row.get("open_ephys_record_node") or inferred.get("record_node", "")),
        "experiment": str(row.get("open_ephys_experiment_name") or _block_part(block, 0)),
        "recording": str(row.get("open_ephys_block_index") or _block_part(block, 1)),
        "duration_seconds": duration,
    }


def _infer_from_output_path(path: str) -> dict[str, str]:
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    result: dict[str, str] = {}
    for index, part in enumerate(parts):
        if part.lower().startswith("record node "):
            result["record_node"] = part
        if re.fullmatch(r"experiment\d+", part, flags=re.IGNORECASE):
            result["experiment"] = part
            if index + 1 < len(parts) and re.fullmatch(r"recording\d+", parts[index + 1], flags=re.IGNORECASE):
                result["recording"] = parts[index + 1]
    return result


def _fallback_block_label(row: dict[str, Any]) -> str:
    experiment = str(row.get("open_ephys_experiment_name") or "experiment1")
    recording = str(row.get("open_ephys_block_index") or "recording1")
    return f"{experiment} / {recording}"


def _block_part(label: str, index: int) -> str:
    parts = [part.strip() for part in str(label).split("/")]
    return parts[index] if index < len(parts) else ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
