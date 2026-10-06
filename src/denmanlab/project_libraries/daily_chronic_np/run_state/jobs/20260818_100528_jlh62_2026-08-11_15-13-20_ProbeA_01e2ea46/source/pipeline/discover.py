from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spikeinterface
import spikeinterface.full as si

from .config import load_config
from .hashing import hash_file
from .machine import discovery_root, raw_folder_for_discovered_session, server_folder_for_discovered_session
from .output_paths import output_paths_for_processed_folder, processed_folder_for_stream, sorting_analyzer_folder_name
from .probe_maps import ensure_probeinterface_json, generated_probe_path
from .recording_identity import recording_identity
from .registry import upsert_rows


def discover(config: dict[str, Any], *, dry_run: bool = False) -> list[dict[str, object]]:
    raw_root = discovery_root(config)
    completion_name = config["project"].get("completion_file_name", "recording_complete.txt")
    require_completion = bool(config["project"].get("require_completion_file", True))
    allow_inactive_fallback = bool(config["project"].get("allow_inactive_completion_fallback", False))
    inactive_minutes = float(config["project"].get("inactive_fallback_minutes", 0))
    minimum_duration = float(config.get("recordings", {}).get("minimum_duration_seconds", 0))
    generate_probe_maps = bool(config.get("probes", {}).get("generate_probeinterface_from_recording", False))

    rows: list[dict[str, object]] = []
    for session_folder in _candidate_sessions(raw_root):
        for record_node, oebin in _find_open_ephys_recordings(session_folder):
            rows.extend(
                _discover_oebin(
                    config=config,
                    session_folder=session_folder,
                    record_node=record_node,
                    oebin=oebin,
                    completion_name=completion_name,
                    require_completion=require_completion,
                    allow_inactive_fallback=allow_inactive_fallback,
                    inactive_minutes=inactive_minutes,
                    minimum_duration=minimum_duration,
                    generate_probe_maps=generate_probe_maps,
                    dry_run=dry_run,
                )
            )
    return rows


def _discover_oebin(
    *,
    config: dict[str, Any],
    session_folder: Path,
    record_node: Path,
    oebin: Path,
    completion_name: str,
    require_completion: bool,
    allow_inactive_fallback: bool,
    inactive_minutes: float,
    minimum_duration: float,
    generate_probe_maps: bool,
    dry_run: bool,
) -> list[dict[str, object]]:
        metadata = json.loads(oebin.read_text(encoding="utf-8"))
        session_info = _parse_session_folder(session_folder.name)
        local_session_folder = raw_folder_for_discovered_session(session_folder, config)
        server_session_folder = server_folder_for_discovered_session(session_folder, config)
        recording_id, recording_block = recording_identity(
            session_folder=session_folder.name,
            record_node=record_node.name,
            experiment=oebin.parent.parent.name,
            recording=oebin.parent.name,
        )
        completion_found = any(
            (folder / completion_name).exists() for folder in (session_folder, record_node, oebin.parent)
        )
        inactive_complete = (
            allow_inactive_fallback
            and inactive_minutes > 0
            and _latest_mtime_minutes_ago(session_folder) >= inactive_minutes
        )
        raw_status = "complete" if completion_found or inactive_complete or not require_completion else "incomplete"
        raw_note = ""
        if not completion_found and inactive_complete:
            raw_note = f"accepted by inactive fallback ({inactive_minutes:g} min)"
        neo_stream_names, neo_stream_ids = si.get_neo_streams("openephysbinary", record_node)

        rows: list[dict[str, object]] = []
        for stream in metadata.get("continuous", []):
            if int(stream.get("num_channels", 0)) != 384:
                continue
            stream_name = _resolve_neo_stream_name(stream, neo_stream_names)
            stream_id = neo_stream_ids[neo_stream_names.index(stream_name)]
            probe_label = str(stream.get("stream_name", "")).strip()
            probe_map = _find_probe_map(session_folder, probe_label)

            n_channels = int(stream.get("num_channels", 0))
            sampling_frequency = float(stream.get("sample_rate", 0.0))
            duration_seconds = _estimate_duration_seconds(oebin, stream, n_channels, sampling_frequency)
            recording_suffix = _recording_suffix(oebin)
            session_id = "_".join(
                part
                for part in (
                    session_info["animal_id"],
                    session_info["recording_date"],
                    session_info["task"],
                    recording_suffix,
                    probe_label,
                )
                if part
            )
            processed_folder = _processed_folder_for_discovered_stream(
                config=config,
                session_folder=session_folder,
                local_session_folder=local_session_folder,
                record_node=record_node,
                oebin=oebin,
                stream=stream,
                session_info=session_info,
                probe_label=probe_label,
            )
            output_paths = output_paths_for_processed_folder(
                processed_folder,
                analyzer_folder_name=sorting_analyzer_folder_name(config),
            )
            row_probe_map = probe_map
            generated_map_note = ""
            if row_probe_map is None and generate_probe_maps:
                row_for_generation = {
                    "raw_folder": str(local_session_folder),
                    "processed_folder": str(processed_folder),
                    "probe_label": probe_label,
                    "stream_name": stream_name,
                }
                row_probe_map = generated_probe_path(row_for_generation)
                generated_map_note = "ProbeInterface JSON generated from Open Ephys metadata"
                if not dry_run:
                    rec = si.read_openephys(
                        record_node,
                        stream_name=stream_name,
                        experiment_name=oebin.parent.parent.name,
                        load_sync_timestamps=False,
                    )
                    row_probe_map = ensure_probeinterface_json(rec, row_for_generation, write=True)

            status, notes = _status_for_discovered_row(
                raw_status=raw_status,
                probe_map=row_probe_map,
                duration_seconds=duration_seconds,
                minimum_duration=minimum_duration,
            )
            rows.append(
                {
                    "session_id": session_id,
                    "recording_id": recording_id,
                    "recording_block": recording_block,
                    "mouse_id": session_info["mouse_id"],
                    "animal_id": session_info["animal_id"],
                    "recording_date": session_info["recording_date"],
                    "task": session_info["task"],
                    "raw_folder": local_session_folder,
                    "server_raw_folder": server_session_folder,
                    "local_raw_folder": local_session_folder,
                    "open_ephys_experiment_name": oebin.parent.parent.name,
                    "open_ephys_block_index": oebin.parent.name,
                    "open_ephys_record_node": record_node.name,
                    "open_ephys_recording_folder": str(oebin.parent),
                    "stream_name": stream_name,
                    "stream_id": stream_id,
                    "probe_label": probe_label,
                    "probe_type": "Neuropixels",
                    "probeinterface_json": row_probe_map or "",
                    "probeinterface_hash": hash_file(row_probe_map) if row_probe_map and Path(row_probe_map).exists() else "",
                    "n_channels": n_channels,
                    "sampling_frequency": sampling_frequency,
                    "duration_seconds": f"{duration_seconds:.3f}",
                    "sorter_name": config.get("sorting", {}).get("sorter_name", "kilosort4"),
                    "spikeinterface_version": spikeinterface.__version__,
                    "status": status,
                    "raw_status": raw_status,
                    "local_stage_status": _local_stage_status(local_session_folder, config),
                    "local_stage_folder": str(local_session_folder),
                    "preprocess_status": "pending" if status == "registered" else "blocked",
                    "sort_status": "pending" if status == "registered" else "blocked",
                    "qc_status": "pending" if status == "registered" else "blocked",
                    "phy_export_status": "pending" if status == "registered" else "blocked",
                    "unitmatch_status": "planned",
                    "processed_folder": output_paths["processed_folder"],
                    "preprocessed_folder": output_paths["preprocessed_folder"],
                    "sorter_output_folder": output_paths["sorter_output_folder"],
                    "sorting_analyzer_folder": output_paths["sorting_analyzer_folder"],
                    "quality_metrics_csv": output_paths["quality_metrics_csv"],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "notes": "; ".join(note for note in (notes, raw_note, generated_map_note) if note),
                }
            )
        return rows



def _processed_folder_for_discovered_stream(
    *,
    config: dict[str, Any],
    session_folder: Path,
    local_session_folder: Path,
    record_node: Path,
    oebin: Path,
    stream: dict[str, Any],
    session_info: dict[str, str],
    probe_label: str,
) -> Path:
    if config.get("project", {}).get("output_layout") == "recording_local_probe_folder":
        output_name = config.get("project", {}).get("recording_local_output_name", "spikeinterface_output")
        recording_relative = oebin.parent.relative_to(session_folder)
        folder_name = str(stream["folder_name"]).strip("/\\")
        return local_session_folder / recording_relative / "continuous" / folder_name / output_name

    if local_session_folder != session_folder:
        processed_root = Path(config["project"]["processed_root"])
        return processed_root / session_info["animal_id"] / local_session_folder.name / probe_label

    return processed_folder_for_stream(
        config=config,
        session_folder=session_folder,
        record_node=record_node,
        stream=stream,
        session_info=session_info,
        probe_label=probe_label,
    )


def _local_stage_status(local_session_folder: Path, config: dict[str, Any]) -> str:
    if not config.get("staging", {}).get("enabled", False):
        return "not_required"
    if not local_session_folder.exists():
        return "pending"
    has_local_raw = any(local_session_folder.glob("Record Node */experiment*/recording*/structure.oebin"))
    return "staged" if has_local_raw else "pending"
def _candidate_sessions(raw_root: Path) -> list[Path]:
    return [path for path in sorted(raw_root.iterdir()) if path.is_dir() and _find_open_ephys_recordings(path)]


def _find_open_ephys_recordings(session_folder: Path) -> list[tuple[Path, Path]]:
    recordings: list[tuple[Path, Path]] = []
    for record_node in sorted(session_folder.glob("Record Node *")):
        for oebin in sorted(record_node.glob("experiment*/recording*/structure.oebin")):
            recordings.append((record_node, oebin))
    return recordings


def _recording_suffix(oebin: Path) -> str:
    experiment = oebin.parent.parent.name
    recording = oebin.parent.name
    if experiment == "experiment1" and recording == "recording1":
        return ""
    return f"{experiment}_{recording}"


def _recording_block_index(recording_name: str) -> int:
    match = re.search(r"recording(\d+)$", recording_name)
    if not match:
        return 0
    return max(0, int(match.group(1)) - 1)


def _parse_session_folder(name: str) -> dict[str, str]:
    match = re.search(r"(?P<date>20\d{2}-\d{2}-\d{2})(?:[_-](?P<task>.*))?", name)
    if not match:
        return {"mouse_id": name, "animal_id": name, "recording_date": "", "task": ""}
    animal_id = name[: match.start("date")].rstrip("_-")
    task = (match.group("task") or "").strip("_-")
    mouse_id = animal_id or name
    return {"mouse_id": mouse_id, "animal_id": mouse_id, "recording_date": match.group("date"), "task": task}


def _resolve_neo_stream_name(stream: dict[str, Any], neo_stream_names: list[str]) -> str:
    folder_name = str(stream.get("folder_name", "")).strip("/")
    return next((name for name in neo_stream_names if folder_name in name), str(stream.get("stream_name", "")))


def _find_probe_map(session_folder: Path, probe_label: str) -> Path | None:
    candidates = [
        session_folder / f"{probe_label}_probeinterface.json",
        session_folder / f"{probe_label.lower()}_probeinterface.json",
        session_folder / f"{probe_label}.probeinterface.json",
    ]
    candidates.extend(sorted(session_folder.glob(f"**/*{probe_label}*probeinterface*.json")))
    candidates.extend(sorted(session_folder.glob(f"**/*{probe_label.lower()}*probeinterface*.json")))
    return next((path for path in candidates if path.exists() and path.is_file()), None)


def _estimate_duration_seconds(
    oebin: Path,
    stream: dict[str, Any],
    n_channels: int,
    sampling_frequency: float,
) -> float:
    dat = oebin.parent / "continuous" / str(stream["folder_name"]).strip("/\\") / "continuous.dat"
    if not dat.exists() or n_channels <= 0 or sampling_frequency <= 0:
        return 0.0
    bytes_per_sample = 2
    frames = dat.stat().st_size / (bytes_per_sample * n_channels)
    return frames / sampling_frequency


def _status_for_discovered_row(
    *,
    raw_status: str,
    probe_map: Path | None,
    duration_seconds: float,
    minimum_duration: float,
) -> tuple[str, str]:
    notes: list[str] = []
    if raw_status != "complete":
        notes.append("missing recording_complete.txt")
    if probe_map is None:
        notes.append("missing ProbeInterface JSON")
    if duration_seconds < minimum_duration:
        notes.append(f"duration below minimum ({duration_seconds:.3f} < {minimum_duration:.3f})")
    if raw_status != "complete":
        return "skipped", "; ".join(notes)
    if probe_map is None:
        return "missing_probe_map", "; ".join(notes)
    if duration_seconds < minimum_duration:
        return "skipped", "; ".join(notes)
    return "registered", "ready for preprocessing"


def _latest_mtime_minutes_ago(folder: Path) -> float:
    latest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        if any(part.startswith("spikeinterface_output") for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    age_seconds = datetime.now().timestamp() - latest
    return max(0.0, age_seconds / 60.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Open Ephys recordings and update the CSV registry.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = discover(config, dry_run=args.dry_run)
    for row in rows:
        print(
            f"{row['session_id']} | {row['probe_label']} | {row['status']} | "
            f"raw={row['raw_status']} | {row['notes']}"
        )
    if not args.dry_run:
        upsert_rows(
            config["project"]["registry_csv"],
            rows,
            preserve_completed=True,
            preserve_processing_state=True,
        )
        print(f"updated {config['project']['registry_csv']} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()




