from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .nwb_digital_events import (
    digital_event_config,
    digital_event_inventory,
    load_digital_events,
    nwb_output_folder_name,
    nwb_protocol_for_row,
    resolved_line_names,
    safe_event_name,
    save_line_name,
)
from .registry import find_row, read_registry, update_row


@dataclass
class MouseArenaNwbTables:
    output_folder: Path
    nwb_path: Path
    source_recording_folder: Path
    units: pd.DataFrame
    spike_times: dict[int, np.ndarray]
    trials: pd.DataFrame
    digital_events: pd.DataFrame
    behavior_events: pd.DataFrame
    sync_events: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class MouseArenaSessionFrame:
    output_folder: Path
    source_recording_folder: Path
    session_events: pd.DataFrame
    digital_events: pd.DataFrame
    behavior_events: pd.DataFrame
    sync_events: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class MouseArenaStimulusTables:
    output_folder: Path
    source_recording_folder: Path
    trials: pd.DataFrame
    session_events: pd.DataFrame
    frame_sync_residuals: pd.DataFrame
    digital_events: pd.DataFrame
    behavior_events: pd.DataFrame
    sync_events: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class MouseArenaUnitsTable:
    output_folder: Path
    units: pd.DataFrame
    spike_times: dict[int, np.ndarray]
    manifest: dict[str, Any]


def build_mouse_arena_nwb_tables(config: dict[str, Any], *, session_id: str) -> MouseArenaNwbTables:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    protocol = nwb_protocol_for_row(config, selected)
    resolution_log: list[dict[str, Any]] = []
    output_recording_folder = _recording_folder(selected)
    output_folder = output_recording_folder / nwb_output_folder_name(config)
    nwb_path = output_folder / _nwb_filename(config, selected)
    input_overrides = _read_input_overrides(output_folder)
    if input_overrides.get("trials_csv") or protocol != "mouse_arena":
        try:
            source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
        except FileNotFoundError:
            source_recording_folder = output_recording_folder
    else:
        source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    try:
        digital_events, time_zero = _load_digital_events(
            config,
            source_recording_folder,
            override_recording_folder=output_recording_folder,
        )
    except (FileNotFoundError, OSError):
        if not input_overrides.get("trials_csv") and protocol == "mouse_arena":
            raise
        digital_events = pd.DataFrame(columns=["event_name", "line", "edge", "ordinal", "timestamp", "time", "sample_number", "state"])
        time_zero = 0.0
    override_manifest: dict[str, Any] = {}
    source_rows: list[dict[str, str]] = []

    if input_overrides.get("trials_csv"):
        trials = _load_trials_override(input_overrides["trials_csv"])
        override_manifest["trials_csv"] = str(input_overrides["trials_csv"])
        behavior_folder: Path | None = None
        behavior_events = pd.DataFrame()
        sync_events = pd.DataFrame()
        alignment = {"mode": "external_trials_csv", "source": str(input_overrides["trials_csv"])}
    elif protocol == "mouse_arena":
        behavior_folder = _find_behavior_folder(config, selected)
        behavior_events = pd.read_csv(behavior_folder / _behavior_filename(config, "events_filename"))
        sync_events = pd.read_csv(behavior_folder / _behavior_filename(config, "sync_events_filename"))
        alignment = _estimate_behavior_alignment(config, digital_events, sync_events)
        trials = _build_trials_table(config, behavior_events, digital_events, alignment)
    else:
        behavior_folder = None
        behavior_events = pd.DataFrame()
        sync_events = pd.DataFrame()
        trials = pd.DataFrame(columns=["start_time", "stop_time"])
        alignment = {"mode": "not_applicable", "protocol": protocol}

    if input_overrides.get("units_csv"):
        try:
            rows = _session_probe_rows(config, selected)
            source_rows = rows
            baseline_units, baseline_spike_times = _build_units_table(rows, config=config, resolution_log=resolution_log)
        except Exception as exc:
            baseline_units = pd.DataFrame()
            baseline_spike_times = {}
            override_manifest["units_baseline_warning"] = f"Could not load pipeline units for spike-time matching: {exc}"
        units, spike_times, units_override_info = _load_units_override(
            input_overrides["units_csv"],
            baseline_units=baseline_units,
            baseline_spike_times=baseline_spike_times,
        )
        override_manifest["units_csv"] = str(input_overrides["units_csv"])
        override_manifest["units"] = units_override_info
    else:
        rows = _session_probe_rows(config, selected)
        source_rows = rows
        units, spike_times = _build_units_table(rows, config=config, resolution_log=resolution_log)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session_id": _session_base_id(selected),
        "mouse_id": selected.get("mouse_id") or selected.get("animal_id", ""),
        "recording_date": selected.get("recording_date", ""),
        "protocol": protocol,
        "recording_folder": str(output_recording_folder),
        "source_recording_folder": str(source_recording_folder),
        "behavior_folder": str(behavior_folder) if behavior_folder is not None else "",
        "nwb_path": str(nwb_path),
        "probes": [row.get("probe_label", "") for row in source_rows],
        "num_units": int(len(units)),
        "num_trials": int(len(trials)),
        "num_digital_events": int(len(digital_events)),
        "digital_event_lines": _digital_line_config(config),
        "time_zero": {"source": "nidaq_continuous_start", "timestamp_seconds": float(time_zero)},
        "behavior_to_ephys_alignment": alignment,
        "time_policy": (
            "Ephys TTL timestamps are ground truth. Behavior clock alignment is used only for behavior-only events or missing TTL fallbacks."
            if protocol == "mouse_arena"
            else "Ephys TTL timestamps are stored directly. Protocol-specific interpretation is not applied."
        ),
        "analog": _analog_manifest(config, source_recording_folder),
        "source_rows": [{key: row.get(key, "") for key in ["session_id", "probe_label", "processed_folder", "sorter_output_folder"]} for row in source_rows],
        "source_resolution": resolution_log,
    }
    if override_manifest:
        manifest["input_overrides"] = override_manifest
    manifest["subject"] = _subject_metadata(config, selected, _nwb_session_start_time(manifest))
    manifest["session_metadata"] = _session_metadata(config, protocol=protocol)
    return MouseArenaNwbTables(
        output_folder=output_folder,
        nwb_path=nwb_path,
        source_recording_folder=source_recording_folder,
        units=units,
        spike_times=spike_times,
        trials=trials,
        digital_events=digital_events,
        behavior_events=behavior_events,
        sync_events=sync_events,
        manifest=manifest,
    )


def build_mouse_arena_session_dataframe(config: dict[str, Any], *, session_id: str) -> MouseArenaSessionFrame:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    resolution_log: list[dict[str, Any]] = []
    recording_folder = _recording_folder(selected)
    source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    output_folder = recording_folder / nwb_output_folder_name(config)

    behavior_folder = _find_behavior_folder(config, selected)
    behavior_events = pd.read_csv(behavior_folder / _behavior_filename(config, "events_filename"))
    sync_events = pd.read_csv(behavior_folder / _behavior_filename(config, "sync_events_filename"))
    digital_events, time_zero = _load_digital_events(
        config,
        source_recording_folder,
        override_recording_folder=recording_folder,
    )
    alignment = _estimate_behavior_alignment(config, digital_events, sync_events)
    session_events = _build_session_events_dataframe(behavior_events, digital_events, alignment)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session_id": _session_base_id(selected),
        "mouse_id": selected.get("mouse_id") or selected.get("animal_id", ""),
        "recording_date": selected.get("recording_date", ""),
        "recording_folder": str(recording_folder),
        "source_recording_folder": str(source_recording_folder),
        "behavior_folder": str(behavior_folder),
        "session_events_csv": str(output_folder / "session_events.csv"),
        "digital_events_csv": str(output_folder / "nwb_digital_events.csv"),
        "num_behavior_events": int(len(behavior_events)),
        "num_session_events": int(len(session_events)),
        "num_digital_events": int(len(digital_events)),
        "event_type_counts": session_events["event_type"].value_counts(dropna=False).to_dict(),
        "digital_event_lines": _digital_line_config(config),
        "time_zero": {"source": "nidaq_continuous_start", "timestamp_seconds": float(time_zero)},
        "behavior_to_ephys_alignment": alignment,
        "source_resolution": resolution_log,
        "time_policy": "Ephys TTL timestamps are ground truth. Behavior clock alignment is used only for behavior-only events or missing TTL fallbacks.",
    }
    return MouseArenaSessionFrame(
        output_folder=output_folder,
        source_recording_folder=source_recording_folder,
        session_events=session_events,
        digital_events=digital_events,
        behavior_events=behavior_events,
        sync_events=sync_events,
        manifest=manifest,
    )


def build_mouse_arena_units_table(config: dict[str, Any], *, session_id: str) -> MouseArenaUnitsTable:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    resolution_log: list[dict[str, Any]] = []
    rows = _session_probe_rows(config, selected)
    recording_folder = _recording_folder(selected)
    output_folder = recording_folder / nwb_output_folder_name(config)
    units, spike_times = _build_units_table(rows, config=config, resolution_log=resolution_log)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session_id": _session_base_id(selected),
        "mouse_id": selected.get("mouse_id") or selected.get("animal_id", ""),
        "recording_date": selected.get("recording_date", ""),
        "recording_folder": str(recording_folder),
        "units_csv": str(output_folder / "nwb_units.csv"),
        "num_units": int(len(units)),
        "num_probes": int(len(rows)),
        "probe_unit_counts": units["probe_label"].value_counts(dropna=False).to_dict() if "probe_label" in units else {},
        "columns": list(units.columns),
        "source_rows": [{key: row.get(key, "") for key in ["session_id", "probe_label", "processed_folder", "sorter_output_folder"]} for row in rows],
        "source_resolution": resolution_log,
        "id_policy": "nwb_unit_id is a numeric NWB row id. unique_unit_id is probe_id + source_unit_id and is stable across the combined recording table.",
    }
    return MouseArenaUnitsTable(output_folder=output_folder, units=units, spike_times=spike_times, manifest=manifest)


def write_mouse_arena_units_table(config: dict[str, Any], *, session_id: str) -> dict[str, str]:
    table = build_mouse_arena_units_table(config, session_id=session_id)
    table.output_folder.mkdir(parents=True, exist_ok=True)
    units_csv = table.output_folder / "nwb_units.csv"
    manifest_json = table.output_folder / "nwb_units_manifest.json"
    table.units.drop(columns=["spike_times"], errors="ignore").to_csv(units_csv, index=False)
    manifest_json.write_text(json.dumps(table.manifest, indent=2, default=str), encoding="utf-8")
    resolution_json = _write_source_resolution_log(config, table.output_folder, table.manifest)
    _update_registry_for_recording_session(config, session_id, {"nwb_source_resolution_json": str(resolution_json)})
    return {"units_csv": str(units_csv), "manifest_json": str(manifest_json), "source_resolution_json": str(resolution_json)}


def build_mouse_arena_stimulus_tables(config: dict[str, Any], *, session_id: str) -> MouseArenaStimulusTables:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    resolution_log: list[dict[str, Any]] = []
    recording_folder = _recording_folder(selected)
    source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    output_folder = recording_folder / nwb_output_folder_name(config)

    behavior_folder = _find_behavior_folder(config, selected)
    behavior_events = pd.read_csv(behavior_folder / _behavior_filename(config, "events_filename"))
    sync_events = pd.read_csv(behavior_folder / _behavior_filename(config, "sync_events_filename"))
    digital_events, time_zero = _load_digital_events(
        config,
        source_recording_folder,
        override_recording_folder=recording_folder,
    )
    alignment = _estimate_behavior_alignment(config, digital_events, sync_events)
    session_events = _build_session_events_dataframe(behavior_events, digital_events, alignment)
    trials = _build_trials_table(config, behavior_events, digital_events, alignment)
    frame_sync_residuals = _build_frame_sync_residuals(digital_events, sync_events, alignment)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session_id": _session_base_id(selected),
        "mouse_id": selected.get("mouse_id") or selected.get("animal_id", ""),
        "recording_date": selected.get("recording_date", ""),
        "recording_folder": str(recording_folder),
        "source_recording_folder": str(source_recording_folder),
        "behavior_folder": str(behavior_folder),
        "trials_csv": str(output_folder / "trials_df.csv"),
        "session_events_csv": str(output_folder / "session_events.csv"),
        "frame_sync_residuals_csv": str(output_folder / "frame_sync_residuals.csv"),
        "digital_events_csv": str(output_folder / "nwb_digital_events.csv"),
        "num_trials": int(len(trials)),
        "num_session_events": int(len(session_events)),
        "num_digital_events": int(len(digital_events)),
        "num_trial_start_ttls": int(((digital_events["event_name"] == "trial_start") & (digital_events["edge"] == "rising")).sum()),
        "num_unmatched_trial_start_ttls": int(
            ((digital_events["event_name"] == "trial_start") & (digital_events["edge"] == "rising")).sum() - len(trials)
        ),
        "digital_event_counts": digital_events["event_name"].value_counts(dropna=False).to_dict(),
        "frame_sync_residual_ms": _residual_summary(frame_sync_residuals),
        "behavior_to_ephys_alignment": alignment,
        "source_resolution": resolution_log,
        "time_zero": {"source": "nidaq_continuous_start", "timestamp_seconds": float(time_zero)},
        "time_policy": "Trial starts and rewards use ephys TTL timestamps. Behavior-only times are mapped to ephys time using game-frame sync events.",
    }
    return MouseArenaStimulusTables(
        output_folder=output_folder,
        source_recording_folder=source_recording_folder,
        trials=trials,
        session_events=session_events,
        frame_sync_residuals=frame_sync_residuals,
        digital_events=digital_events,
        behavior_events=behavior_events,
        sync_events=sync_events,
        manifest=manifest,
    )


def write_mouse_arena_session_dataframe(config: dict[str, Any], *, session_id: str) -> dict[str, str]:
    frame = build_mouse_arena_session_dataframe(config, session_id=session_id)
    frame.output_folder.mkdir(parents=True, exist_ok=True)
    session_events_csv = frame.output_folder / "session_events.csv"
    digital_events_csv = frame.output_folder / "nwb_digital_events.csv"
    manifest_json = frame.output_folder / "session_events_manifest.json"
    frame.session_events.to_csv(session_events_csv, index=False)
    frame.digital_events.to_csv(digital_events_csv, index=False)
    manifest_json.write_text(json.dumps(frame.manifest, indent=2, default=str), encoding="utf-8")
    resolution_json = _write_source_resolution_log(config, frame.output_folder, frame.manifest)
    fields = _behavior_registry_fields(config, frame.manifest)
    fields["nwb_source_resolution_json"] = str(resolution_json)
    _update_registry_for_recording_session(config, session_id, fields)
    return {
        "session_events_csv": str(session_events_csv),
        "digital_events_csv": str(digital_events_csv),
        "manifest_json": str(manifest_json),
        "source_resolution_json": str(resolution_json),
    }


def write_mouse_arena_stimulus_tables(config: dict[str, Any], *, session_id: str) -> dict[str, str]:
    tables = build_mouse_arena_stimulus_tables(config, session_id=session_id)
    tables.output_folder.mkdir(parents=True, exist_ok=True)
    trials_csv = tables.output_folder / "trials_df.csv"
    session_events_csv = tables.output_folder / "session_events.csv"
    frame_sync_csv = tables.output_folder / "frame_sync_residuals.csv"
    digital_events_csv = tables.output_folder / "nwb_digital_events.csv"
    manifest_json = tables.output_folder / "stimulus_behavior_manifest.json"

    tables.trials.to_csv(trials_csv, index=False)
    tables.session_events.to_csv(session_events_csv, index=False)
    tables.frame_sync_residuals.to_csv(frame_sync_csv, index=False)
    tables.digital_events.to_csv(digital_events_csv, index=False)
    manifest_json.write_text(json.dumps(tables.manifest, indent=2, default=str), encoding="utf-8")
    resolution_json = _write_source_resolution_log(config, tables.output_folder, tables.manifest)
    fields = _behavior_registry_fields(config, tables.manifest)
    fields["nwb_source_resolution_json"] = str(resolution_json)
    _update_registry_for_recording_session(config, session_id, fields)
    return {
        "trials_csv": str(trials_csv),
        "session_events_csv": str(session_events_csv),
        "frame_sync_residuals_csv": str(frame_sync_csv),
        "digital_events_csv": str(digital_events_csv),
        "manifest_json": str(manifest_json),
        "source_resolution_json": str(resolution_json),
    }


def save_mouse_arena_digital_event_npy(
    config: dict[str, Any],
    *,
    session_id: str,
    event_name: str,
    output_name: str,
) -> dict[str, Any]:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    recording_folder = _recording_folder(selected)
    resolution_log: list[dict[str, Any]] = []
    source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    output_folder = recording_folder / nwb_output_folder_name(config) / "digital_events"
    digital_events, _ = _load_digital_events(
        config,
        source_recording_folder,
        override_recording_folder=recording_folder,
    )
    event_name = _normalize_digital_event_name(event_name)
    output_name = _safe_output_stem(output_name)
    if not output_name:
        raise ValueError("output_name is required")
    rows = digital_events[(digital_events["event_name"] == event_name) & (digital_events["edge"] == "rising")]
    if rows.empty:
        raise ValueError(f"No digital events found for {event_name!r}")
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{output_name}.npy"
    np.save(path, rows["time"].to_numpy(dtype=float))
    return {"path": str(path), "event_name": event_name, "output_name": output_name, "count": int(len(rows))}


def build_nwb_digital_event_inventory(config: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    recording_folder = _recording_folder(selected)
    resolution_log: list[dict[str, Any]] = []
    source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    events, time_zero = _load_digital_events(
        config,
        source_recording_folder,
        override_recording_folder=recording_folder,
    )
    output_folder = recording_folder / nwb_output_folder_name(config)
    names_path = output_folder / "digital_line_names.json"
    preview_columns = ["event_name", "line", "edge", "time", "sample_number"]
    return {
        "session_id": session_id,
        "protocol": nwb_protocol_for_row(config, selected),
        "source_recording_folder": str(source_recording_folder),
        "output_folder": str(output_folder),
        "event_count": int(len(events)),
        "line_count": int(events["line"].nunique()) if not events.empty else 0,
        "inventory": digital_event_inventory(events),
        "preview": events[preview_columns].head(12).to_dict(orient="records"),
        "line_names": {str(key): value for key, value in resolved_line_names(config, recording_folder=recording_folder).items()},
        "line_names_path": str(names_path) if names_path.exists() else "",
        "time_zero": float(time_zero),
        "source_resolution": resolution_log,
    }


def save_nwb_digital_line_name(
    config: dict[str, Any],
    *,
    session_id: str,
    line: int,
    name: str,
) -> dict[str, Any]:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    recording_folder = _recording_folder(selected)
    path = save_line_name(config, recording_folder=recording_folder, line=int(line), name=name)
    return {
        "session_id": session_id,
        "line": int(line),
        "event_name": resolved_line_names(config, recording_folder=recording_folder).get(int(line), f"line_{int(line)}"),
        "path": str(path),
    }


def save_nwb_digital_line_npy(
    config: dict[str, Any],
    *,
    session_id: str,
    line: int,
    edge: str,
    output_name: str,
) -> dict[str, Any]:
    selected = find_row(config["project"]["registry_csv"], session_id=session_id)
    recording_folder = _recording_folder(selected)
    resolution_log: list[dict[str, Any]] = []
    source_recording_folder = _resolve_recording_folder_for_nwb(config, selected, resolution_log)
    events, _ = _load_digital_events(
        config,
        source_recording_folder,
        override_recording_folder=recording_folder,
    )
    edge = str(edge).strip().lower()
    if edge not in {"rising", "falling"}:
        raise ValueError("edge must be rising or falling")
    rows = events[(events["line"] == int(line)) & (events["edge"] == edge)]
    if rows.empty:
        raise ValueError(f"No {edge} digital events found on line {int(line)}")
    stem = _safe_output_stem(output_name)
    if not stem:
        raise ValueError("output_name is required")
    output_folder = recording_folder / nwb_output_folder_name(config) / "digital_events"
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{stem}.npy"
    np.save(path, rows["time"].to_numpy(dtype=float))
    return {
        "session_id": session_id,
        "line": int(line),
        "edge": edge,
        "event_name": str(rows["event_name"].iloc[0]),
        "count": int(len(rows)),
        "path": str(path),
    }


def write_mouse_arena_nwb(
    config: dict[str, Any],
    *,
    session_id: str,
    overwrite: bool = False,
    tables_only: bool = False,
) -> dict[str, str]:
    tables = build_mouse_arena_nwb_tables(config, session_id=session_id)
    if tables.nwb_path.exists() and not overwrite and not tables_only:
        raise FileExistsError(f"NWB file already exists: {tables.nwb_path}")
    tables.output_folder.mkdir(parents=True, exist_ok=True)

    units_csv = tables.output_folder / "nwb_units.csv"
    trials_csv = tables.output_folder / "nwb_trials.csv"
    digital_csv = tables.output_folder / "nwb_digital_events.csv"
    behavior_csv = tables.output_folder / "behavior_events_aligned.csv"
    manifest_json = tables.output_folder / "nwb_manifest.json"

    tables.units.drop(columns=["spike_times"], errors="ignore").to_csv(units_csv, index=False)
    tables.trials.to_csv(trials_csv, index=False)
    tables.digital_events.to_csv(digital_csv, index=False)
    _aligned_behavior_events(tables.behavior_events, tables.manifest["behavior_to_ephys_alignment"]).to_csv(behavior_csv, index=False)
    manifest_json.write_text(json.dumps(tables.manifest, indent=2, default=str), encoding="utf-8")
    resolution_json = _write_source_resolution_log(config, tables.output_folder, tables.manifest)

    if not tables_only:
        _write_nwb_file(config, tables)
        nwb_backup = _backup_nwb_file(config, tables)
        if nwb_backup:
            tables.manifest["nwb_backup"] = nwb_backup
            manifest_json.write_text(json.dumps(tables.manifest, indent=2, default=str), encoding="utf-8")
            _update_registry_for_recording_session(config, session_id, nwb_backup)

    outputs = {
        "nwb": str(tables.nwb_path) if tables.nwb_path.exists() else "",
        "units_csv": str(units_csv),
        "trials_csv": str(trials_csv),
        "digital_events_csv": str(digital_csv),
        "behavior_events_csv": str(behavior_csv),
        "manifest_json": str(manifest_json),
        "source_resolution_json": str(resolution_json),
    }
    outputs.update(_behavior_registry_fields(config, tables.manifest))
    if not tables_only:
        outputs.update(tables.manifest.get("nwb_backup", {}))
    fields = _behavior_registry_fields(config, tables.manifest)
    fields["nwb_source_resolution_json"] = str(resolution_json)
    _update_registry_for_recording_session(config, session_id, fields)
    outputs.update(_maybe_backup_derived_outputs(config, session_id=session_id))
    return outputs


def _behavior_registry_fields(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    behavior_folder = str(manifest.get("behavior_folder") or "")
    fields = {"behavior_session_folder": behavior_folder}
    if behavior_folder:
        fields["behavior_events_csv"] = str(Path(behavior_folder) / _behavior_filename(config, "events_filename"))
    return fields


def _write_source_resolution_log(config: dict[str, Any], output_folder: Path, manifest: dict[str, Any]) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / _source_resolution_filename(config)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session_id": manifest.get("session_id", ""),
        "recording_folder": manifest.get("recording_folder", ""),
        "source_recording_folder": manifest.get("source_recording_folder", ""),
        "policy": "NWB-only source resolution. Preprocessing and sorting still require local staged raw data.",
        "checked_locations": manifest.get("source_resolution", []),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _backup_nwb_file(config: dict[str, Any], tables: MouseArenaNwbTables) -> dict[str, str]:
    backup_root = str(config.get("mouse_arena_nwb", {}).get("backup_root") or "").strip()
    if not backup_root:
        return {}
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        if not tables.nwb_path.exists():
            return {
                "nwb_backup_status": "failed",
                "nwb_backup_error": f"NWB file does not exist: {tables.nwb_path}",
                "nwb_backup_checked_at": checked_at,
            }
        mouse_id = str(tables.manifest.get("mouse_id") or "unknown_mouse")
        recording_date = str(tables.manifest.get("recording_date") or "unknown_date")
        if config.get("mouse_arena_nwb", {}).get("backup_by_mouse", True):
            backup_folder = Path(backup_root) / mouse_id / recording_date
        else:
            backup_folder = Path(backup_root)
        backup_folder.mkdir(parents=True, exist_ok=True)
        backup_path = backup_folder / tables.nwb_path.name
        shutil.copy2(tables.nwb_path, backup_path)
        return {
            "nwb_backup_status": "verified",
            "nwb_backup_folder": str(backup_folder),
            "nwb_backup_path": str(backup_path),
            "nwb_backup_checked_at": checked_at,
            "nwb_backup_error": "",
        }
    except Exception as exc:
        return {
            "nwb_backup_status": "failed",
            "nwb_backup_folder": str(Path(backup_root)),
            "nwb_backup_checked_at": checked_at,
            "nwb_backup_error": str(exc),
        }


def _update_registry_for_recording_session(config: dict[str, Any], session_id: str, fields: dict[str, str]) -> None:
    if not fields:
        return
    rows = read_registry(config["project"]["registry_csv"])
    selected = next((row for row in rows if row.get("session_id") == session_id), None)
    if selected is None:
        return
    raw_folder = str(Path(selected.get("raw_folder", "")).resolve())
    for row in rows:
        if str(Path(row.get("raw_folder", "")).resolve()) != raw_folder:
            continue
        row.update({key: value for key, value in fields.items() if value is not None})
        update_row(config["project"]["registry_csv"], row)


def _maybe_backup_derived_outputs(config: dict[str, Any], *, session_id: str) -> dict[str, str]:
    backup_config = config.get("backup", {})
    derived_config = backup_config.get("derived_outputs", {})
    if not backup_config.get("enabled", False) or not derived_config.get("enabled", False):
        return {}
    if not derived_config.get("auto_after_nwb_export", False):
        return {}
    try:
        from .derived_backup import backup_derived_outputs_for_recording_session

        result = backup_derived_outputs_for_recording_session(
            config,
            session_id=session_id,
            update_registry=True,
        )
        return {
            "derived_backup_status": result.status,
            "derived_backup_folder": str(result.backup_folder),
            "derived_backup_error": result.error,
        }
    except Exception as exc:
        return {
            "derived_backup_status": "failed",
            "derived_backup_error": str(exc),
        }


def nwb_output_paths_for_row(config: dict[str, Any], row: dict[str, str]) -> dict[str, str]:
    recording_folder = _recording_folder(row)
    output_folder = recording_folder / nwb_output_folder_name(config)
    nwb_path = output_folder / _nwb_filename(config, row)
    active_overrides = _read_input_overrides(output_folder)
    return {
        "output_folder": str(output_folder),
        "nwb": str(nwb_path) if nwb_path.exists() else "",
        "manifest": str(output_folder / "nwb_manifest.json") if (output_folder / "nwb_manifest.json").exists() else "",
        "units_csv": str(output_folder / "nwb_units.csv") if (output_folder / "nwb_units.csv").exists() else "",
        "units_manifest": str(output_folder / "nwb_units_manifest.json") if (output_folder / "nwb_units_manifest.json").exists() else "",
        "trials_csv": str(output_folder / "nwb_trials.csv") if (output_folder / "nwb_trials.csv").exists() else "",
        "stimulus_trials_csv": str(output_folder / "trials_df.csv") if (output_folder / "trials_df.csv").exists() else "",
        "frame_sync_residuals_csv": str(output_folder / "frame_sync_residuals.csv") if (output_folder / "frame_sync_residuals.csv").exists() else "",
        "stimulus_manifest": str(output_folder / "stimulus_behavior_manifest.json") if (output_folder / "stimulus_behavior_manifest.json").exists() else "",
        "digital_events_csv": str(output_folder / "nwb_digital_events.csv") if (output_folder / "nwb_digital_events.csv").exists() else "",
        "input_overrides": str(output_folder / "input_overrides.json") if (output_folder / "input_overrides.json").exists() else "",
        "active_trials_csv": active_overrides.get("trials_csv", ""),
        "active_units_csv": active_overrides.get("units_csv", ""),
        "behavior_session_folder": row.get("behavior_session_folder", ""),
        "behavior_events_csv": row.get("behavior_events_csv", ""),
        "nwb_backup": _existing_path_string(row.get("nwb_backup_path", "")),
        "nwb_backup_folder": row.get("nwb_backup_folder", ""),
        "nwb_backup_status": row.get("nwb_backup_status", ""),
        "source_resolution": str(output_folder / _source_resolution_filename(config)) if (output_folder / _source_resolution_filename(config)).exists() else "",
    }


def _existing_path_string(value: str) -> str:
    if not str(value).strip():
        return ""
    return str(value) if Path(value).exists() else ""


def _resolve_recording_folder_for_nwb(
    config: dict[str, Any],
    row: dict[str, str],
    resolution_log: list[dict[str, Any]],
) -> Path:
    candidates = _recording_folder_candidates(config, row)
    return _first_existing_candidate(
        candidates,
        label="recording_folder",
        required_relative=Path("events")
        / str(digital_event_config(config).get("stream_folder", "NI-DAQmx-109.PXI-6133"))
        / "TTL"
        / "timestamps.npy",
        resolution_log=resolution_log,
        allow_missing_fallback=_recording_folder(row),
    )


def _resolve_processed_folder_for_nwb(
    config: dict[str, Any],
    row: dict[str, str],
    resolution_log: list[dict[str, Any]],
) -> Path:
    candidates = _processed_folder_candidates(config, row)
    return _first_existing_candidate(
        candidates,
        label=f"{row.get('probe_label', 'probe')}_processed_folder",
        required_relative=Path("summary.json"),
        resolution_log=resolution_log,
        allow_missing_fallback=Path(row["processed_folder"]),
    )


def _recording_folder_candidates(config: dict[str, Any], row: dict[str, str]) -> list[tuple[str, Path]]:
    local = _recording_folder(row)
    candidates: list[tuple[str, Path]] = [("registry_raw_folder", local)]
    archive_root = config.get("backup", {}).get("local_recording_archive", {}).get("root")
    local_raw_root = config.get("staging", {}).get("local_raw_root") or config.get("project", {}).get("raw_root")
    if archive_root:
        candidates.append(("local_recording_archive_root", _replace_root(Path(row.get("raw_folder", "")), Path(local_raw_root), Path(archive_root)) / local.relative_to(Path(row.get("raw_folder", "")))))
    if row.get("local_recording_archive_folder"):
        candidates.append(("registry_local_recording_archive_folder", Path(row["local_recording_archive_folder"]) / local.relative_to(Path(row.get("raw_folder", "")))))
    if row.get("server_raw_folder"):
        server_base = Path(row["server_raw_folder"])
        candidates.append(("server_raw_folder", server_base / local.relative_to(Path(row.get("raw_folder", "")))))
    return _dedupe_candidates(candidates)


def _processed_folder_candidates(config: dict[str, Any], row: dict[str, str]) -> list[tuple[str, Path]]:
    processed = Path(row["processed_folder"])
    candidates: list[tuple[str, Path]] = [("registry_processed_folder", processed)]
    local_raw = Path(row.get("raw_folder", ""))
    relative_to_raw = _safe_relative(processed, local_raw)
    archive_root = config.get("backup", {}).get("local_recording_archive", {}).get("root")
    local_raw_root = config.get("staging", {}).get("local_raw_root") or config.get("project", {}).get("raw_root")
    if archive_root:
        candidates.append(("local_recording_archive_root", _replace_root(local_raw, Path(local_raw_root), Path(archive_root)) / relative_to_raw))
    if row.get("local_recording_archive_folder"):
        candidates.append(("registry_local_recording_archive_folder", Path(row["local_recording_archive_folder"]) / relative_to_raw))
    if row.get("derived_local_backup_folder"):
        candidates.append(("registry_derived_local_backup_folder", Path(row["derived_local_backup_folder"]) / relative_to_raw))
    if config.get("backup", {}).get("derived_outputs", {}).get("local_archive_root"):
        candidates.append(
            (
                "derived_local_archive_root",
                _replace_root(local_raw, Path(config.get("project", {}).get("raw_root", local_raw.parent)), Path(config["backup"]["derived_outputs"]["local_archive_root"])) / relative_to_raw,
            )
        )
    if row.get("server_raw_folder"):
        candidates.append(("server_raw_folder", Path(row["server_raw_folder"]) / relative_to_raw))
    return _dedupe_candidates(candidates)


def _first_existing_candidate(
    candidates: list[tuple[str, Path]],
    *,
    label: str,
    required_relative: Path,
    resolution_log: list[dict[str, Any]],
    allow_missing_fallback: Path,
) -> Path:
    checked = []
    for source, candidate in candidates:
        required = candidate / required_relative
        exists = required.exists()
        checked.append({"source": source, "path": str(candidate), "required": str(required), "exists": bool(exists)})
        if exists:
            resolution_log.append({"label": label, "selected": str(candidate), "checked": checked})
            return candidate
    resolution_log.append({"label": label, "selected": str(allow_missing_fallback), "checked": checked, "warning": "no existing candidate found"})
    checked_text = "\n".join(f"- {item['source']}: {item['required']} exists={item['exists']}" for item in checked)
    raise FileNotFoundError(f"No usable NWB source found for {label}. Checked:\n{checked_text}")


def _replace_root(path: Path, old_root: Path, new_root: Path) -> Path:
    try:
        return new_root / path.resolve().relative_to(old_root.resolve())
    except Exception:
        return new_root / path.name


def _safe_relative(path: Path, parent: Path) -> Path:
    try:
        return path.resolve().relative_to(parent.resolve())
    except Exception:
        return Path(path.name)


def _dedupe_candidates(candidates: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    deduped: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source, path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source, path))
    return deduped


def _source_resolution_filename(config: dict[str, Any]) -> str:
    return str(config.get("mouse_arena_nwb", {}).get("source_resolution", {}).get("log_filename", "nwb_source_resolution.json"))


def _session_probe_rows(config: dict[str, Any], selected: dict[str, str]) -> list[dict[str, str]]:
    nwb_config = config.get("mouse_arena_nwb", {})
    wanted = list(nwb_config.get("probes") or ["ProbeA", "ProbeB"])
    rows = [
        row
        for row in read_registry(config["project"]["registry_csv"])
        if row.get("raw_folder") == selected.get("raw_folder") and row.get("probe_label") in wanted
    ]
    by_probe = {row.get("probe_label", ""): row for row in rows}
    missing = [probe for probe in wanted if probe not in by_probe]
    if missing:
        raise ValueError(f"Missing registry rows for probe(s): {', '.join(missing)}")
    ordered = [by_probe[probe] for probe in wanted]
    source_fallback_enabled = bool(nwb_config.get("source_resolution", {}).get("enabled", True))
    if nwb_config.get("require_complete_probes", True) and not source_fallback_enabled:
        incomplete = [row["session_id"] for row in ordered if row.get("status") != "complete"]
        if incomplete:
            raise ValueError(f"NWB export requires complete probe rows: {', '.join(incomplete)}")
    return ordered


def _recording_folder(row: dict[str, str]) -> Path:
    return (
        Path(row["raw_folder"])
        / "Record Node 101"
        / row.get("open_ephys_experiment_name", "experiment1")
        / row.get("open_ephys_block_index", "recording1")
    )


def _nwb_filename(config: dict[str, Any], row: dict[str, str]) -> str:
    protocol = nwb_protocol_for_row(config, row)
    template = str(
        config.get("nwb", {}).get("output_filename_template")
        or config.get("mouse_arena_nwb", {}).get(
            "output_filename_template",
            "{mouse_id}_{recording_date}_{recording_time}_{protocol}.nwb",
        )
    )
    recording_date, recording_time = _date_time_from_raw_folder(row)
    return template.format(
        mouse_id=row.get("mouse_id") or row.get("animal_id", "mouse"),
        recording_date=row.get("recording_date") or recording_date,
        recording_time=recording_time,
        session_id=_session_base_id(row),
        protocol=protocol,
    )


def _session_base_id(row: dict[str, str]) -> str:
    session_id = row.get("session_id", "")
    return re.sub(r"_Probe[A-Z].*$", "", session_id)


def _date_time_from_raw_folder(row: dict[str, str]) -> tuple[str, str]:
    raw_name = Path(row.get("raw_folder", "")).name
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", raw_name)
    if match:
        return match.group(1), match.group(2)
    return row.get("recording_date", ""), "00-00-00"


def _find_behavior_folder(config: dict[str, Any], row: dict[str, str]) -> Path:
    behavior = config.get("mouse_arena_nwb", {}).get("behavior", {})
    root = Path(str(behavior.get("root", "")))
    mouse = row.get("mouse_id") or row.get("animal_id", "")
    recording_date, _ = _date_time_from_raw_folder(row)
    base = root / mouse
    candidates = [path for path in base.glob(f"{recording_date}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No behavior folders found under {base} for {recording_date}")
    target_dt = _parse_recording_datetime(row)
    candidates = [path for path in candidates if (path / _behavior_filename(config, "events_filename")).exists()]
    if not candidates:
        raise FileNotFoundError(f"No behavior folders with events.csv found under {base} for {recording_date}")
    return min(candidates, key=lambda path: abs((_datetime_from_behavior_name(path.name) - target_dt).total_seconds()))


def _parse_recording_datetime(row: dict[str, str]) -> datetime:
    date, time = _date_time_from_raw_folder(row)
    return datetime.strptime(f"{date}_{time}", "%Y-%m-%d_%H-%M-%S")


def _datetime_from_behavior_name(name: str) -> datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    if not match:
        return datetime.min
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def _behavior_filename(config: dict[str, Any], key: str) -> str:
    return str(config.get("mouse_arena_nwb", {}).get("behavior", {}).get(key, "events.csv"))


def _load_digital_events(
    config: dict[str, Any],
    recording_folder: Path,
    *,
    override_recording_folder: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    return load_digital_events(
        config,
        recording_folder=recording_folder,
        override_recording_folder=override_recording_folder,
    )


def _digital_time_zero(config: dict[str, Any], recording_folder: Path, stream: str) -> float:
    mode = str(config.get("mouse_arena_nwb", {}).get("digital_events", {}).get("time_zero", "nidaq_continuous_start"))
    if mode != "nidaq_continuous_start":
        return 0.0
    timestamps = np.load(recording_folder / "continuous" / stream / "timestamps.npy", mmap_mode="r")
    return float(timestamps[0])


def _digital_line_config(config: dict[str, Any]) -> dict[str, int]:
    digital = digital_event_config(config)
    configured = {
        safe_event_name(name): int(line)
        for line, name in dict(digital.get("line_names", {})).items()
        if str(line).lstrip("-").isdigit() and safe_event_name(name)
    }
    configured.setdefault("trial_start", int(digital.get("trial_starts_line", 5)))
    configured.setdefault("reward", int(digital.get("rewards_line", 4)))
    configured.setdefault("game_frame", int(digital.get("frames_line", 6)))
    return configured


def _estimate_behavior_alignment(config: dict[str, Any], digital_events: pd.DataFrame, sync_events: pd.DataFrame) -> dict[str, Any]:
    alignment_config = config.get("mouse_arena_nwb", {}).get("clock_alignment", {})
    method = str(alignment_config.get("method", "game_frame_clock_fit"))
    seed = _trial_start_alignment(digital_events, sync_events)
    if method == "game_frame_clock_fit":
        frame_alignment = _game_frame_clock_alignment(config, digital_events, sync_events)
        if frame_alignment is not None:
            return frame_alignment
    if method == "causal_frame_fit":
        frame_alignment = _causal_frame_alignment(config, digital_events, sync_events, seed)
        if frame_alignment is not None:
            return frame_alignment
    return seed


def _trial_start_alignment(digital_events: pd.DataFrame, sync_events: pd.DataFrame) -> dict[str, Any]:
    if sync_events.empty or digital_events.empty:
        return {"mode": "none", "slope": 1.0, "intercept": 0.0, "n_pulses": 0}
    sync = sync_events[sync_events["event_type"] == "trial_start"]["t"].to_numpy(dtype=float)
    ttl = digital_events[
        (digital_events["event_name"] == "trial_start") & (digital_events["edge"] == "rising")
    ]["time"].to_numpy(dtype=float)
    n = min(len(sync), len(ttl))
    if n == 0:
        return {"mode": "none", "slope": 1.0, "intercept": 0.0, "n_pulses": 0}
    offsets = ttl[:n] - sync[:n]
    slope, intercept = np.polyfit(sync[:n], ttl[:n], 1)
    residuals = ttl[:n] - (slope * sync[:n] + intercept)
    return {
        "mode": "linear_trial_start_fit",
        "slope": float(slope),
        "intercept": float(intercept),
        "n_pulses": int(n),
        "offset_first_seconds": float(offsets[0]),
        "offset_median_seconds": float(np.nanmedian(offsets)),
        "offset_last_seconds": float(offsets[-1]),
        "residual_median_abs_seconds": float(np.nanmedian(np.abs(residuals))),
        "residual_max_abs_seconds": float(np.nanmax(np.abs(residuals))),
    }


def _causal_frame_alignment(
    config: dict[str, Any],
    digital_events: pd.DataFrame,
    sync_events: pd.DataFrame,
    seed_alignment: dict[str, Any],
) -> dict[str, Any] | None:
    alignment_config = config.get("mouse_arena_nwb", {}).get("clock_alignment", {})
    behavior_event = str(alignment_config.get("behavior_sync_event", "game_frame"))
    ephys_event = _normalize_digital_event_name(str(alignment_config.get("ephys_event", "game_frame")))
    max_lag = float(alignment_config.get("max_lag_seconds", 0.025))
    behavior_times = sync_events[sync_events["event_type"] == behavior_event]["t"].to_numpy(dtype=float)
    ephys_times = digital_events[
        (digital_events["event_name"] == ephys_event) & (digital_events["edge"] == "rising")
    ]["time"].to_numpy(dtype=float)
    if len(behavior_times) < 2 or len(ephys_times) < 2:
        return None

    mapped_behavior = _behavior_to_ephys_time(behavior_times, seed_alignment)
    positions = np.searchsorted(ephys_times, mapped_behavior, side="left")
    valid = positions < len(ephys_times)
    if not np.any(valid):
        return None
    candidate_behavior = behavior_times[valid]
    candidate_ephys = ephys_times[positions[valid]]
    lags = candidate_ephys - mapped_behavior[valid]
    lag_mask = (lags >= 0.0) & (lags <= max_lag)
    candidate_behavior = candidate_behavior[lag_mask]
    candidate_ephys = candidate_ephys[lag_mask]
    matched_positions = positions[valid][lag_mask]
    lags = lags[lag_mask]
    if len(candidate_behavior) < 2:
        return None

    if bool(alignment_config.get("deduplicate_ephys_matches", True)):
        keep_indices = []
        seen: set[int] = set()
        order = np.argsort(lags)
        for index in order:
            position = int(matched_positions[index])
            if position in seen:
                continue
            seen.add(position)
            keep_indices.append(int(index))
        keep_indices = sorted(keep_indices)
        candidate_behavior = candidate_behavior[keep_indices]
        candidate_ephys = candidate_ephys[keep_indices]
        lags = lags[keep_indices]

    if len(candidate_behavior) < 2:
        return None

    slope, intercept = np.polyfit(candidate_behavior, candidate_ephys, 1)
    residuals = candidate_ephys - (slope * candidate_behavior + intercept)
    return {
        "mode": "causal_frame_fit",
        "seed_alignment": seed_alignment,
        "behavior_sync_event": behavior_event,
        "ephys_event": ephys_event,
        "slope": float(slope),
        "intercept": float(intercept),
        "n_pulses": int(len(candidate_behavior)),
        "raw_behavior_frame_count": int(len(behavior_times)),
        "raw_ephys_frame_count": int(len(ephys_times)),
        "max_lag_seconds": float(max_lag),
        "lag_median_seconds": float(np.nanmedian(lags)),
        "lag_q05_seconds": float(np.nanpercentile(lags, 5)),
        "lag_q95_seconds": float(np.nanpercentile(lags, 95)),
        "lag_max_seconds": float(np.nanmax(lags)),
        "residual_median_abs_seconds": float(np.nanmedian(np.abs(residuals))),
        "residual_max_abs_seconds": float(np.nanmax(np.abs(residuals))),
    }


def _game_frame_clock_alignment(
    config: dict[str, Any],
    digital_events: pd.DataFrame,
    sync_events: pd.DataFrame,
) -> dict[str, Any] | None:
    alignment_config = config.get("mouse_arena_nwb", {}).get("clock_alignment", {})
    behavior_event = str(alignment_config.get("behavior_sync_event", "game_frame"))
    ephys_event = _normalize_digital_event_name(str(alignment_config.get("ephys_event", "game_frame")))
    behavior_times = sync_events[sync_events["event_type"] == behavior_event]["t"].to_numpy(dtype=float)
    ephys_times = digital_events[
        (digital_events["event_name"] == ephys_event) & (digital_events["edge"] == "rising")
    ]["time"].to_numpy(dtype=float)
    if len(behavior_times) < 2 or len(ephys_times) < 2:
        return None
    max_abs_residual = alignment_config.get("max_abs_match_residual_seconds")
    alignment, matches = _fit_behavior_clock_from_game_frames(
        behavior_times,
        ephys_times,
        behavior_event=behavior_event,
        ephys_event=ephys_event,
        max_abs_residual_s=float(max_abs_residual) if max_abs_residual is not None else None,
    )
    residuals = matches["fit_residual_seconds"].to_numpy(dtype=float)
    alignment.update(
        {
            "residual_median_abs_seconds": float(np.nanmedian(np.abs(residuals))) if len(residuals) else math.nan,
            "residual_max_abs_seconds": float(np.nanmax(np.abs(residuals))) if len(residuals) else math.nan,
        }
    )
    return alignment


def _fit_behavior_clock_from_game_frames(
    behavior_times: np.ndarray,
    ephys_times: np.ndarray,
    *,
    behavior_event: str = "game_frame",
    ephys_event: str = "game_frame",
    max_abs_residual_s: float | None = None,
    n_refinements: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    behavior_times = np.asarray(behavior_times, dtype=float)
    ephys_times = np.asarray(ephys_times, dtype=float)
    if len(behavior_times) < 2 or len(ephys_times) < 2:
        raise ValueError("Need at least two behavior and ephys game-frame events to fit a clock.")
    behavior_interval = float(np.nanmedian(np.diff(behavior_times)))
    ephys_interval = float(np.nanmedian(np.diff(ephys_times)))
    if max_abs_residual_s is None:
        max_abs_residual_s = min(0.010, 0.45 * min(behavior_interval, ephys_interval))

    slope = (ephys_times[-1] - ephys_times[0]) / (behavior_times[-1] - behavior_times[0])
    intercept = ephys_times[0] - slope * behavior_times[0]
    stages: list[dict[str, Any]] = [{"stage": "span_seed", "slope": float(slope), "intercept": float(intercept)}]
    matches = pd.DataFrame()
    for refinement in range(n_refinements):
        matches = _nearest_unique_frame_matches(behavior_times, ephys_times, slope, intercept, max_abs_residual_s)
        if len(matches) < 2:
            raise ValueError(f"Only {len(matches)} game-frame matches survived the residual threshold.")
        slope, intercept = np.polyfit(matches["behavior_time"], matches["ephys_time"], 1)
        refit_residuals = matches["ephys_time"] - (slope * matches["behavior_time"] + intercept)
        stages.append(
            {
                "stage": f"refine_{refinement + 1}",
                "slope": float(slope),
                "intercept": float(intercept),
                "n_matches": int(len(matches)),
                "median_abs_residual_seconds": float(np.nanmedian(np.abs(refit_residuals))),
                "max_abs_residual_seconds": float(np.nanmax(np.abs(refit_residuals))),
            }
        )
    matches = _nearest_unique_frame_matches(behavior_times, ephys_times, slope, intercept, max_abs_residual_s)
    residuals = matches["ephys_time"] - (slope * matches["behavior_time"] + intercept)
    matches["fit_residual_seconds"] = residuals
    alignment = {
        "mode": "game_frame_clock_fit",
        "behavior_sync_event": behavior_event,
        "ephys_event": ephys_event,
        "slope": float(slope),
        "intercept": float(intercept),
        "n_matched_pulses": int(len(matches)),
        "raw_behavior_frame_count": int(len(behavior_times)),
        "raw_ephys_game_frame_count": int(len(ephys_times)),
        "max_abs_match_residual_seconds": float(max_abs_residual_s),
        "behavior_median_interval_seconds": behavior_interval,
        "ephys_median_interval_seconds": ephys_interval,
        "fit_residual_median_abs_seconds": float(np.nanmedian(np.abs(residuals))),
        "fit_residual_q95_abs_seconds": float(np.nanpercentile(np.abs(residuals), 95)),
        "fit_residual_max_abs_seconds": float(np.nanmax(np.abs(residuals))),
        "stages": stages,
    }
    return alignment, matches


def _nearest_unique_frame_matches(
    behavior_times: np.ndarray,
    ephys_times: np.ndarray,
    slope: float,
    intercept: float,
    max_abs_residual_s: float,
) -> pd.DataFrame:
    mapped = slope * behavior_times + intercept
    right = np.searchsorted(ephys_times, mapped, side="left")
    rows = []
    for behavior_index, right_index in enumerate(right):
        candidates = []
        if 0 <= right_index < len(ephys_times):
            candidates.append(right_index)
        if 0 <= right_index - 1 < len(ephys_times):
            candidates.append(right_index - 1)
        if not candidates:
            continue
        ephys_index = min(candidates, key=lambda index: abs(ephys_times[index] - mapped[behavior_index]))
        residual = ephys_times[ephys_index] - mapped[behavior_index]
        if abs(residual) <= max_abs_residual_s:
            rows.append(
                {
                    "behavior_frame_index": int(behavior_index),
                    "ephys_frame_index": int(ephys_index),
                    "behavior_time": float(behavior_times[behavior_index]),
                    "ephys_time": float(ephys_times[ephys_index]),
                    "mapped_time": float(mapped[behavior_index]),
                    "residual_seconds": float(residual),
                }
            )
    matches = pd.DataFrame(rows)
    if matches.empty:
        return matches
    return (
        matches.assign(abs_residual_seconds=matches["residual_seconds"].abs())
        .sort_values("abs_residual_seconds")
        .drop_duplicates("ephys_frame_index", keep="first")
        .sort_values("behavior_frame_index")
        .reset_index(drop=True)
    )


def _build_trials_table(
    config: dict[str, Any],
    behavior_events: pd.DataFrame,
    digital_events: pd.DataFrame,
    behavior_to_ephys_alignment: dict[str, Any],
) -> pd.DataFrame:
    trials = behavior_events[behavior_events["event_type"].isin(_trial_like_event_types())].copy()
    trials.insert(0, "session_event_id", trials.index.to_numpy(dtype=int))
    trials = trials.reset_index(drop=True)
    trials["behavior_t_start"] = pd.to_numeric(trials["t_start"], errors="coerce")
    trials["behavior_t_end"] = pd.to_numeric(trials["t_end"], errors="coerce")
    trials["behavior_aligned_start_time"] = _behavior_to_ephys_time(trials["behavior_t_start"], behavior_to_ephys_alignment)
    trials["behavior_aligned_end_time"] = _behavior_to_ephys_time(trials["behavior_t_end"], behavior_to_ephys_alignment)
    trials = _expand_extra_json(trials)
    trials["trial_like_ordinal"] = np.arange(1, len(trials) + 1, dtype=int)

    trial_ttls = digital_events[
        (digital_events["event_name"] == "trial_start") & (digital_events["edge"] == "rising")
    ].set_index("ordinal")["time"].to_dict()
    reward_ttls = digital_events[
        (digital_events["event_name"] == "reward") & (digital_events["edge"] == "rising")
    ].set_index("ordinal")["time"].to_dict()
    trials["ttl_trial_start_time"] = [
        trial_ttls.get(_safe_int(ordinal), math.nan) for ordinal in trials["trial_like_ordinal"]
    ]
    trials["source_event_type"] = trials["event_type"]
    trials["event_type"] = "trial"
    trials["behavior_trial_id"] = trials.get("trial_id", pd.Series(index=trials.index, dtype=float))
    trials["trial_id"] = trials["trial_like_ordinal"].astype("Int64")
    trials["trial_number"] = trials["trial_id"]
    trials["outcome"] = trials.apply(_derive_trial_outcome, axis=1)
    trials["reward_time"] = [
        reward_ttls.get(_safe_int(reward_count), math.nan) if outcome == "hit" else math.nan
        for reward_count, outcome in zip(trials.get("reward_count", pd.Series(dtype=float)), trials["outcome"])
    ]
    trials["trial_start_time"] = trials["ttl_trial_start_time"]
    trials["t_start"] = trials["trial_start_time"]
    trials["t_end"] = trials["behavior_aligned_end_time"]
    trials.loc[trials["t_end"].le(trials["t_start"]), "t_end"] = np.nan
    trials["start_time"] = trials["t_start"]
    trials["stop_time"] = trials["t_end"]
    trials["rewarded"] = trials["reward_time"].notna() | trials["outcome"].eq("hit")
    trials["start_alignment_residual_seconds"] = trials["ttl_trial_start_time"] - trials["behavior_aligned_start_time"]
    first_columns = [
        "start_time",
        "stop_time",
        "trial_id",
        "trial_number",
        "event_type",
        "outcome",
        "source_event_type",
        "behavior_trial_id",
        "trial_start_time",
        "reward_time",
        "behavior_t_start",
        "behavior_t_end",
        "behavior_aligned_start_time",
        "behavior_aligned_end_time",
        "start_alignment_residual_seconds",
        "trial_like_ordinal",
        "session_event_id",
        "rewarded",
        "reward_count",
        "stim_id",
        "contrast",
        "orientation",
    ]
    ordered = [column for column in first_columns if column in trials.columns]
    ordered.extend(column for column in trials.columns if column not in ordered and column != "extra")
    return trials[ordered].reset_index(drop=True)


def _build_session_events_dataframe(
    behavior_events: pd.DataFrame,
    digital_events: pd.DataFrame,
    behavior_to_ephys_alignment: dict[str, Any],
) -> pd.DataFrame:
    events = behavior_events.copy()
    events.insert(0, "session_event_id", np.arange(len(events), dtype=int))
    events["behavior_t_start"] = pd.to_numeric(events["t_start"], errors="coerce")
    events["behavior_t_end"] = pd.to_numeric(events["t_end"], errors="coerce")
    events["behavior_aligned_start_time"] = _behavior_to_ephys_time(events["behavior_t_start"], behavior_to_ephys_alignment)
    events["behavior_aligned_end_time"] = _behavior_to_ephys_time(events["behavior_t_end"], behavior_to_ephys_alignment)

    trial_like_mask = events["event_type"].isin(_trial_like_event_types())
    events["trial_like_ordinal"] = np.nan
    events.loc[trial_like_mask, "trial_like_ordinal"] = np.arange(1, int(trial_like_mask.sum()) + 1, dtype=int)
    trial_ttls = digital_events[
        (digital_events["event_name"] == "trial_start") & (digital_events["edge"] == "rising")
    ].set_index("ordinal")["time"].to_dict()
    reward_ttls = digital_events[
        (digital_events["event_name"] == "reward") & (digital_events["edge"] == "rising")
    ].set_index("ordinal")["time"].to_dict()
    events["ttl_trial_start_time"] = [
        trial_ttls.get(_safe_int(ordinal), math.nan) for ordinal in events["trial_like_ordinal"]
    ]
    expanded_for_outcome = _expand_extra_json(events)
    outcomes = expanded_for_outcome.apply(_derive_trial_outcome, axis=1)
    events["reward_time"] = [
        reward_ttls.get(_safe_int(reward_count), math.nan) if outcome == "hit" else math.nan
        for reward_count, outcome in zip(events.get("reward_count", pd.Series(dtype=float)), outcomes)
    ]
    events["ephys_start_time"] = events["behavior_aligned_start_time"]
    ttl_mask = events["ttl_trial_start_time"].notna()
    events.loc[ttl_mask, "ephys_start_time"] = events.loc[ttl_mask, "ttl_trial_start_time"]
    events["ephys_end_time"] = events["behavior_aligned_end_time"]
    events["ephys_start_time_source"] = np.where(ttl_mask, "ttl_trial_start", "behavior_linear_fit")
    events["ephys_end_time_source"] = "behavior_linear_fit"
    events["start_alignment_residual_seconds"] = events["ttl_trial_start_time"] - events["behavior_aligned_start_time"]
    events = _expand_extra_json(events)

    first_columns = [
        "session_event_id",
        "event_type",
        "phase_id",
        "trial_id",
        "trial_like_ordinal",
        "ephys_start_time",
        "ephys_end_time",
        "ephys_start_time_source",
        "ephys_end_time_source",
        "ttl_trial_start_time",
        "reward_time",
        "start_alignment_residual_seconds",
        "behavior_t_start",
        "behavior_t_end",
        "behavior_aligned_start_time",
        "behavior_aligned_end_time",
        "rewarded",
        "reward_count",
        "stim_id",
        "contrast",
        "orientation",
        "user_assisted",
        "x",
        "y",
        "heading",
    ]
    ordered = [column for column in first_columns if column in events.columns]
    ordered.extend(column for column in events.columns if column not in ordered and column not in {"t_start", "t_end"})
    return events[ordered].reset_index(drop=True)


def _expand_extra_json(frame: pd.DataFrame) -> pd.DataFrame:
    extras: list[dict[str, Any]] = []
    for value in frame.get("extra", pd.Series(index=frame.index, dtype=object)):
        if isinstance(value, str) and value.strip():
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        extras.append({f"extra_{key}": item for key, item in payload.items()})
    extra_frame = pd.DataFrame(extras, index=frame.index)
    return pd.concat([frame.drop(columns=["extra"], errors="ignore"), extra_frame], axis=1)


def _aligned_behavior_events(behavior_events: pd.DataFrame, alignment: dict[str, Any]) -> pd.DataFrame:
    aligned = behavior_events.copy()
    aligned["aligned_start_time"] = _behavior_to_ephys_time(pd.to_numeric(aligned["t_start"], errors="coerce"), alignment)
    aligned["aligned_end_time"] = _behavior_to_ephys_time(pd.to_numeric(aligned["t_end"], errors="coerce"), alignment)
    return aligned


def _build_frame_sync_residuals(
    digital_events: pd.DataFrame,
    sync_events: pd.DataFrame,
    alignment: dict[str, Any],
) -> pd.DataFrame:
    behavior_times = sync_events[sync_events["event_type"] == "game_frame"]["t"].to_numpy(dtype=float)
    ephys_times = digital_events[
        (digital_events["event_name"] == "game_frame") & (digital_events["edge"] == "rising")
    ]["time"].to_numpy(dtype=float)
    if len(behavior_times) == 0 or len(ephys_times) == 0:
        return pd.DataFrame(
            columns=[
                "ephys_game_frame_ordinal",
                "ephys_game_frame_time",
                "nearest_behavior_game_frame_ordinal",
                "nearest_behavior_game_frame_time",
                "nearest_behavior_game_frame_ephys_time",
                "residual_seconds",
                "residual_ms",
            ]
        )
    behavior_ephys_times = _behavior_to_ephys_time(behavior_times, alignment)
    positions = np.searchsorted(behavior_ephys_times, ephys_times, side="left")
    rows = []
    for ephys_index, position in enumerate(positions):
        candidates = []
        if 0 <= position < len(behavior_ephys_times):
            candidates.append(position)
        if 0 <= position - 1 < len(behavior_ephys_times):
            candidates.append(position - 1)
        if not candidates:
            continue
        behavior_index = min(candidates, key=lambda index: abs(ephys_times[ephys_index] - behavior_ephys_times[index]))
        residual = ephys_times[ephys_index] - behavior_ephys_times[behavior_index]
        rows.append(
            {
                "ephys_game_frame_ordinal": int(ephys_index + 1),
                "ephys_game_frame_time": float(ephys_times[ephys_index]),
                "nearest_behavior_game_frame_ordinal": int(behavior_index + 1),
                "nearest_behavior_game_frame_time": float(behavior_times[behavior_index]),
                "nearest_behavior_game_frame_ephys_time": float(behavior_ephys_times[behavior_index]),
                "residual_seconds": float(residual),
                "residual_ms": float(residual * 1000),
            }
        )
    return pd.DataFrame(rows)


def _residual_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "residual_ms" not in frame.columns:
        return {"count": 0}
    residuals = frame["residual_ms"].to_numpy(dtype=float)
    return {
        "count": int(len(residuals)),
        "mean": float(np.nanmean(residuals)),
        "median": float(np.nanmedian(residuals)),
        "median_abs": float(np.nanmedian(np.abs(residuals))),
        "q05": float(np.nanpercentile(residuals, 5)),
        "q95": float(np.nanpercentile(residuals, 95)),
        "max_abs": float(np.nanmax(np.abs(residuals))),
    }


def _behavior_to_ephys_time(values: Any, alignment: dict[str, Any]) -> Any:
    slope = float(alignment.get("slope", 1.0))
    intercept = float(alignment.get("intercept", 0.0))
    return values * slope + intercept


def _trial_like_event_types() -> tuple[str, ...]:
    return ("trial", "fall_trial")


def _derive_trial_outcome(row: pd.Series) -> str:
    source_event_type = str(row.get("source_event_type", row.get("event_type", ""))).lower()
    extra_outcome = str(row.get("extra_outcome", "")).strip().lower()
    extra_reason = str(row.get("extra_reason", "")).strip().lower()
    if source_event_type == "fall_trial" or "fall" in extra_outcome or "fall" in extra_reason:
        return "fall"
    if extra_outcome in {"correct", "hit", "rewarded", "success"}:
        return "hit"
    if _as_bool(row.get("rewarded", False)):
        return "hit"
    return "miss"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _normalize_digital_event_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if cleaned in {"frame", "frames"}:
        return "game_frame"
    if cleaned in {"trial_starts", "trial"}:
        return "trial_start"
    if cleaned in {"rewards"}:
        return "reward"
    return cleaned


def _safe_output_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")


def _build_units_table(
    rows: list[dict[str, str]],
    *,
    config: dict[str, Any],
    resolution_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    unit_rows: list[dict[str, Any]] = []
    spike_times_by_id: dict[int, np.ndarray] = {}
    nwb_unit_id = 0
    for row in rows:
        processed = _resolve_processed_folder_for_nwb(config, row, resolution_log)
        sorter = _sorter_output_folder(row, config=config, processed=processed, resolution_log=resolution_log)
        sampling_frequency = float(row.get("sampling_frequency") or 30000.0)
        spike_clusters = np.load(sorter / "spike_clusters.npy").reshape(-1)
        spike_samples = np.load(sorter / "spike_times.npy").reshape(-1)
        amplitudes = _load_optional_array(sorter / "amplitudes.npy")
        unit_summary = _read_unit_table(processed / "unit_summary.csv")
        metrics = _read_unit_table(processed / "quality_metrics.csv")
        template_metrics = _read_unit_table(processed / "template_metrics.csv")
        labels = _read_latest_unitrefine(processed)
        ks_labels = _read_cluster_tsv(sorter / "cluster_group.tsv", "kilosort_label")
        ks_unit_labels = _read_cluster_tsv(sorter / "cluster_KSLabel.tsv", "kilosort_ks_label")
        ks_amplitude = _read_cluster_tsv(sorter / "cluster_Amplitude.tsv", "kilosort_amplitude")
        ks_contam = _read_cluster_tsv(sorter / "cluster_ContamPct.tsv", "kilosort_contamination_pct")
        probe_label = row.get("probe_label", "")
        probe_id = probe_label.replace("Probe", "") or probe_label
        for unit_id in sorted(np.unique(spike_clusters).tolist(), key=lambda value: int(value)):
            mask = spike_clusters == unit_id
            spike_times = spike_samples[mask].astype(float) / sampling_frequency
            item: dict[str, Any] = {
                "nwb_unit_id": nwb_unit_id,
                "unique_unit_id": f"{probe_id}_{int(unit_id)}",
                "probe_label": probe_label,
                "probe_id": probe_id,
                "source_unit_id": int(unit_id),
                "n_spikes": int(mask.sum()),
                "spike_time_first": float(np.nanmin(spike_times)) if spike_times.size else math.nan,
                "spike_time_last": float(np.nanmax(spike_times)) if spike_times.size else math.nan,
                "sampling_frequency": sampling_frequency,
                "processed_folder": str(processed),
                "sorter_output_folder": str(sorter),
            }
            if amplitudes is not None and len(amplitudes) == len(spike_clusters):
                amp = np.asarray(amplitudes).reshape(-1)[mask]
                item["kilosort_spike_amplitude_median"] = float(np.nanmedian(amp)) if amp.size else math.nan
            for source in [ks_labels, ks_unit_labels, ks_amplitude, ks_contam, unit_summary, metrics, template_metrics, labels]:
                item.update(source.get(str(int(unit_id)), {}))
            unit_rows.append(item)
            spike_times_by_id[nwb_unit_id] = spike_times
            nwb_unit_id += 1
    return pd.DataFrame(unit_rows), spike_times_by_id


def _read_input_overrides(output_folder: Path) -> dict[str, str]:
    path = output_folder / "input_overrides.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    overrides = payload.get("active_overrides", payload)
    return {
        key: str(value)
        for key, value in overrides.items()
        if key in {"trials_csv", "units_csv"} and value
    }


def _load_trials_override(path: str | Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    if "start_time" not in trials.columns and "t_start" in trials.columns:
        trials["start_time"] = trials["t_start"]
    if "stop_time" not in trials.columns and "t_end" in trials.columns:
        trials["stop_time"] = trials["t_end"]
    if "start_time" not in trials.columns or "stop_time" not in trials.columns:
        raise ValueError(f"Trials CSV must include start_time/stop_time or t_start/t_end: {path}")
    return trials


def _load_units_override(
    path: str | Path,
    *,
    baseline_units: pd.DataFrame,
    baseline_spike_times: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[str, Any]]:
    override = pd.read_csv(path)
    if override.empty:
        raise ValueError(f"Units CSV is empty: {path}")
    baseline_lookup = _baseline_unit_spike_lookup(baseline_units, baseline_spike_times)
    new_spike_times: dict[int, np.ndarray] = {}
    matched = 0
    parsed = 0
    for new_id, (_, row) in enumerate(override.iterrows()):
        spike_times = _lookup_override_spike_times(row, baseline_lookup)
        if spike_times is not None:
            matched += 1
        else:
            spike_times = _parse_spike_times_from_row(row)
            if spike_times.size:
                parsed += 1
        new_spike_times[new_id] = spike_times
    override = override.drop(columns=["nwb_unit_id"], errors="ignore").copy()
    override.insert(0, "nwb_unit_id", np.arange(len(override), dtype=int))
    return override, new_spike_times, {
        "rows": int(len(override)),
        "spike_times_matched_from_pipeline": int(matched),
        "spike_times_parsed_from_csv": int(parsed),
        "spike_times_missing": int(len(override) - matched - parsed),
    }


def _baseline_unit_spike_lookup(units: pd.DataFrame, spike_times: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    for _, row in units.iterrows():
        nwb_id = _safe_int(row.get("nwb_unit_id"))
        if nwb_id is None:
            continue
        spikes = spike_times.get(nwb_id, np.array([], dtype=float))
        for key in _unit_match_keys(row):
            lookup[key] = spikes
    return lookup


def _lookup_override_spike_times(row: pd.Series, lookup: dict[str, np.ndarray]) -> np.ndarray | None:
    for key in _unit_match_keys(row):
        if key in lookup:
            return lookup[key]
    return None


def _unit_match_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    unique = row.get("unique_unit_id")
    if pd.notna(unique) and str(unique).strip():
        keys.append(f"unique:{str(unique).strip()}")
    probe = row.get("probe_label", row.get("probe_id", ""))
    unit = row.get("source_unit_id", row.get("unit_id", row.get("cluster_id", "")))
    if pd.notna(probe) and pd.notna(unit) and str(probe).strip() and str(unit).strip():
        unit_text = str(unit).strip()
        try:
            unit_text = str(int(float(unit_text)))
        except ValueError:
            pass
        keys.append(f"probe_unit:{str(probe).strip()}:{unit_text}")
    nwb_unit_id = row.get("nwb_unit_id")
    if pd.notna(nwb_unit_id) and str(nwb_unit_id).strip():
        keys.append(f"nwb:{int(float(nwb_unit_id))}")
    return keys


def _parse_spike_times_from_row(row: pd.Series) -> np.ndarray:
    value = row.get("spike_times", row.get("times", ""))
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = re.split(r"[\s,;]+", text.strip("[]()"))
        try:
            return np.asarray([float(item) for item in parsed if str(item).strip()], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def _sorter_output_folder(
    row: dict[str, str],
    *,
    config: dict[str, Any],
    processed: Path,
    resolution_log: list[dict[str, Any]],
) -> Path:
    candidates = [
        ("resolved_processed_kilosort4", processed / "kilosort4"),
        ("resolved_processed_sorter_output", processed / "kilosort4" / "sorter_output"),
        ("registry_sorter_output_folder", Path(row["sorter_output_folder"])),
        ("registry_sorter_output_subfolder", Path(row["sorter_output_folder"]) / "sorter_output"),
    ]
    return _first_existing_candidate(
        _dedupe_candidates(candidates),
        label=f"{row.get('probe_label', 'probe')}_sorter_output_folder",
        required_relative=Path("spike_clusters.npy"),
        resolution_log=resolution_log,
        allow_missing_fallback=Path(row["sorter_output_folder"]),
    )


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def _read_unit_table(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "unit_id" not in frame.columns and len(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: "unit_id"})
    if "unit_id" not in frame.columns:
        return {}
    frame["unit_id"] = frame["unit_id"].astype(str)
    return frame.set_index("unit_id").to_dict(orient="index")


def _read_latest_unitrefine(processed: Path) -> dict[str, dict[str, Any]]:
    root = processed / "unitrefine"
    if not root.exists():
        return {}
    candidates = sorted(root.glob("*/unit_labels.csv"), key=lambda path: path.stat().st_mtime)
    return _read_unit_table(candidates[-1]) if candidates else {}


def _read_cluster_tsv(path: Path, value_name: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    if "cluster_id" not in frame.columns:
        return {}
    value_columns = [column for column in frame.columns if column != "cluster_id"]
    if not value_columns:
        return {}
    value_column = value_columns[0]
    return {
        str(int(row["cluster_id"])): {value_name: row[value_column]}
        for _, row in frame.iterrows()
        if pd.notna(row["cluster_id"])
    }


def _analog_manifest(config: dict[str, Any], recording_folder: Path) -> dict[str, Any]:
    analog = config.get("mouse_arena_nwb", {}).get("analog", {})
    stream = str(analog.get("stream_folder", "NI-DAQmx-109.PXI-6133"))
    dat = recording_folder / "continuous" / stream / "continuous.dat"
    names = analog.get("channel_names", {})
    return {
        "include": bool(analog.get("include", False)),
        "stream_folder": stream,
        "continuous_dat": str(dat),
        "channel_names": names,
    }


def _write_nwb_file(config: dict[str, Any], tables: MouseArenaNwbTables) -> None:
    try:
        from pynwb import NWBHDF5IO, NWBFile
        from pynwb.file import Subject
    except ImportError as exc:
        raise RuntimeError("pynwb is required for NWB export. Install/update the spikeinterface environment from environment_spikeinterface.yml.") from exc

    manifest = tables.manifest
    session_start_time = _nwb_session_start_time(manifest)
    protocol = str(manifest.get("protocol") or "mouse_arena")
    session_meta = manifest.get("session_metadata") or _session_metadata(config, protocol=protocol)
    subject_meta = manifest.get("subject") or _subject_metadata(config, {"mouse_id": manifest.get("mouse_id", "")}, session_start_time)
    nwbfile = NWBFile(
        session_description=str(session_meta.get("session_description") or f"{protocol} Neuropixels session."),
        identifier=str(manifest["session_id"]),
        session_start_time=session_start_time,
        experimenter=_metadata_list_or_string(session_meta.get("experimenter", "")),
        lab=str(session_meta.get("lab") or "Denman Lab"),
        institution=str(session_meta.get("institution") or "University of Colorado Anschutz"),
        experiment_description=str(session_meta.get("experiment_description") or f"{protocol} experiment with Neuropixels recordings."),
        keywords=list(session_meta.get("keywords") or []),
        session_id=str(manifest["session_id"]),
    )
    nwbfile.subject = Subject(**_subject_kwargs_for_nwb(subject_meta))
    device = nwbfile.create_device(
        name=str(session_meta.get("device_name") or "DenmanLab_MouseArena_Neuropixels"),
        description=f"Denman Lab Neuropixels acquisition system for {protocol} recordings.",
    )
    _add_probe_electrode_groups_to_nwb(nwbfile, device, manifest)

    _add_trials_to_nwb(nwbfile, tables.trials)
    _add_units_to_nwb(nwbfile, tables.units, tables.spike_times)
    _add_digital_events_to_nwb(nwbfile, tables.digital_events)
    if config.get("mouse_arena_nwb", {}).get("analog", {}).get("include", False):
        _add_analog_to_nwb(config, nwbfile, tables.source_recording_folder)
    _add_pipeline_manifest_to_nwb(nwbfile, manifest)

    with NWBHDF5IO(str(tables.nwb_path), "w") as io:
        io.write(nwbfile)


def _add_pipeline_manifest_to_nwb(nwbfile, manifest: dict[str, Any]) -> None:
    manifest_text = json.dumps(manifest, indent=2, default=str)
    try:
        nwbfile.add_scratch(
            manifest_text,
            name="pipeline_nwb_manifest",
            description="Pipeline provenance manifest, including source paths, protocol, synchronization policy, and input files.",
        )
    except Exception:
        return


def _nwb_session_start_time(manifest: dict[str, Any]) -> datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", str(manifest.get("recording_folder", "")))
    if match:
        return datetime.strptime(match.group(1) + "_" + match.group(2), "%Y-%m-%d_%H-%M-%S").astimezone()
    return datetime.now().astimezone()


def _add_trials_to_nwb(nwbfile, trials: pd.DataFrame) -> None:
    column_types = _nwb_column_types(trials)
    for column in trials.columns:
        if column not in {"start_time", "stop_time"}:
            nwbfile.add_trial_column(name=column, description=f"Trial table column: {column}")
    for _, trial in trials.iterrows():
        start = _finite_or_none(trial.get("start_time"))
        stop = _finite_or_none(trial.get("stop_time"))
        if start is None or stop is None:
            continue
        payload = {
            column: _nwb_scalar(trial[column], column_types.get(column, "text"))
            for column in trials.columns
            if column not in {"start_time", "stop_time"}
        }
        nwbfile.add_trial(start_time=float(start), stop_time=float(stop), **payload)


def _add_units_to_nwb(nwbfile, units: pd.DataFrame, spike_times: dict[int, np.ndarray]) -> None:
    column_types = _nwb_column_types(units.drop(columns=["spike_times"], errors="ignore"))
    for column in units.columns:
        if column not in {"nwb_unit_id", "spike_times"}:
            nwbfile.add_unit_column(name=column, description=f"Spike sorting unit column: {column}")
    for _, unit in units.iterrows():
        unit_id = int(unit["nwb_unit_id"])
        payload = {
            column: _nwb_scalar(unit[column], column_types.get(column, "text"))
            for column in units.columns
            if column not in {"nwb_unit_id", "spike_times"}
        }
        nwbfile.add_unit(id=unit_id, spike_times=spike_times.get(unit_id, np.array([], dtype=float)), **payload)


def _add_digital_events_to_nwb(nwbfile, digital_events: pd.DataFrame) -> None:
    try:
        from pynwb import TimeSeries
    except ImportError:
        return
    name_line_counts = digital_events.groupby("event_name", dropna=False)["line"].nunique().to_dict()
    for (event_name, line, edge), group in digital_events.groupby(["event_name", "line", "edge"], dropna=False):
        timestamps = group["time"].to_numpy(dtype=float)
        data = np.ones(len(group), dtype=np.int8)
        suffix = "ttl" if str(edge) == "rising" else f"{safe_event_name(edge)}_ttl"
        base_name = safe_event_name(event_name)
        if name_line_counts.get(event_name, 0) > 1:
            base_name = f"{base_name}_line_{int(line)}"
        nwbfile.add_acquisition(
            TimeSeries(
                name=f"{base_name}_{suffix}",
                data=data,
                unit="event",
                timestamps=timestamps,
                description=f"NI-DAQ TTL {edge} edges for {event_name}. Digital line {int(line)}.",
            )
        )


def _add_probe_electrode_groups_to_nwb(nwbfile, device, manifest: dict[str, Any]) -> None:
    probes = [row.get("probe_label", "") for row in manifest.get("source_rows", []) if row.get("probe_label")]
    for probe in probes:
        safe_probe = re.sub(r"[^A-Za-z0-9_]+", "_", probe)
        nwbfile.create_electrode_group(
            name=safe_probe,
            description=f"Neuropixels probe stream {probe}. Electrode-contact table is not yet populated in this v1 export.",
            location="brain",
            device=device,
        )


def _add_analog_to_nwb(config: dict[str, Any], nwbfile, recording_folder: Path) -> None:
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import TimeSeries

    analog = config.get("mouse_arena_nwb", {}).get("analog", {})
    stream = str(analog.get("stream_folder", "NI-DAQmx-109.PXI-6133"))
    stream_folder = recording_folder / "continuous" / stream
    dat = stream_folder / "continuous.dat"
    n_channels = len(analog.get("channel_names", {}) or {}) or 8
    samples = dat.stat().st_size // np.dtype("int16").itemsize // n_channels
    data = np.memmap(dat, dtype="int16", mode="r", shape=(samples, n_channels))
    nwbfile.add_acquisition(
        TimeSeries(
            name="nidaq_analog",
            data=H5DataIO(data),
            unit="V",
            starting_time=0.0,
            rate=30000.0,
            conversion=0.0003051851,
            description=f"NI-DAQ analog channels: {json.dumps(analog.get('channel_names', {}), default=str)}",
        )
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _session_metadata(config: dict[str, Any], *, protocol: str = "mouse_arena") -> dict[str, Any]:
    defaults = {
        "session_description": f"{protocol} Neuropixels session exported from the SpikeInterface pipeline.",
        "experiment_description": f"{protocol} experiment with Neuropixels recordings.",
        "experimenter": "Hickman, Jordan",
        "lab": "Denman Lab",
        "institution": "University of Colorado Anschutz",
        "device_name": "DenmanLab_Neuropixels",
        "keywords": [protocol, "neuropixels"],
    }
    generic = config.get("nwb", {}).get("session", {}) or {}
    defaults.update({key: value for key, value in generic.items() if value not in (None, "")})
    if protocol == "mouse_arena":
        configured = config.get("mouse_arena_nwb", {}).get("session", {}) or {}
        defaults.update({key: value for key, value in configured.items() if value not in (None, "")})
    if isinstance(defaults.get("keywords"), str):
        defaults["keywords"] = [item.strip() for item in str(defaults["keywords"]).split(",") if item.strip()]
    return defaults


def _subject_metadata(config: dict[str, Any], row: dict[str, str], session_start_time: datetime) -> dict[str, Any]:
    mouse_id = row.get("mouse_id") or row.get("animal_id") or row.get("subject_id") or ""
    nwb_config = config.get("mouse_arena_nwb", {})
    subjects = nwb_config.get("subjects", {}) or {}
    subject = dict(subjects.get(mouse_id, {}) or nwb_config.get("subject", {}) or {})
    subject_id = str(subject.get("subject_id") or mouse_id)
    subject["subject_id"] = subject_id
    subject.setdefault("species", "Mus musculus")
    subject.setdefault("sex", "U")
    subject.setdefault("strain", "C57/B6")
    subject.setdefault("description", "Mouse arena Neuropixels recording.")
    if subject.get("date_of_birth"):
        age = _age_from_dob(str(subject["date_of_birth"]), session_start_time)
        if age:
            subject["age"] = age
    return {key: value for key, value in subject.items() if value not in (None, "")}


def _age_from_dob(date_of_birth: str, session_start_time: datetime) -> str:
    parsed = pd.to_datetime(date_of_birth, errors="coerce")
    if pd.isna(parsed):
        return ""
    dob = parsed.to_pydatetime()
    if dob.tzinfo is None and session_start_time.tzinfo is not None:
        dob = dob.replace(tzinfo=session_start_time.tzinfo)
    days = max(0, int((session_start_time - dob).total_seconds() // 86400))
    return f"P{days}D"


def _subject_kwargs_for_nwb(subject: dict[str, Any]) -> dict[str, Any]:
    allowed = {"subject_id", "age", "description", "genotype", "sex", "species", "strain", "date_of_birth", "weight"}
    kwargs = {key: value for key, value in subject.items() if key in allowed and value not in (None, "")}
    if "date_of_birth" in kwargs:
        parsed = pd.to_datetime(kwargs["date_of_birth"], errors="coerce")
        if pd.isna(parsed):
            kwargs.pop("date_of_birth", None)
        else:
            kwargs["date_of_birth"] = parsed.to_pydatetime()
    return kwargs


def _metadata_list_or_string(value: Any) -> list[str] | str:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    return str(value or "")


def _nwb_column_types(frame: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            types[column] = "bool"
        elif pd.api.types.is_numeric_dtype(series):
            types[column] = "numeric"
        else:
            numeric = pd.to_numeric(series.dropna(), errors="coerce")
            types[column] = "numeric" if len(numeric) and numeric.notna().all() else "text"
    return types


def _nwb_scalar(value: Any, value_type: str = "text") -> Any:
    if value_type == "numeric":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return math.nan
        return number if math.isfinite(number) else math.nan
    if value_type == "bool":
        return bool(value) if not pd.isna(value) else False
    if pd.isna(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a mouse-arena session to derived NWB tables and NWB.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True, help="Any probe session_id from the target recording.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing NWB file.")
    parser.add_argument("--tables-only", action="store_true", help="Write derived CSV tables/manifest but skip the NWB file.")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = write_mouse_arena_nwb(config, session_id=args.session_id, overwrite=args.overwrite, tables_only=args.tables_only)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
