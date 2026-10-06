from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_name",
    "line",
    "edge",
    "ordinal",
    "timestamp",
    "time",
    "sample_number",
    "state",
]


def nwb_protocol_for_row(config: dict[str, Any], row: dict[str, str]) -> str:
    nwb = config.get("nwb", {})
    requested = str(nwb.get("default_protocol", "auto")).strip().lower()
    if requested and requested != "auto":
        return requested
    text = " ".join(
        str(row.get(key) or "")
        for key in ("task", "raw_folder", "server_raw_folder", "session_id")
    )
    for rule in nwb.get("protocol_rules", []):
        pattern = str(rule.get("match", "")).strip()
        if pattern and re.search(pattern, text, flags=re.IGNORECASE):
            return str(rule.get("protocol") or "generic").strip()
    return "mouse_arena"


def load_digital_events(
    config: dict[str, Any],
    *,
    recording_folder: Path,
    override_recording_folder: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    digital = digital_event_config(config)
    stream = str(digital.get("stream_folder", "NI-DAQmx-109.PXI-6133"))
    ttl = recording_folder / "events" / stream / "TTL"
    states = np.asarray(np.load(ttl / "states.npy")).reshape(-1)
    timestamps = np.asarray(np.load(ttl / "timestamps.npy")).reshape(-1)
    sample_numbers = np.asarray(np.load(ttl / "sample_numbers.npy")).reshape(-1)
    if not (len(states) == len(timestamps) == len(sample_numbers)):
        raise ValueError(
            "Digital event arrays have different lengths: "
            f"states={len(states)}, timestamps={len(timestamps)}, sample_numbers={len(sample_numbers)}"
        )
    time_zero = digital_time_zero(config, recording_folder=recording_folder, stream=stream)
    line_names = resolved_line_names(
        config,
        recording_folder=override_recording_folder or recording_folder,
    )
    include_edges = {
        str(edge).strip().lower()
        for edge in digital.get("include_edges", ["rising", "falling"])
    }
    counters: dict[tuple[int, str], int] = {}
    rows: list[dict[str, Any]] = []
    for index, raw_state in enumerate(states):
        state = int(raw_state)
        if state == 0:
            continue
        line = abs(state)
        edge = "rising" if state > 0 else "falling"
        if edge not in include_edges:
            continue
        key = (line, edge)
        counters[key] = counters.get(key, 0) + 1
        rows.append(
            {
                "event_name": line_names.get(line, f"line_{line}"),
                "line": line,
                "edge": edge,
                "ordinal": counters[key],
                "timestamp": float(timestamps[index]),
                "time": float(timestamps[index] - time_zero),
                "sample_number": int(sample_numbers[index]),
                "state": state,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), time_zero


def digital_event_inventory(events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = events.groupby(["line", "event_name", "edge"], dropna=False, sort=True)
    for (line, name, edge), group in grouped:
        times = pd.to_numeric(group["time"], errors="coerce").dropna().to_numpy(dtype=float)
        intervals = np.diff(times)
        rows.append(
            {
                "line": int(line),
                "event_name": str(name),
                "edge": str(edge),
                "count": int(len(group)),
                "first_time": float(times[0]) if times.size else None,
                "last_time": float(times[-1]) if times.size else None,
                "median_interval": float(np.median(intervals)) if intervals.size else None,
            }
        )
    return rows


def save_line_name(
    config: dict[str, Any],
    *,
    recording_folder: Path,
    line: int,
    name: str,
) -> Path:
    output_folder = recording_folder / nwb_output_folder_name(config)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / "digital_line_names.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    names = dict(payload.get("lines", {}))
    clean_name = safe_event_name(name)
    if not clean_name:
        names.pop(str(int(line)), None)
    else:
        names[str(int(line))] = clean_name
    payload.update({"lines": names, "description": "Recording-specific digital line names used for NWB export."})
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def resolved_line_names(config: dict[str, Any], *, recording_folder: Path) -> dict[int, str]:
    digital = digital_event_config(config)
    configured = {
        int(line): safe_event_name(name)
        for line, name in dict(digital.get("line_names", {})).items()
        if str(line).lstrip("-").isdigit() and safe_event_name(name)
    }
    legacy = config.get("mouse_arena_nwb", {}).get("digital_events", {})
    configured.setdefault(int(legacy.get("trial_starts_line", 5)), "trial_start")
    configured.setdefault(int(legacy.get("rewards_line", 4)), "reward")
    configured.setdefault(int(legacy.get("frames_line", 6)), "game_frame")
    override_path = recording_folder / nwb_output_folder_name(config) / "digital_line_names.json"
    if override_path.exists():
        try:
            overrides = json.loads(override_path.read_text(encoding="utf-8")).get("lines", {})
        except (json.JSONDecodeError, OSError):
            overrides = {}
        for line, name in overrides.items():
            if str(line).lstrip("-").isdigit() and safe_event_name(name):
                configured[int(line)] = safe_event_name(name)
    return configured


def digital_event_config(config: dict[str, Any]) -> dict[str, Any]:
    generic = config.get("nwb", {}).get("digital_events", {})
    legacy = config.get("mouse_arena_nwb", {}).get("digital_events", {})
    return {**legacy, **generic}


def digital_time_zero(config: dict[str, Any], *, recording_folder: Path, stream: str) -> float:
    mode = str(digital_event_config(config).get("time_zero", "nidaq_continuous_start"))
    if mode != "nidaq_continuous_start":
        return 0.0
    timestamps = np.load(
        recording_folder / "continuous" / stream / "timestamps.npy",
        mmap_mode="r",
    )
    return float(timestamps[0])


def nwb_output_folder_name(config: dict[str, Any]) -> str:
    return str(
        config.get("nwb", {}).get("output_folder_name")
        or config.get("mouse_arena_nwb", {}).get("output_folder_name", "nwb")
    )


def safe_event_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
