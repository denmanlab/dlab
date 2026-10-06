from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import probeinterface
import spikeinterface.full as si

from .hashing import hash_array, hash_file, hash_jsonable


def generated_probe_path(row: dict[str, Any]) -> Path:
    return Path(row["processed_folder"]) / "metadata" / f"{row['probe_label']}_probeinterface.json"


def ensure_probeinterface_json(recording, row: dict[str, Any], *, write: bool = True) -> Path:
    existing = row.get("probeinterface_json")
    if existing and Path(existing).exists():
        return Path(existing)
    probe = recording.get_probe()
    if probe is None:
        raise ValueError("Recording has no probe metadata; cannot generate ProbeInterface JSON")
    output = generated_probe_path(row)
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        probeinterface.write_probeinterface(output, probe)
    return output


def load_recording_with_probe(row: dict[str, Any], config: dict[str, Any]):
    record_node = find_record_node(Path(row["raw_folder"]))
    recording = si.read_openephys(
        record_node,
        stream_name=row["stream_name"],
        load_sync_timestamps=bool(config.get("recordings", {}).get("load_sync_timestamps", True)),
    )
    probe_json = ensure_probeinterface_json(recording, row, write=True)
    probe_group = probeinterface.read_probeinterface(probe_json)
    if len(probe_group.probes) != 1:
        raise ValueError(f"Expected one probe in {probe_json}, found {len(probe_group.probes)}")
    recording = recording.set_probe(probe_group.probes[0])
    validate_probe(recording)
    return recording, probe_json


def validate_probe(recording) -> None:
    probe = recording.get_probe()
    if probe is None:
        raise ValueError("Missing probe geometry")
    if probe.get_contact_count() != recording.get_num_channels():
        raise ValueError(
            f"Probe contact count {probe.get_contact_count()} does not match recording channels {recording.get_num_channels()}"
        )
    locations = recording.get_channel_locations()
    if locations is None or len(locations) != recording.get_num_channels():
        raise ValueError("Missing or incomplete channel locations")


def compute_probe_hashes(recording, probe_json: Path) -> dict[str, str]:
    channel_ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
    locations = np.asarray(recording.get_channel_locations(), dtype="float64")
    probe = recording.get_probe()
    contact_positions = np.asarray(probe.contact_positions, dtype="float64")
    shank_ids = [] if probe.shank_ids is None else [str(value) for value in probe.shank_ids]
    return {
        "probeinterface_json": str(probe_json),
        "probeinterface_hash": hash_file(probe_json),
        "channel_ids_hash": hash_jsonable(channel_ids),
        "geometry_hash": hash_jsonable({"channel_ids": channel_ids, "locations": locations.round(6).tolist()}),
        "site_hash": hash_jsonable({"contact_positions": contact_positions.round(6).tolist(), "shank_ids": shank_ids}),
    }


def find_record_node(session_folder: Path) -> Path:
    if (session_folder / "settings.xml").exists():
        return session_folder
    nodes = sorted(session_folder.glob("Record Node *"))
    if len(nodes) != 1:
        raise ValueError(f"Expected one Record Node folder under {session_folder}, found {len(nodes)}")
    return nodes[0]


def channel_ids_to_indices(recording, channel_ids) -> list[int]:
    all_ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
    lookup = {channel_id: index for index, channel_id in enumerate(all_ids)}
    return [lookup[str(channel_id)] for channel_id in channel_ids if str(channel_id) in lookup]
