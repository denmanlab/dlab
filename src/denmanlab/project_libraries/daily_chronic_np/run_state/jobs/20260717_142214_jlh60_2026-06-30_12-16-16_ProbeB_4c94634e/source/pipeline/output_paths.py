from __future__ import annotations

from pathlib import Path
from typing import Any


RECORDING_LOCAL_LAYOUT = "recording_local_probe_folder"


def processed_folder_for_stream(
    *,
    config: dict[str, Any],
    session_folder: Path,
    record_node: Path,
    stream: dict[str, Any],
    session_info: dict[str, str],
    probe_label: str,
) -> Path:
    if config.get("project", {}).get("output_layout") == RECORDING_LOCAL_LAYOUT:
        output_name = config.get("project", {}).get("recording_local_output_name", "spikeinterface_output")
        return continuous_stream_folder(record_node=record_node, stream=stream) / output_name

    processed_root = Path(config["project"]["processed_root"])
    return processed_root / session_info["animal_id"] / session_folder.name / probe_label


def processed_folder_for_registry_row(row: dict[str, Any], config: dict[str, Any]) -> Path:
    if config.get("project", {}).get("output_layout") == RECORDING_LOCAL_LAYOUT:
        output_name = config.get("project", {}).get("recording_local_output_name", "spikeinterface_output")
        return continuous_stream_folder_for_row(row) / output_name
    return Path(row["processed_folder"])


def continuous_stream_folder(*, record_node: Path, stream: dict[str, Any]) -> Path:
    folder_name = str(stream["folder_name"]).strip("/\\")
    oebins = sorted(record_node.glob("experiment*/recording*/structure.oebin"))
    if len(oebins) != 1:
        raise ValueError(f"Expected exactly one structure.oebin under {record_node}, found {len(oebins)}")
    return oebins[0].parent / "continuous" / folder_name


def continuous_stream_folder_for_row(row: dict[str, Any]) -> Path:
    record_node = find_record_node(Path(row["raw_folder"]))
    folder_name = _folder_name_from_stream_name(str(row.get("stream_name", "")), str(row.get("probe_label", "")))
    oebins = sorted(record_node.glob("experiment*/recording*/structure.oebin"))
    if len(oebins) != 1:
        raise ValueError(f"Expected exactly one structure.oebin under {record_node}, found {len(oebins)}")
    return oebins[0].parent / "continuous" / folder_name


def output_paths_for_processed_folder(processed_folder: Path) -> dict[str, Path]:
    return {
        "processed_folder": processed_folder,
        "preprocessed_folder": processed_folder / "preprocessing",
        "sorter_output_folder": processed_folder / "kilosort4",
        "sorting_analyzer_folder": processed_folder / "sorting_analyzer",
        "quality_metrics_csv": processed_folder / "quality_metrics.csv",
    }


def smoke_processed_folder(base_processed_folder: Path, duration_seconds: float) -> Path:
    parent = base_processed_folder.parent
    suffix = f"_smoke_{int(duration_seconds)}s"
    return parent / f"{base_processed_folder.name}{suffix}"


def suffixed_processed_folder(base_processed_folder: Path, suffix: str) -> Path:
    suffix = suffix.strip().strip("_")
    if not suffix:
        return base_processed_folder
    return base_processed_folder.parent / f"{base_processed_folder.name}_{suffix}"


def find_record_node(session_folder: Path) -> Path:
    if (session_folder / "settings.xml").exists():
        return session_folder
    nodes = sorted(session_folder.glob("Record Node *"))
    if len(nodes) != 1:
        raise ValueError(f"Expected one Record Node folder under {session_folder}, found {len(nodes)}")
    return nodes[0]


def _folder_name_from_stream_name(stream_name: str, probe_label: str) -> str:
    if "#" in stream_name:
        return stream_name.split("#", 1)[1]
    if stream_name:
        return stream_name
    raise ValueError(f"Cannot derive continuous stream folder for probe {probe_label!r}")
