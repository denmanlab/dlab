from __future__ import annotations

import argparse
import json
from pathlib import Path

import spikeinterface.full as si


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Open Ephys recording with SpikeInterface.")
    parser.add_argument("path", help="Session folder or Record Node folder")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    record_node = path
    if not (record_node / "settings.xml").exists():
        nodes = sorted(path.glob("Record Node *"))
        if len(nodes) != 1:
            raise SystemExit(f"Expected exactly one Record Node folder under {path}, found {len(nodes)}")
        record_node = nodes[0]

    oebins = sorted(record_node.glob("experiment*/recording*/structure.oebin"))
    if len(oebins) != 1:
        raise SystemExit(f"Expected exactly one structure.oebin under {record_node}, found {len(oebins)}")

    metadata = json.loads(oebins[0].read_text())
    completion_paths = [
        path / "recording_complete.txt",
        record_node / "recording_complete.txt",
        oebins[0].parent / "recording_complete.txt",
    ]

    print(f"session: {path}")
    print(f"record_node: {record_node}")
    print(f"structure: {oebins[0]}")
    print(f"completion_file_found: {any(p.exists() for p in completion_paths)}")
    print("continuous_streams:")
    for stream in metadata.get("continuous", []):
        print(
            "  "
            f"{stream.get('stream_name')} | "
            f"{stream.get('folder_name')} | "
            f"{stream.get('num_channels')} ch | "
            f"{stream.get('sample_rate')} Hz"
        )

    neo_stream_names, neo_stream_ids = si.get_neo_streams("openephysbinary", record_node)
    print("neo_streams:")
    for name, stream_id in zip(neo_stream_names, neo_stream_ids):
        print(f"  {name} | id={stream_id}")

    print("spikeinterface_loads:")
    for stream in metadata.get("continuous", []):
        if stream.get("num_channels") != 384:
            continue
        folder_name = stream.get("folder_name", "").strip("/")
        stream_name = next((name for name in neo_stream_names if folder_name in name), stream.get("stream_name"))
        rec = si.read_openephys(record_node, stream_name=stream_name, load_sync_timestamps=False)
        locations = rec.get_channel_locations()
        first_location = None if locations is None else locations[0].tolist()
        print(
            "  "
            f"{stream_name} | "
            f"{rec.get_num_channels()} ch | "
            f"{rec.get_sampling_frequency()} Hz | "
            f"{rec.get_num_frames()} frames | "
            f"{rec.get_total_duration():.3f} s | "
            f"locations={None if locations is None else locations.shape} | "
            f"first_location={first_location}"
        )


if __name__ == "__main__":
    main()
