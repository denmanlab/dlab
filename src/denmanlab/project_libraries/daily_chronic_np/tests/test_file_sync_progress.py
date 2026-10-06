from __future__ import annotations

from pathlib import Path

from pipeline.file_sync import SyncProgress, copy_tree_update_only
from pipeline.stage_recording import stage_recording_for_row


def test_copy_tree_reports_chunk_progress_and_reuses_current_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "small.bin").write_bytes(b"a" * 10)
    (source / "large.bin").write_bytes(b"b" * 25)

    events: list[SyncProgress] = []
    result = copy_tree_update_only(
        source,
        destination,
        progress_callback=events.append,
        copy_chunk_bytes=8,
    )

    assert result.copied_files == 2
    assert result.copied_bytes == 35
    assert (destination / "small.bin").read_bytes() == b"a" * 10
    assert (destination / "large.bin").read_bytes() == b"b" * 25
    assert events[-1].completed_bytes == 35
    assert events[-1].total_bytes == 35
    assert events[-1].percent == 100.0
    assert {event.status for event in events} >= {"copying", "complete"}

    reused_events: list[SyncProgress] = []
    reused = copy_tree_update_only(source, destination, progress_callback=reused_events.append)
    assert reused.copied_files == 0
    assert reused.skipped_files == 2
    assert reused_events[-1].percent == 100.0
    assert all(event.status == "skipped" for event in reused_events)


def test_staging_ignores_foreign_registry_local_path_and_localizes_probe_outputs(tmp_path: Path):
    server_root = tmp_path / "server"
    local_root = tmp_path / "local"
    source = server_root / "jlh62_2026-07-20_12-30-31"
    stream = (
        source
        / "Record Node 101"
        / "experiment1"
        / "recording1"
        / "continuous"
        / "Neuropix-PXI-100.ProbeA"
    )
    stream.mkdir(parents=True)
    (stream / "continuous.dat").write_bytes(b"recording")
    row = {
        "session_id": "jlh62_2026-07-20_12-30-31_ProbeA",
        "raw_folder": r"C:\Users\denma\Documents\Open Ephys\jlh62_2026-07-20_12-30-31",
        "server_raw_folder": str(source),
        "local_raw_folder": r"C:\Users\denma\Documents\Open Ephys\jlh62_2026-07-20_12-30-31",
        "open_ephys_record_node": "Record Node 101",
        "open_ephys_experiment_name": "experiment1",
        "open_ephys_block_index": "recording1",
        "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
        "probe_label": "ProbeA",
    }
    config = {
        "project": {
            "raw_root": str(local_root),
            "processed_root": str(tmp_path / "processed"),
            "output_layout": "recording_local_probe_folder",
            "recording_local_output_name": "spikeinterface_output",
            "sorting_analyzer_folder_name": "analyzer",
        },
        "staging": {
            "enabled": True,
            "server_raw_root": str(server_root),
            "local_raw_root": str(local_root),
        },
    }

    result, fields = stage_recording_for_row(row, config)

    expected_local = local_root / source.name
    expected_output = (
        expected_local
        / "Record Node 101"
        / "experiment1"
        / "recording1"
        / "continuous"
        / "Neuropix-PXI-100.ProbeA"
        / "spikeinterface_output"
    )
    assert result.destination == expected_local
    assert fields["raw_folder"] == str(expected_local)
    assert fields["processed_folder"] == str(expected_output)
    assert fields["sorter_output_folder"] == str(expected_output / "kilosort4")
    assert fields["sorting_analyzer_folder"] == str(expected_output / "analyzer")
    assert fields["probeinterface_json"] == str(expected_output / "metadata" / "ProbeA_probeinterface.json")
