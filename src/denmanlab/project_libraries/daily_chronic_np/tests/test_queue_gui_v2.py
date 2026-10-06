from __future__ import annotations

from pathlib import Path
import json

from fastapi.testclient import TestClient

from pipeline.profiles import list_profiles, resolve_profile, save_profile
from pipeline.queue_store import QueueStore
from pipeline.queue_worker import _next_probe_action, _restore_recoverable_interrupted_output
from pipeline.recording_identity import recording_identity, recording_identity_for_row
from pipeline.registry import upsert_rows
from pipeline_gui.app import create_app
from pipeline_gui.services import queue_events_payload, queue_job_log_payload


def test_recording_identity_separates_open_ephys_blocks():
    first_id, first_label = recording_identity(
        session_folder="jlh60_2026-07-08_12-14-57",
        record_node="Record Node 101",
        experiment="experiment1",
        recording="recording1",
    )
    second_id, second_label = recording_identity(
        session_folder="jlh60_2026-07-08_12-14-57",
        record_node="Record Node 101",
        experiment="experiment2",
        recording="recording1",
    )
    assert first_id != second_id
    assert first_label == "experiment1 / recording1"
    assert second_label == "experiment2 / recording1"

    inferred_id, inferred_label = recording_identity_for_row(
        {
            "raw_folder": "C:/raw/jlh60_2026-07-08_12-14-57",
            "processed_folder": (
                "C:/raw/jlh60_2026-07-08_12-14-57/Record Node 101/"
                "experiment2/recording1/continuous/ProbeA/spikeinterface_output"
            ),
        }
    )
    assert inferred_id == second_id
    assert inferred_label == second_label


def test_processing_profiles_are_versioned_and_resolved(tmp_path: Path):
    config_path = tmp_path / "pipeline" / "config_analysis.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  root: pipeline/profiles",
                "  default: default",
                "project:",
                f"  raw_root: {tmp_path.as_posix()}",
                f"  processed_root: {(tmp_path / 'processed').as_posix()}",
                f"  registry_csv: {(tmp_path / 'registry.csv').as_posix()}",
                "jobs:",
                "  n_jobs: 8",
                "preprocessing:",
                "  car:",
                "    operator: median",
            ]
        ),
        encoding="utf-8",
    )
    profiles = list_profiles(config_path)
    assert profiles[0]["name"] == "default"
    payload = {
        "jobs": {"n_jobs": 4},
        "preprocessing": {"car": {"operator": "median"}},
    }
    first = save_profile(config_path, "four_jobs", payload, description="test")
    second = save_profile(config_path, "four_jobs", payload, description="test")
    assert first["version"] == 1
    assert second["version"] == 2
    effective, metadata = resolve_profile(config_path, "four_jobs")
    assert effective["jobs"]["n_jobs"] == 4
    assert metadata["version"] == 2
    assert metadata["hash"]


def test_queue_store_persists_items_jobs_and_pause(tmp_path: Path):
    path = tmp_path / "queue.sqlite3"
    store = QueueStore(path)
    item = store.enqueue(
        recording_id="rec_1",
        session_ids=["session_ProbeA", "session_ProbeB"],
        profile_name="default",
        raw_folder="C:/raw/session",
    )
    assert item["status"] == "queued"
    assert [probe["probe_label"] for probe in item["probes"]] == ["ProbeA", "ProbeB"]
    store.update_probe(item["queue_id"], "session_ProbeA", status="complete")
    store.request_pause()

    reopened = QueueStore(path)
    persisted = reopened.get_item(item["queue_id"])
    assert persisted["probes"][0]["status"] == "complete"
    assert reopened.pause_requested() is True
    reopened.clear_pause()
    assert reopened.pause_requested() is False


def test_queue_events_payload_includes_active_worker_item(tmp_path: Path):
    store = QueueStore(tmp_path / "run_state" / "pipeline_queue.sqlite3")
    item = store.enqueue(
        recording_id="rec_1",
        session_ids=["session_ProbeA"],
        profile_name="default",
        raw_folder="C:/raw/session",
    )
    store.update_item(item["queue_id"], status="running", started_at="2026-07-18T12:00:00+00:00")
    store.set_meta("worker_status", "running")
    store.set_meta("active_queue_id", item["queue_id"])
    store.set_meta("active_session_id", "session_ProbeA")
    store.add_event(
        queue_id=item["queue_id"],
        session_id="session_ProbeA",
        event="staging_progress",
        message="Staging 50.0%.",
        payload={"percent": 50.0},
    )

    payload = queue_events_payload(tmp_path)
    assert payload["worker"]["active_queue_id"] == item["queue_id"]
    assert payload["active_item"]["recording_id"] == "rec_1"
    assert payload["queue"][0]["queue_id"] == item["queue_id"]
    assert payload["queue"][0]["probes"][0]["current_step"] == ""
    assert payload["events"][-1]["payload"]["percent"] == 50.0


def test_stale_worker_reconciliation_keeps_item_first_and_retryable(tmp_path: Path):
    store = QueueStore(tmp_path / "queue.sqlite3")
    item = store.enqueue(
        recording_id="rec_1",
        session_ids=["session_ProbeA"],
        profile_name="default",
        raw_folder="C:/raw/session",
    )
    store.update_item(item["queue_id"], status="running", current_probe="session_ProbeA")
    store.update_probe(item["queue_id"], "session_ProbeA", status="running")
    result = store.reconcile_stale_worker(error="worker stopped")
    reconciled = store.get_item(item["queue_id"])
    assert result["items"] == 1
    assert reconciled["status"] == "paused"
    assert reconciled["position"] == item["position"]
    assert reconciled["probes"][0]["status"] == "queued"
    store.clear_pause()
    assert store.next_item()["queue_id"] == item["queue_id"]


def test_interrupted_completed_sorter_is_restored_for_qc(tmp_path: Path):
    processed = tmp_path / "spikeinterface_output"
    partial_sorter = processed / "kilosort4"
    partial_sorter.mkdir(parents=True)
    (partial_sorter / "spikeinterface_log.json").write_text("{}", encoding="utf-8")
    interrupted = tmp_path / "spikeinterface_output_interrupted_20260717_140013"
    completed_sorter = interrupted / "kilosort4" / "sorter_output"
    completed_sorter.mkdir(parents=True)
    (completed_sorter / "spike_times.npy").write_bytes(b"complete")
    row = {
        "processed_folder": str(processed),
        "sorter_output_folder": str(processed / "kilosort4"),
        "sort_status": "running",
    }
    restored = _restore_recoverable_interrupted_output(row)
    assert restored == interrupted
    assert (processed / "kilosort4" / "sorter_output" / "spike_times.npy").exists()
    assert _next_probe_action(row) == "restart-qc"
    assert any(tmp_path.glob("spikeinterface_output_failed_*"))


def test_queue_job_log_payload_exposes_progress_and_phase_events(tmp_path: Path):
    store = QueueStore(tmp_path / "run_state" / "pipeline_queue.sqlite3")
    item = store.enqueue(
        recording_id="rec_1",
        session_ids=["session_ProbeA"],
        profile_name="default",
        raw_folder="C:/raw/session",
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    events = tmp_path / "events.jsonl"
    stdout.write_text("[1.0s] session: START compute waveforms\n", encoding="utf-8")
    stderr.write_text(
        "compute_waveforms:  50%|#####     | 5/10 [00:05<00:05, 1.00s/it]\n",
        encoding="utf-8",
    )
    events.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-17T20:00:00+00:00",
                "event": "phase_started",
                "name": "compute waveforms",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store.add_job(
        job_id="job_1",
        queue_id=item["queue_id"],
        session_id="session_ProbeA",
        kind="restart-qc",
        command=["python", "-m", "pipeline.restart_qc"],
        stdout_log=str(stdout),
        stderr_log=str(stderr),
        event_log=str(events),
        pid=123,
        profile_name="default",
        profile_version=1,
        config_hash="config",
        source_hash="source",
    )
    payload = queue_job_log_payload(tmp_path, job_id="job_1", stream="events", tail=20)
    assert payload["job"]["kind"] == "restart-qc"
    assert payload["progress"]["percent"] == 50
    assert payload["latest_phase"]["name"] == "compute waveforms"
    assert payload["phase_events"][0]["event"] == "phase_started"


def test_recordings_api_groups_blocks_and_labels_short_survey(tmp_path: Path):
    registry = tmp_path / "registry.csv"
    processed = (
        tmp_path
        / "raw"
        / "jlh60_2026-07-08_12-14-57"
        / "Record Node 101"
        / "experiment2"
        / "recording1"
        / "continuous"
        / "ProbeA"
        / "spikeinterface_output"
    )
    recording_id, block = recording_identity(
        session_folder="jlh60_2026-07-08_12-14-57",
        record_node="Record Node 101",
        experiment="experiment2",
        recording="recording1",
    )
    upsert_rows(
        registry,
        [
            {
                "session_id": "jlh60_2026-07-08_12-14-57_experiment2_recording1_ProbeA",
                "recording_id": recording_id,
                "recording_block": block,
                "mouse_id": "jlh60",
                "recording_date": "2026-07-08",
                "task": "12-14-57",
                "raw_folder": str(tmp_path / "raw" / "jlh60_2026-07-08_12-14-57"),
                "server_raw_folder": str(tmp_path / "server" / "jlh60_2026-07-08_12-14-57"),
                "open_ephys_experiment_name": "experiment2",
                "open_ephys_block_index": "recording1",
                "open_ephys_record_node": "Record Node 101",
                "probe_label": "ProbeA",
                "duration_seconds": "30.25",
                "status": "skipped",
                "processed_folder": str(processed),
            }
        ],
    )
    config = tmp_path / "pipeline" / "config_analysis.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "\n".join(
            [
                "machine:",
                "  role: analysis",
                "profiles:",
                "  root: pipeline/profiles",
                "  default: default",
                "project:",
                f"  raw_root: {(tmp_path / 'raw').as_posix()}",
                f"  processed_root: {(tmp_path / 'processed').as_posix()}",
                f"  registry_csv: {registry.as_posix()}",
                "recordings:",
                "  minimum_duration_seconds: 300",
                "jobs:",
                "  n_jobs: 8",
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(config_path=config, repo_root=tmp_path))
    response = client.get("/api/recordings")
    assert response.status_code == 200
    recording = response.json()["recordings"][0]
    assert recording["recording_id"] == recording_id
    assert recording["recording_block"] == "experiment2 / recording1"
    assert recording["duration_seconds"] == 30.25
    assert recording["survey_like"] is True
    assert "survey block" in recording["eligibility_reason"]
    assert client.get("/api/profiles").status_code == 200


def test_long_later_experiment_is_still_excluded_as_survey(tmp_path: Path):
    registry = tmp_path / "registry.csv"
    recording_id, block = recording_identity(
        session_folder="jlh60_2026-07-10_12-00-00",
        record_node="Record Node 101",
        experiment="experiment2",
        recording="recording1",
    )
    upsert_rows(
        registry,
        [
            {
                "session_id": "jlh60_2026-07-10_12-00-00_experiment2_recording1_ProbeA",
                "recording_id": recording_id,
                "recording_block": block,
                "mouse_id": "jlh60",
                "recording_date": "2026-07-10",
                "raw_folder": str(tmp_path / "raw" / "jlh60_2026-07-10_12-00-00"),
                "open_ephys_experiment_name": "experiment2",
                "open_ephys_block_index": "recording1",
                "open_ephys_record_node": "Record Node 101",
                "probe_label": "ProbeA",
                "duration_seconds": "1600",
                "status": "registered",
            }
        ],
    )
    config = tmp_path / "pipeline" / "config_analysis.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "\n".join(
            [
                "profiles:",
                "  root: pipeline/profiles",
                "  default: default",
                "project:",
                f"  raw_root: {(tmp_path / 'raw').as_posix()}",
                f"  processed_root: {(tmp_path / 'processed').as_posix()}",
                f"  registry_csv: {registry.as_posix()}",
                "recordings:",
                "  minimum_duration_seconds: 300",
                "jobs:",
                "  n_jobs: 8",
            ]
        ),
        encoding="utf-8",
    )
    recording = TestClient(create_app(config_path=config, repo_root=tmp_path)).get("/api/recordings").json()["recordings"][0]
    assert recording["survey_like"] is True
    assert recording["short_recording"] is False
    assert recording["eligible"] is False
    assert "excluded from the default queue" in recording["eligibility_reason"]
