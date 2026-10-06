from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pipeline.run_one as run_one_module
from pipeline.backup import backup_folder_for_raw, verify_raw_backup
from pipeline.derived_backup import backup_derived_outputs_for_raw, backup_derived_outputs_for_recording_session, find_derived_output_folders
from pipeline.discover import _parse_session_folder
from pipeline.export_phy import export_phy_for_row
from pipeline.hashing import hash_array, hash_jsonable
from pipeline.mouse_arena_nwb import build_mouse_arena_nwb_tables
from pipeline.local_cleanup import assess_recording_cleanup, cleanup_recording_for_session
from pipeline.nwb_digital_events import (
    digital_event_inventory,
    load_digital_events,
    nwb_protocol_for_row,
    resolved_line_names,
    save_line_name,
)
from pipeline.output_paths import (
    output_paths_for_processed_folder,
    sorting_analyzer_folder_for_processed_folder,
    sorting_analyzer_folder_name,
    suffixed_processed_folder,
)
from pipeline.probe_maps import channel_ids_to_indices, validate_probe
from pipeline.registry import read_registry, upsert_rows
from pipeline.restart_qc import _archive_qc_outputs
from pipeline.reports import _bad_channel_rows, _shared_trace_scale, write_preprocessing_trace_plots, write_shareable_summary_png
from pipeline.run_pending import _is_completed_base_row
from pipeline.run_one import (
    _attach_automated_label_properties,
    _analyzer_extension_specs,
    _bad_channels_for_sorting,
    _compute_analyzer_extension,
    _delete_analyzer_extension,
    _failed_status_update,
    _failure_message,
    _fallback_n_jobs,
    _job_kwargs,
    _kilosort4_params,
    _looks_like_unexpected_job_kwarg,
    _looks_like_multiprocessing_spawn_failure,
    _patch_phy_dat_path,
    _phy_export_kwargs,
    _prepare_phy_extensions,
    _phy_metric_export_kwargs,
    _prune_phy_metric_tsvs,
    _quality_metric_compute_names,
    _sanitize_output_suffix,
    _sorting_analyzer_kwargs,
    _validate_analyzer_storage_path,
    _write_channel_qc,
    _write_phy_unitrefine_tsvs,
)
from pipeline.qc_variant import config_with_variant_overrides
from pipeline.timings import QCTimingRecorder, latest_timing_summary
from pipeline.unitrefine import resolve_rows_for_cli, run_unitrefine_for_analyzer, unitrefine_output_folder
from pipeline_gui.services import (
    JobRecord,
    JobManager,
    build_job_command,
    clear_nwb_input_override_payload,
    config_text_payload,
    config_payload,
    _blocking_running_rows,
    _parse_nvidia_smi,
    is_allowed_file,
    job_progress,
    probe_payload,
    reports_payload,
    save_nwb_input_csv_payload,
    save_config_text,
    status_payload,
)


class FakeRecording:
    def __init__(self, channel_ids=("CH0", "CH1", "CH2")):
        self._channel_ids = np.asarray(channel_ids)

    def get_channel_ids(self):
        return self._channel_ids


class FakeProbe:
    def __init__(self, count):
        self._count = count

    def get_contact_count(self):
        return self._count


class FakeProbeRecording(FakeRecording):
    def __init__(self, channel_ids=("CH0", "CH1", "CH2"), contacts=3, locations=None):
        super().__init__(channel_ids)
        self._probe = FakeProbe(contacts)
        self._locations = np.zeros((len(channel_ids), 2)) if locations is None else locations

    def get_probe(self):
        return self._probe

    def get_num_channels(self):
        return len(self._channel_ids)

    def get_channel_locations(self):
        return self._locations


class FakeTraceRecording(FakeRecording):
    def __init__(self, traces):
        super().__init__([f"CH{index}" for index in range(traces.shape[1])])
        self._traces = traces

    def get_sampling_frequency(self):
        return 1000.0

    def get_num_frames(self):
        return self._traces.shape[0]

    def get_traces(self, start_frame, end_frame, channel_ids=None, return_scaled=True):
        if channel_ids is None:
            indices = slice(None)
        else:
            lookup = {channel_id: index for index, channel_id in enumerate(self._channel_ids)}
            indices = [lookup[channel_id] for channel_id in channel_ids]
        return self._traces[start_frame:end_frame, indices]


class FakeComputeAnalyzer:
    def __init__(self, errors: list[Exception] | None = None):
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.errors = list(errors or [])

    def compute(self, extension, **kwargs):
        self.calls.append((extension, dict(kwargs)))
        if self.errors:
            raise self.errors.pop(0)


class FakeDeleteAnalyzer:
    def __init__(self, folder: Path):
        self.folder = folder
        self.extensions = {"spike_locations": None}

    def delete_extension(self, extension):
        raise AttributeError("'NoneType' object has no attribute 'delete'")


class FakeSpawnFailureAnalyzer:
    def __init__(self, folder: Path, failures: int = 1):
        self.folder = folder
        self.extensions = {"spike_amplitudes": None}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.failures = failures

    def delete_extension(self, extension):
        raise AttributeError("'NoneType' object has no attribute 'delete'")

    def compute(self, extension, **kwargs):
        self.calls.append((extension, dict(kwargs)))
        if len(self.calls) <= self.failures:
            raise OSError(22, "Invalid argument")


class FakePhyExtensionAnalyzer:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def has_extension(self, extension):
        return extension in self.existing

    def compute(self, extension, **kwargs):
        self.calls.append((extension, dict(kwargs)))
        self.existing.add(extension)


class FakeSorting:
    def __init__(self, unit_ids=(0, 1, 2)):
        self._unit_ids = list(unit_ids)
        self.properties = {}

    def get_unit_ids(self):
        return self._unit_ids

    def set_property(self, key, values):
        self.properties[key] = list(values)


class FakeExtension:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class FakeAnalyzer:
    def __init__(self):
        self.recording = FakeProbeRecording(
            channel_ids=("CH0", "CH1", "CH2", "CH3"),
            contacts=4,
            locations=np.asarray([[0, 0], [20, 20], [0, 40], [20, 60]], dtype=float),
        )
        self.sorting = FakeSorting()
        self._extensions = {
            "unit_locations": FakeExtension(np.asarray([[0, 10], [20, 30], [0, 50]], dtype=float)),
            "templates": FakeExtension(np.random.default_rng(1).normal(size=(3, 60, 4))),
        }

    def get_extension(self, name):
        return self._extensions[name]


def test_hashes_are_deterministic():
    assert hash_jsonable({"b": 2, "a": 1}) == hash_jsonable({"a": 1, "b": 2})
    assert hash_array([[1, 2], [3, 4]]) == hash_array(np.asarray([[1, 2], [3, 4]]))


def test_registry_upsert_is_readable(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    upsert_rows(
        registry,
        [
            {
                "session_id": "s1",
                "stream_id": "1",
                "probe_label": "ProbeA",
                "status": "registered",
            }
        ],
    )
    rows = read_registry(registry)
    assert len(rows) == 1
    assert rows[0]["status"] == "registered"

    upsert_rows(registry, [{"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA", "status": "complete"}])
    rows = read_registry(registry)
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"


def test_session_folder_parses_mouse_id_and_registry_backfills_it(tmp_path: Path):
    parsed = _parse_session_folder("jlh602026-06-29_14-31-15")
    assert parsed["mouse_id"] == "jlh60"
    assert parsed["animal_id"] == "jlh60"
    assert parsed["recording_date"] == "2026-06-29"
    assert parsed["task"] == "14-31-15"

    registry = tmp_path / "sessions.csv"
    upsert_rows(registry, [{"session_id": "s1", "animal_id": "jlh60", "stream_id": "1", "probe_label": "ProbeA"}])
    row = read_registry(registry)[0]
    assert row["mouse_id"] == "jlh60"


def test_registry_upsert_can_preserve_completed_rows(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    key = {"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA"}
    upsert_rows(registry, [{**key, "status": "complete", "processed_folder": "old"}])
    upsert_rows(registry, [{**key, "status": "registered", "processed_folder": "new"}], preserve_completed=True)
    rows = read_registry(registry)
    assert rows[0]["status"] == "complete"
    assert rows[0]["processed_folder"] == "old"


def test_registry_discovery_refresh_preserves_failed_processing_state(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    key = {"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA"}
    upsert_rows(
        registry,
        [
            {
                **key,
                "status": "failed",
                "preprocess_status": "complete",
                "sort_status": "complete",
                "qc_status": "failed",
                "queue_status": "interrupted",
                "current_step": "interrupted",
                "error_message": "QC stopped",
                "duration_seconds": "100",
            }
        ],
    )
    upsert_rows(
        registry,
        [
            {
                **key,
                "status": "registered",
                "preprocess_status": "pending",
                "sort_status": "pending",
                "qc_status": "pending",
                "duration_seconds": "101",
            }
        ],
        preserve_processing_state=True,
    )
    row = read_registry(registry)[0]
    assert row["status"] == "failed"
    assert row["sort_status"] == "complete"
    assert row["qc_status"] == "failed"
    assert row["queue_status"] == "interrupted"
    assert row["error_message"] == "QC stopped"
    assert row["duration_seconds"] == "101"


def test_unitrefine_cli_row_selection(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    processed = tmp_path / "probe" / "spikeinterface_output"
    upsert_rows(
        registry,
        [
            {"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA", "status": "complete", "processed_folder": str(processed)},
            {"session_id": "s2", "stream_id": "2", "probe_label": "ProbeB", "status": "registered", "processed_folder": "pending"},
        ],
    )
    config = {"project": {"registry_csv": str(registry)}}
    rows = resolve_rows_for_cli(SimpleNamespace(processed_folder=None, session_id="s1", all_complete=False), config)
    assert rows[0]["session_id"] == "s1"
    rows = resolve_rows_for_cli(SimpleNamespace(processed_folder=None, session_id=None, all_complete=True), config)
    assert [row["session_id"] for row in rows] == ["s1"]
    rows = resolve_rows_for_cli(SimpleNamespace(processed_folder=str(processed), session_id=None, all_complete=False), config)
    assert rows[0]["processed_folder"] == str(processed)


def test_output_suffix_is_safe_and_sibling_folder(tmp_path: Path):
    base = tmp_path / "Neuropix-PXI-100.ProbeA" / "spikeinterface_output"
    assert _sanitize_output_suffix("bad chans kept hp300") == "bad_chans_kept_hp300"
    assert suffixed_processed_folder(base, "bad_chans_kept_hp300") == base.parent / "spikeinterface_output_bad_chans_kept_hp300"
    with pytest.raises(ValueError):
        _sanitize_output_suffix("!!!")


def test_run_pending_rerun_complete_only_uses_base_rows(tmp_path: Path):
    config = {
        "project": {
            "output_layout": "recording_local_probe_folder",
            "recording_local_output_name": "spikeinterface_output",
        }
    }
    base = {"status": "complete", "session_id": "session_ProbeA", "processed_folder": str(tmp_path / "spikeinterface_output")}
    variant = {
        "status": "complete",
        "session_id": "session_ProbeA_v2",
        "processed_folder": str(tmp_path / "spikeinterface_output_v2"),
        "notes": "output variant: v2",
    }
    smoke = {
        "status": "complete",
        "session_id": "session_ProbeA_smoke_10s",
        "processed_folder": str(tmp_path / "spikeinterface_output_smoke_10s"),
        "notes": "duration-limited smoke run: 10s",
    }
    assert _is_completed_base_row(base, config)
    assert not _is_completed_base_row(variant, config)
    assert not _is_completed_base_row(smoke, config)


def test_gui_job_command_construction(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("project: {}\n", encoding="utf-8")
    assert build_job_command("discover", config)[2:] == ["pipeline.discover", "--config", str(config)]
    assert build_job_command("backup", config)[2:] == ["pipeline.backup", "--config", str(config), "--all", "--update-registry"]
    assert build_job_command("backup-derived", config)[2:] == ["pipeline.derived_backup", "--config", str(config), "--all", "--update-registry"]
    assert build_job_command("run-pending", config)[2:] == ["pipeline.run_pending", "--config", str(config)]
    assert build_job_command("run-one", config, {"session_id": "s1"})[2:] == [
        "pipeline.run_one",
        "--config",
        str(config),
        "--session-id",
        "s1",
    ]
    assert build_job_command("smoke", config, {"session_id": "s1", "duration_seconds": 10})[2:] == [
        "pipeline.run_one",
        "--config",
        str(config),
        "--session-id",
        "s1",
        "--duration-seconds",
        "10.0",
    ]
    assert build_job_command("rerun-complete", config, {"output_suffix": "v2"})[2:] == [
        "pipeline.run_pending",
        "--config",
        str(config),
        "--rerun-complete",
        "--output-suffix",
        "v2",
    ]
    assert build_job_command("unitrefine", config, {"processed_folder": "out", "label_set": "custom"})[2:] == [
        "pipeline.unitrefine",
        "--config",
        str(config),
        "--processed-folder",
        "out",
        "--label-set",
        "custom",
    ]
    assert build_job_command("resume-qc", config, {"session_id": "s1", "skip_phy": True})[2:] == [
        "pipeline.resume_qc",
        "--config",
        str(config),
        "--session-id",
        "s1",
        "--recompute-extension",
        "spike_locations",
        "--skip-phy",
    ]
    assert build_job_command("restart-qc", config, {"session_id": "s1", "skip_phy": True})[2:] == [
        "pipeline.restart_qc",
        "--config",
        str(config),
        "--session-id",
        "s1",
        "--skip-phy",
    ]
    assert build_job_command("export-phy", config, {"session_id": "s1", "overwrite_phy": True})[2:] == [
        "pipeline.export_phy",
        "--config",
        str(config),
        "--session-id",
        "s1",
        "--overwrite-phy",
    ]
    assert build_job_command("write-summary-png", config, {"include_preprocessing_traces": True})[2:] == [
        "pipeline.reports",
        "--config",
        str(config),
        "--write-summary-png",
        "--include-preprocessing-traces",
    ]
    assert build_job_command(
        "qc-variant",
        config,
        {
            "session_id": "s1",
            "variant_name": "fast",
            "n_jobs": 4,
            "metric_names": "firing_rate,snr",
            "include_spike_locations": False,
            "include_principal_components": True,
            "phy_compute_pc_features": False,
            "skip_phy": True,
        },
    )[2:] == [
        "pipeline.qc_variant",
        "--config",
        str(config),
        "--session-id",
        "s1",
        "--variant-name",
        "fast",
        "--n-jobs",
        "4",
        "--metric-name",
        "firing_rate,snr",
        "--no-include-spike-locations",
        "--include-principal-components",
        "--no-phy-compute-pc-features",
        "--skip-phy",
    ]


def test_gui_job_command_validation(tmp_path: Path):
    config = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="session_id"):
        build_job_command("run-one", config, {})
    with pytest.raises(ValueError, match="output_suffix"):
        build_job_command("rerun-complete", config, {"output_suffix": "   "})
    with pytest.raises(ValueError, match="session_id or processed_folder"):
        build_job_command("unitrefine", config, {})


def test_failed_status_update_marks_running_substatus_failed():
    update = _failed_status_update(
        {"preprocess_status": "complete", "sort_status": "complete", "qc_status": "running", "phy_export_status": "pending"},
        error_message="boom",
    )
    assert update["status"] == "failed"
    assert update["qc_status"] == "failed"
    assert "sort_status" not in update
    assert update["error_message"] == "boom"
    assert _failure_message(KeyboardInterrupt()) == "KeyboardInterrupt"


def test_gui_payloads_and_file_allowlist(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed = raw_root / "session1" / "Record Node 101" / "experiment1" / "recording1" / "continuous" / "ProbeA" / "spikeinterface_output"
    report = processed / "report" / "summary.html"
    png = processed / "report" / "summary.png"
    labels = processed / "unitrefine" / "unitrefine_full" / "unit_labels.csv"
    report.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    report.write_text("<html></html>", encoding="utf-8")
    png.write_bytes(b"png")
    labels.write_text("unit_id,label\n0,sua\n", encoding="utf-8")
    (processed / "quality_metrics.csv").write_text("unit_id,snr\n0,3\n", encoding="utf-8")
    (processed / "unit_summary.csv").write_text("unit_id\n0\n", encoding="utf-8")
    (processed / "channel_qc.csv").write_text("channel_id,label\nCH0,good\n", encoding="utf-8")

    registry = tmp_path / "sessions.csv"
    upsert_rows(
        registry,
        [
            {
                "session_id": "s1",
                "stream_id": "1",
                "probe_label": "ProbeA",
                "status": "complete",
                "backup_status": "verified",
                "processed_folder": str(processed),
                "sorting_analyzer_folder": str(processed / "sorting_analyzer"),
                "quality_metrics_csv": str(processed / "quality_metrics.csv"),
            }
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  raw_root: {raw_root.as_posix()}",
                f"  processed_root: {(tmp_path / 'processed').as_posix()}",
                f"  registry_csv: {registry.as_posix()}",
                "preprocessing:",
                "  car:",
                "    enabled: true",
                "    operator: median",
                "sorting:",
                "  params:",
                "    do_CAR: false",
                "    do_correction: true",
                "    delete_recording_dat: true",
            ]
        ),
        encoding="utf-8",
    )

    status = status_payload(config)
    assert status["rows"][0]["summary_png"] == str(png)
    assert status["rows"][0]["unitrefine_labels"] == str(labels)
    reports = reports_payload(config)
    assert reports["reports"][0]["report"] == str(report)
    assert reports["reports"][0]["unitrefine_labels_csv"] == str(labels)
    policy = config_payload(config)["effective_policy"]
    assert any("SpikeInterface CAR" in item for item in policy)
    detail = probe_payload(config, "s1")
    assert detail["row"]["session_id"] == "s1"
    assert any(step["key"] == "reports" and step["status"] == "complete" for step in detail["workflow"])
    assert config_text_payload(config)["text"].startswith("project:")
    saved = save_config_text(config, config_text_payload(config)["text"])
    assert "effective_policy" in saved
    assert is_allowed_file(report, config, tmp_path)
    assert not is_allowed_file(tmp_path / "outside.txt", config, tmp_path)


def test_nwb_csv_overrides_are_saved_warned_and_clearable(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_folder = raw_root / "jlh602026-06-29_14-31-15"
    registry = tmp_path / "sessions.csv"
    upsert_rows(
        registry,
        [
            {
                "session_id": "jlh602026-06-29_14-31-15_ProbeA",
                "raw_folder": str(raw_folder),
                "stream_id": "1",
                "probe_label": "ProbeA",
                "mouse_id": "jlh60",
                "status": "complete",
            }
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  raw_root: {raw_root.as_posix()}",
                f"  registry_csv: {registry.as_posix()}",
                "mouse_arena_nwb:",
                "  output_folder_name: nwb",
                "  require_complete_probes: false",
            ]
        ),
        encoding="utf-8",
    )

    trials = save_nwb_input_csv_payload(
        config,
        {
            "session_id": "jlh602026-06-29_14-31-15_ProbeA",
            "kind": "trials",
            "filename": "my_trials.csv",
            "csv_text": "t_start,t_end,outcome\n1.0,2.0,hit\n",
        },
    )
    assert Path(trials["saved_csv"]).exists()
    assert "imported_csv" in trials["saved_csv"]
    assert trials["paths"]["active_trials_csv"] == trials["saved_csv"]

    warning_units = save_nwb_input_csv_payload(
        config,
        {
            "session_id": "jlh602026-06-29_14-31-15_ProbeA",
            "kind": "units",
            "filename": "external_units_no_spikes.csv",
            "csv_text": "label,snr\nunit_a,5.0\n",
        },
    )
    assert any("spike times may be empty" in warning for warning in warning_units["warnings"])

    units = save_nwb_input_csv_payload(
        config,
        {
            "session_id": "jlh602026-06-29_14-31-15_ProbeA",
            "kind": "units",
            "filename": "external_units.csv",
            "csv_text": "unit_id,label,snr,spike_times\nunit_a,good,5.0,\"[0.1, 0.2]\"\n",
        },
    )
    assert units["paths"]["active_units_csv"] == units["saved_csv"]
    assert not any("spike times may be empty" in warning for warning in units["warnings"])

    tables = build_mouse_arena_nwb_tables({"project": {"registry_csv": str(registry)}, "mouse_arena_nwb": {"output_folder_name": "nwb"}}, session_id="jlh602026-06-29_14-31-15_ProbeA")
    assert len(tables.trials) == 1
    assert len(tables.units) == 1
    assert tables.manifest["behavior_folder"] == ""
    assert tables.manifest["behavior_to_ephys_alignment"]["mode"] == "external_trials_csv"
    assert tables.spike_times[0].tolist() == [0.1, 0.2]

    cleared = clear_nwb_input_override_payload(
        config,
        {"session_id": "jlh602026-06-29_14-31-15_ProbeA", "kind": "units"},
    )
    assert cleared["removed_path"] == units["saved_csv"]
    assert cleared["paths"]["active_units_csv"] == ""


def test_nwb_digital_events_include_all_lines_edges_and_recording_names(tmp_path: Path):
    recording = tmp_path / "recording1"
    ttl = recording / "events" / "NI-DAQmx-109.PXI-6133" / "TTL"
    continuous = recording / "continuous" / "NI-DAQmx-109.PXI-6133"
    ttl.mkdir(parents=True)
    continuous.mkdir(parents=True)
    np.save(ttl / "states.npy", np.asarray([4, -4, 7, -7, 8], dtype=np.int16))
    np.save(ttl / "timestamps.npy", np.asarray([10.1, 10.2, 10.3, 10.4, 10.5]))
    np.save(ttl / "sample_numbers.npy", np.asarray([1, 2, 3, 4, 5], dtype=np.int64))
    np.save(continuous / "timestamps.npy", np.asarray([10.0, 10.1]))
    config = {
        "nwb": {
            "output_folder_name": "nwb",
            "digital_events": {
                "stream_folder": "NI-DAQmx-109.PXI-6133",
                "include_edges": ["rising", "falling"],
                "line_names": {"4": "reward"},
            },
        }
    }

    events, time_zero = load_digital_events(config, recording_folder=recording)
    assert time_zero == 10.0
    assert events[["line", "edge"]].to_records(index=False).tolist() == [
        (4, "rising"),
        (4, "falling"),
        (7, "rising"),
        (7, "falling"),
        (8, "rising"),
    ]
    assert events.loc[events["line"] == 7, "event_name"].unique().tolist() == ["line_7"]
    inventory = digital_event_inventory(events)
    assert {(row["line"], row["edge"], row["count"]) for row in inventory} == {
        (4, "rising", 1),
        (4, "falling", 1),
        (7, "rising", 1),
        (7, "falling", 1),
        (8, "rising", 1),
    }

    path = save_line_name(config, recording_folder=recording, line=7, name="photodiode_sync")
    assert path.exists()
    assert resolved_line_names(config, recording_folder=recording)[7] == "photodiode_sync"
    renamed, _ = load_digital_events(config, recording_folder=recording)
    assert renamed.loc[renamed["line"] == 7, "event_name"].unique().tolist() == ["photodiode_sync"]


def test_nwb_protocol_rules_keep_visual_stim_separate_from_mouse_arena():
    config = {
        "nwb": {
            "default_protocol": "auto",
            "protocol_rules": [{"match": "(visual[_-]?stim|visualstim|vstim)", "protocol": "visual_stim"}],
        }
    }
    assert nwb_protocol_for_row(config, {"raw_folder": "jlh60_2026-07-02_visual_stim"}) == "visual_stim"
    assert nwb_protocol_for_row(config, {"task": "vstimV1"}) == "visual_stim"
    assert nwb_protocol_for_row(config, {"raw_folder": "jlh602026-07-02_12-00-00"}) == "mouse_arena"


def test_gui_job_manager_guards_active_job(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    upsert_rows(registry, [{"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA", "status": "registered"}])
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["project:", f"  registry_csv: {registry.as_posix()}", f"  raw_root: {tmp_path.as_posix()}"]),
        encoding="utf-8",
    )
    manager = JobManager(repo_root=tmp_path, config_path=config, log_dir=tmp_path / "logs")
    manager._jobs["existing"] = JobRecord(
        job_id="existing",
        kind="run-one",
        command=[],
        status="running",
        created_at="",
        started_at="",
        finished_at="",
        returncode=None,
        stdout_log="",
        stderr_log="",
        error="",
    )
    with pytest.raises(RuntimeError, match="already running"):
        manager.start_job("discover")


def test_gui_job_manager_guards_running_registry(tmp_path: Path):
    registry = tmp_path / "sessions.csv"
    upsert_rows(registry, [{"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA", "status": "sorting"}])
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["project:", f"  registry_csv: {registry.as_posix()}", f"  raw_root: {tmp_path.as_posix()}"]),
        encoding="utf-8",
    )
    manager = JobManager(repo_root=tmp_path, config_path=config, log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError, match="running row"):
        manager.start_job("discover")


def test_restart_qc_running_guard_allows_target_qc_only():
    rows = [
        {"session_id": "s1", "status": "qc_running"},
        {"session_id": "s2", "status": "failed"},
    ]
    assert _blocking_running_rows("restart-qc", {"session_id": "s1"}, rows) == []
    assert _blocking_running_rows("export-phy", {"session_id": "s1"}, rows) == []
    assert _blocking_running_rows("restart-qc", {"session_id": "s2"}, rows) == [rows[0]]
    assert _blocking_running_rows("export-phy", {"session_id": "s2"}, rows) == [rows[0]]
    sorting_rows = [{"session_id": "s1", "status": "sorting"}]
    assert _blocking_running_rows("restart-qc", {"session_id": "s1"}, sorting_rows) == sorting_rows
    assert _blocking_running_rows("run-one", {"session_id": "s1"}, rows) == [rows[0]]


def test_export_phy_refuses_existing_without_overwrite(tmp_path: Path):
    processed = tmp_path / "spikeinterface_output"
    analyzer = processed / "sorting_analyzer"
    phy = processed / "phy"
    analyzer.mkdir(parents=True)
    phy.mkdir(parents=True)
    row = {
        "session_id": "s1",
        "probe_label": "ProbeA",
        "processed_folder": str(processed),
        "sorting_analyzer_folder": str(analyzer),
    }
    config = {"project": {"registry_csv": str(tmp_path / "sessions.csv")}}
    with pytest.raises(FileExistsError, match="Phy folder already exists"):
        export_phy_for_row(row, config, overwrite=False, show_progress=False)


def test_gui_progress_parser_reads_phase_and_tqdm_lines(tmp_path: Path):
    out = tmp_path / "job.out.log"
    err = tmp_path / "job.err.log"
    out.write_text(
        "[5h15m06s] session:ProbeA: START export Phy\n",
        encoding="utf-8",
    )
    err.write_text(
        "\x1b[Aextract PCs (no parallelization):  10%|#         | 252/2501 [24:57<3:50:43,  6.16s/it]\n",
        encoding="utf-8",
    )
    progress = job_progress(
        JobRecord(
            job_id="j1",
            kind="run-pending",
            command=[],
            status="running",
            created_at="",
            started_at="",
            finished_at="",
            returncode=None,
            stdout_log=str(out),
            stderr_log=str(err),
            error="",
        )
    )
    assert progress["phase"]["phase"] == "export Phy"
    assert progress["progress"]["task"] == "extract PCs (no parallelization)"
    assert progress["progress"]["current"] == 252
    assert progress["progress"]["total"] == 2501
    assert "10%" in progress["summary"]


def test_gpu_status_parser_handles_nvidia_smi_csv():
    gpus = _parse_nvidia_smi("0, NVIDIA RTX, 23, 1024, 24576, 61\n")
    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA RTX",
            "utilization_percent": 23.0,
            "memory_used_mib": 1024.0,
            "memory_total_mib": 24576.0,
            "temperature_c": 61.0,
        }
    ]


def test_gui_fastapi_status_config_and_jobs_endpoints(tmp_path: Path):
    from fastapi.testclient import TestClient

    from pipeline_gui.app import create_app

    registry = tmp_path / "sessions.csv"
    upsert_rows(registry, [{"session_id": "s1", "stream_id": "1", "probe_label": "ProbeA", "status": "complete"}])
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  registry_csv: {registry.as_posix()}",
                f"  raw_root: {tmp_path.as_posix()}",
                "sorting:",
                "  params:",
                "    do_CAR: false",
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(config_path=config, repo_root=tmp_path))
    assert client.get("/").status_code == 200
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["rows"][0]["session_id"] == "s1"
    config_response = client.get("/api/config")
    assert config_response.status_code == 200
    assert "effective_policy" in config_response.json()
    jobs = client.get("/api/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["active_job"] is None
    system = client.get("/api/system")
    assert system.status_code == 200
    assert {"cpu", "memory", "disks", "gpus"} <= set(system.json())
    probe = client.get("/api/probes/s1")
    assert probe.status_code == 200
    assert probe.json()["row"]["session_id"] == "s1"
    config_text = client.get("/api/config/text")
    assert config_text.status_code == 200
    assert "project:" in config_text.json()["text"]


def test_bad_channel_ids_are_converted_to_indices():
    rec = FakeRecording()
    assert channel_ids_to_indices(rec, ["CH2", "CH0"]) == [2, 0]


def test_kilosort4_param_policy():
    params = _kilosort4_params({"sorting": {"params": {"do_CAR": True, "nblocks": 2}}}, FakeRecording(), [])
    assert params["do_CAR"] is False
    assert params["do_correction"] is True
    assert params["skip_kilosort_preprocessing"] is False
    assert params["save_preprocessed_copy"] is False
    assert params["delete_recording_dat"] is True
    assert params["bad_channels"] == []

    si_motion_params = _kilosort4_params(
        {"preprocessing": {"motion_correction": {"enabled": True}}, "sorting": {"params": {}}},
        FakeRecording(),
        [],
    )
    assert si_motion_params["do_correction"] is False


def test_bad_channel_detection_does_not_exclude_by_default():
    rec = FakeRecording()
    excluded = _bad_channels_for_sorting(
        rec,
        ["CH1", "CH2"],
        ["good", "noise", "dead"],
        {"preprocessing": {"bad_channels": {"exclude_from_sorting": False}}},
    )
    assert excluded == []

    excluded = _bad_channels_for_sorting(
        rec,
        ["CH1", "CH2"],
        ["good", "noise", "dead"],
        {"preprocessing": {"bad_channels": {"exclude_from_sorting": True, "exclude_labels": ["dead"]}}},
    )
    assert excluded == ["CH2"]


def test_sorting_analyzer_and_phy_export_kwargs_use_configured_parallel_sparse_policy():
    config = {
        "qc": {
            "phy_compute_pc_features": False,
            "phy_compute_amplitudes": True,
            "phy_copy_binary": False,
            "phy_add_unitrefine_labels": True,
            "sorting_analyzer": {"sparse": True, "num_spikes_for_sparsity": 200},
        },
        "jobs": {"n_jobs": 8, "fallback_n_jobs": 2, "chunk_duration": "1s", "progress_bar": True},
    }
    assert _sorting_analyzer_kwargs(config) == {"sparse": True, "num_spikes_for_sparsity": 200}
    assert _job_kwargs(config) == {"n_jobs": 8, "chunk_duration": "1s", "progress_bar": True}
    assert _fallback_n_jobs(config) == 2
    phy_kwargs = _phy_export_kwargs(config)
    metric_kwargs = _phy_metric_export_kwargs(config)
    assert phy_kwargs["compute_pc_features"] is False
    assert phy_kwargs["compute_amplitudes"] is True
    assert phy_kwargs["copy_binary"] is False
    assert phy_kwargs["n_jobs"] == 8
    assert phy_kwargs["chunk_duration"] == "1s"
    assert phy_kwargs["progress_bar"] is True
    assert phy_kwargs["additional_properties"] == ["unitrefine_label", "unitrefine_probability"]
    assert metric_kwargs == {"add_quality_metrics": False, "add_template_metrics": False}


def test_analyzer_extension_specs_and_metric_aliases_are_configurable():
    config = {
        "qc": {
            "analyzer_extensions": [
                "random_spikes",
                {"name": "spike_locations", "enabled": False},
                {"name": "principal_components", "params": {"n_components": 5, "mode": "by_channel_local"}},
                "quality_metrics",
            ],
            "quality_metrics": {
                "metric_names": [
                    "firing_rate",
                    "isi_violations_ratio",
                    "rp_contamination",
                    "snr",
                ]
            },
        }
    }
    specs = _analyzer_extension_specs(config)
    assert [name for name, _params in specs] == ["random_spikes", "principal_components", "quality_metrics"]
    assert specs[1][1] == {"n_components": 5, "mode": "by_channel_local"}
    assert specs[2][1]["metric_names"] == ["firing_rate", "isi_violation", "rp_violation", "snr"]
    assert _quality_metric_compute_names(config) == ["firing_rate", "isi_violation", "rp_violation", "snr"]

    unitrefine_config = {
        "qc": {
            "automated_labels": {"enabled": True, "method": "unitrefine"},
            "analyzer_extensions": [
                {"name": "quality_metrics", "params": {"metric_names": ["firing_rate"]}},
            ],
            "quality_metrics": {
                "unitrefine_metric_names": ["num_spikes", "synchrony", "d_prime", "nearest_neighbor"],
            },
        }
    }
    specs = _analyzer_extension_specs(unitrefine_config)
    assert specs == [
        (
            "quality_metrics",
            {"metric_names": ["firing_rate", "num_spikes", "synchrony", "d_prime", "nearest_neighbor"]},
        )
    ]


def test_qc_variant_config_overrides_are_isolated():
    config = {
        "qc": {
            "phy_compute_pc_features": False,
            "analyzer_extensions": ["spike_locations", "principal_components", "quality_metrics"],
        },
        "jobs": {"n_jobs": 8},
    }
    variant = config_with_variant_overrides(
        config,
        n_jobs=2,
        metric_names=["firing_rate", "snr"],
        include_spike_locations=False,
        include_principal_components=True,
        phy_compute_pc_features=True,
    )
    assert config["jobs"]["n_jobs"] == 8
    assert variant["jobs"]["n_jobs"] == 2
    assert variant["qc"]["phy_compute_pc_features"] is True
    assert variant["qc"]["quality_metrics"]["metric_names"] == ["firing_rate", "snr"]
    spike_locations = next(item for item in variant["qc"]["analyzer_extensions"] if isinstance(item, dict) and item["name"] == "spike_locations")
    assert spike_locations["enabled"] is False


def test_qc_timing_recorder_writes_summary(tmp_path: Path):
    recorder = QCTimingRecorder(tmp_path / "qc_timings.json", context={"session_id": "s1"})
    with recorder.phase("compute waveforms", category="analyzer_extension", metadata={"extension": "waveforms"}):
        pass
    recorder.record("skip existing templates", category="analyzer_extension", status="skipped")
    summary = latest_timing_summary(tmp_path)
    assert summary["exists"] is True
    assert summary["events"][-1]["status"] == "skipped"
    assert summary["total_elapsed_seconds"] == 0


def test_prepare_phy_extensions_reuses_existing_and_computes_missing():
    analyzer = FakePhyExtensionAnalyzer(existing=("spike_amplitudes",))
    config = {
        "qc": {"phy_compute_pc_features": True, "phy_compute_amplitudes": True},
        "jobs": {"n_jobs": 8, "fallback_n_jobs": 2, "chunk_duration": "1s", "progress_bar": True},
    }
    _prepare_phy_extensions(analyzer, config)
    assert analyzer.calls == [
        (
            "principal_components",
            {
                "n_components": 5,
                "mode": "by_channel_local",
                "n_jobs": 8,
                "chunk_duration": "1s",
                "progress_bar": True,
            },
        )
    ]


def test_prepare_phy_extensions_does_not_compute_when_disabled():
    analyzer = FakePhyExtensionAnalyzer()
    _prepare_phy_extensions(analyzer, {"qc": {"phy_compute_pc_features": False, "phy_compute_amplitudes": False}})
    assert analyzer.calls == []


def test_phy_dat_path_patch_points_to_original_continuous_dat(tmp_path: Path):
    probe_folder = tmp_path / "Neuropix-PXI-100.ProbeA"
    processed = probe_folder / "spikeinterface_output"
    phy = processed / "phy"
    phy.mkdir(parents=True)
    continuous = probe_folder / "continuous.dat"
    continuous.write_bytes(b"raw")
    params = phy / "params.py"
    params.write_text(
        "dat_path = r'None'\n"
        "n_channels_dat = 384\n"
        "hp_filtered = True\n",
        encoding="utf-8",
    )
    patched = _patch_phy_dat_path(
        phy,
        {"processed_folder": str(processed)},
        {"qc": {"phy_dat_path": {"enabled": True, "source": "original_continuous_dat", "hp_filtered": False}}},
    )
    assert patched == continuous
    text = params.read_text(encoding="utf-8")
    assert f"dat_path = {str(continuous)!r}" in text
    assert "hp_filtered = False" in text


def test_prune_phy_metric_tsvs_keeps_defaults_and_unitrefine(tmp_path: Path):
    phy = tmp_path / "phy"
    phy.mkdir()
    for name in [
        "cluster_group.tsv",
        "cluster_si_unit_ids.tsv",
        "cluster_unitrefine_label.tsv",
        "cluster_snr.tsv",
        "cluster_amplitude_cutoff.tsv",
    ]:
        (phy / name).write_text("cluster_id\tvalue\n0\tx\n", encoding="utf-8")
    removed = _prune_phy_metric_tsvs(
        phy,
        {"qc": {"phy_add_quality_metrics": False, "phy_add_template_metrics": False, "phy_prune_metric_tsvs": True}},
    )
    assert sorted(path.name for path in removed) == ["cluster_amplitude_cutoff.tsv", "cluster_snr.tsv"]
    assert (phy / "cluster_group.tsv").exists()
    assert (phy / "cluster_si_unit_ids.tsv").exists()
    assert (phy / "cluster_unitrefine_label.tsv").exists()


def test_write_phy_unitrefine_tsvs_from_labels(tmp_path: Path):
    processed = tmp_path / "spikeinterface_output"
    phy = processed / "phy"
    labels = processed / "unitrefine" / "unitrefine_full" / "unit_labels.csv"
    phy.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    (phy / "cluster_si_unit_ids.tsv").write_text("cluster_id\tsi_unit_id\n10\t0\n11\t1\n", encoding="utf-8")
    labels.write_text(
        "unit_id,unitrefine_label,unitrefine_probability\n"
        "0,sua,0.91\n"
        "1,noise,0.82\n",
        encoding="utf-8",
    )
    written = _write_phy_unitrefine_tsvs(phy, processed, {"qc": {"phy_add_unitrefine_labels": True}})
    assert [path.name for path in written] == ["cluster_unitrefine_label.tsv", "cluster_unitrefine_probability.tsv"]
    assert (phy / "cluster_unitrefine_label.tsv").read_text(encoding="utf-8").splitlines() == [
        "cluster_id\tunitrefine_label",
        "10\tsua",
        "11\tnoise",
    ]


def test_analyzer_extension_compute_receives_job_kwargs():
    analyzer = FakeComputeAnalyzer()
    job_kwargs = {"n_jobs": 8, "chunk_duration": "1s", "progress_bar": True}
    _compute_analyzer_extension(analyzer, "quality_metrics", job_kwargs)
    assert analyzer.calls == [("quality_metrics", job_kwargs)]


def test_analyzer_extension_fallback_is_limited_to_unexpected_job_kwargs():
    analyzer = FakeComputeAnalyzer([TypeError("got an unexpected keyword argument 'n_jobs'")])
    job_kwargs = {"n_jobs": 8}
    _compute_analyzer_extension(analyzer, "correlograms", job_kwargs)
    assert analyzer.calls == [("correlograms", job_kwargs), ("correlograms", {})]
    assert _looks_like_unexpected_job_kwarg(TypeError("unexpected keyword argument 'n_jobs'"), job_kwargs)


def test_analyzer_extension_internal_type_error_is_not_silenced():
    analyzer = FakeComputeAnalyzer([TypeError("internal metric type problem")])
    with pytest.raises(TypeError, match="internal metric type problem"):
        _compute_analyzer_extension(analyzer, "quality_metrics", {"n_jobs": 8})
    assert analyzer.calls == [("quality_metrics", {"n_jobs": 8})]


def test_analyzer_extension_retries_with_fallback_jobs_after_windows_spawn_failure(tmp_path: Path):
    extension_folder = tmp_path / "extensions" / "spike_amplitudes"
    extension_folder.mkdir(parents=True)
    (extension_folder / "info.json").write_text("{}", encoding="utf-8")
    analyzer = FakeSpawnFailureAnalyzer(tmp_path)
    fallback = _compute_analyzer_extension(
        analyzer,
        "spike_amplitudes",
        {"n_jobs": 8, "chunk_duration": "1s"},
        fallback_n_jobs=2,
    )
    assert analyzer.calls == [
        ("spike_amplitudes", {"n_jobs": 8, "chunk_duration": "1s"}),
        ("spike_amplitudes", {"n_jobs": 2, "chunk_duration": "1s"}),
    ]
    assert fallback == {"n_jobs": 2, "chunk_duration": "1s"}
    assert not extension_folder.exists()
    assert _looks_like_multiprocessing_spawn_failure(OSError(22, "Invalid argument"), {"n_jobs": 8})
    assert not _looks_like_multiprocessing_spawn_failure(OSError(22, "Invalid argument"), {"n_jobs": 1})


def test_analyzer_extension_can_fall_back_to_serial_if_two_jobs_also_fails(tmp_path: Path):
    extension_folder = tmp_path / "extensions" / "spike_amplitudes"
    extension_folder.mkdir(parents=True)
    analyzer = FakeSpawnFailureAnalyzer(tmp_path, failures=2)
    fallback = _compute_analyzer_extension(
        analyzer,
        "spike_amplitudes",
        {"n_jobs": 8, "chunk_duration": "1s"},
        fallback_n_jobs=2,
    )
    assert analyzer.calls == [
        ("spike_amplitudes", {"n_jobs": 8, "chunk_duration": "1s"}),
        ("spike_amplitudes", {"n_jobs": 2, "chunk_duration": "1s"}),
        ("spike_amplitudes", {"n_jobs": 1, "chunk_duration": "1s"}),
    ]
    assert fallback == {"n_jobs": 1, "chunk_duration": "1s"}


def test_configured_short_analyzer_folder_preserves_legacy_lookup(tmp_path: Path):
    config = {"project": {"sorting_analyzer_folder_name": "analyzer"}}
    processed = tmp_path / "spikeinterface_output"
    legacy = processed / "sorting_analyzer"
    legacy.mkdir(parents=True)

    assert sorting_analyzer_folder_name(config) == "analyzer"
    assert output_paths_for_processed_folder(
        processed,
        analyzer_folder_name=sorting_analyzer_folder_name(config),
    )["sorting_analyzer_folder"] == processed / "analyzer"
    assert sorting_analyzer_folder_for_processed_folder(processed, config) == legacy

    configured = processed / "analyzer"
    configured.mkdir()
    assert sorting_analyzer_folder_for_processed_folder(processed, config) == configured


def test_analyzer_path_preflight_rejects_windows_max_path_before_compute(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_one_module, "_windows_long_paths_disabled", lambda: True)
    config = {
        "qc": {
            "analyzer_extensions": [
                {
                    "name": "principal_components",
                    "enabled": True,
                    "params": {"mode": "by_channel_local"},
                }
            ]
        }
    }
    long_folder = tmp_path / ("very_long_recording_name_" * 8) / "sorting_analyzer"
    with pytest.raises(RuntimeError, match="too long.*sorting_analyzer_folder_name"):
        _validate_analyzer_storage_path(long_folder, config)

    short_folder = tmp_path / "analyzer"
    _validate_analyzer_storage_path(short_folder, config)


def test_delete_analyzer_extension_falls_back_for_partial_extension_folder(tmp_path: Path):
    extension_folder = tmp_path / "extensions" / "spike_locations"
    extension_folder.mkdir(parents=True)
    (extension_folder / "info.json").write_text("{}", encoding="utf-8")
    analyzer = FakeDeleteAnalyzer(tmp_path)
    _delete_analyzer_extension(analyzer, "spike_locations")
    assert not extension_folder.exists()
    assert "spike_locations" not in analyzer.extensions


def test_restart_qc_archives_only_qc_outputs(tmp_path: Path):
    processed = tmp_path / "spikeinterface_output"
    analyzer = processed / "sorting_analyzer"
    sorter = processed / "kilosort4"
    report = processed / "report" / "summary.html"
    analyzer.mkdir(parents=True)
    sorter.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    (analyzer / "spikeinterface_info.json").write_text("{}", encoding="utf-8")
    (sorter / "params.py").write_text("# sorter", encoding="utf-8")
    (processed / "quality_metrics.csv").write_text("unit_id\n", encoding="utf-8")
    report.write_text("<html></html>", encoding="utf-8")

    archive = _archive_qc_outputs(processed)
    assert archive is not None
    assert (archive / "sorting_analyzer" / "spikeinterface_info.json").exists()
    assert (archive / "quality_metrics.csv").exists()
    assert (archive / "report" / "summary.html").exists()
    assert not analyzer.exists()
    assert not (processed / "quality_metrics.csv").exists()
    assert sorter.exists()
    assert (sorter / "params.py").exists()


def test_unitrefine_labels_are_attached_as_phy_properties(tmp_path: Path):
    labels = tmp_path / "unit_labels.csv"
    labels.write_text(
        "unit_id,unitrefine_label,unitrefine_probability\n"
        "0,sua,0.95\n"
        "1,mua,0.72\n"
        "2,noise,0.88\n",
        encoding="utf-8",
    )
    analyzer = FakeAnalyzer()
    _attach_automated_label_properties(
        analyzer,
        {"unit_labels_csv": str(labels)},
        {"qc": {"phy_add_unitrefine_labels": True}},
    )
    assert analyzer.sorting.properties["unitrefine_label"] == ["sua", "mua", "noise"]
    assert analyzer.sorting.properties["unitrefine_probability"] == [0.95, 0.72, 0.88]


def test_unitrefine_exports_labels_and_summary(tmp_path: Path, monkeypatch):
    def fake_unitrefine_label_units(sorting_analyzer, noise_neural_classifier=None, sua_mua_classifier=None):
        return pd.DataFrame(
            {
                "unitrefine_label": ["sua", "mua", "noise"],
                "unitrefine_probability": [0.91, 0.73, 0.88],
            },
            index=[0, 1, 2],
        )

    monkeypatch.setattr("pipeline.unitrefine.unitrefine_label_units", fake_unitrefine_label_units)
    result = run_unitrefine_for_analyzer(
        FakeAnalyzer(),
        tmp_path,
        {"qc": {"automated_labels": {"enabled": True, "method": "unitrefine", "label_set": "unitrefine full"}}},
    )
    assert result["status"] == "complete"
    assert result["label_set"] == "unitrefine_full"
    assert result["label_counts"] == {"sua": 1, "mua": 1, "noise": 1}
    assert Path(result["unit_labels_csv"]).exists()
    assert Path(result["summary_json"]).exists()


def test_unitrefine_failure_policy_records_or_raises(tmp_path: Path, monkeypatch):
    def fake_failure(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("pipeline.unitrefine.unitrefine_label_units", fake_failure)
    config = {"qc": {"automated_labels": {"enabled": True, "method": "unitrefine", "fail_on_error": False}}}
    result = run_unitrefine_for_analyzer(FakeAnalyzer(), tmp_path, config)
    assert result["status"] == "failed"
    assert "model unavailable" in result["error"]
    assert Path(result["summary_json"]).exists()

    with pytest.raises(RuntimeError, match="model unavailable"):
        run_unitrefine_for_analyzer(FakeAnalyzer(), tmp_path / "strict", config, fail_on_error=True)


def test_unitrefine_refuses_to_overwrite_existing_label_set(tmp_path: Path, monkeypatch):
    def fake_unitrefine_label_units(*args, **kwargs):
        return pd.DataFrame({"unitrefine_label": ["sua"], "unitrefine_probability": [0.9]}, index=[0])

    monkeypatch.setattr("pipeline.unitrefine.unitrefine_label_units", fake_unitrefine_label_units)
    config = {"qc": {"automated_labels": {"enabled": True, "method": "unitrefine", "label_set": "custom"}}}
    run_unitrefine_for_analyzer(FakeAnalyzer(), tmp_path, config)
    with pytest.raises(FileExistsError, match="custom"):
        run_unitrefine_for_analyzer(FakeAnalyzer(), tmp_path, config)
    result = run_unitrefine_for_analyzer(FakeAnalyzer(), tmp_path, config, overwrite=True)
    assert result["status"] == "complete"
    assert unitrefine_output_folder(tmp_path, "custom").exists()


def test_channel_qc_records_labels_and_sorting_exclusions(tmp_path: Path):
    rec = FakeRecording()
    csv_path = tmp_path / "channel_qc.csv"
    _write_channel_qc(rec, ["CH1", "CH2"], ["good", "noise", "dead"], ["CH2"], csv_path)
    rows = csv_path.read_text(encoding="utf-8")
    assert "label,detected_bad,excluded_from_sorting" in rows
    assert "CH1,noise,True,False" in rows
    assert "CH2,dead,True,True" in rows


def test_preprocessing_trace_plot_is_written(tmp_path: Path):
    rng = np.random.default_rng(0)
    pre = FakeTraceRecording(rng.normal(size=(2000, 8)))
    post = FakeTraceRecording(rng.normal(size=(2000, 8)))
    outputs = write_preprocessing_trace_plots(
        pre_car_recording=pre,
        post_car_recording=post,
        output_folder=tmp_path,
        seconds=0.25,
        max_channels=4,
    )
    assert Path(outputs["pre_post_car_traces_png"]).exists()


def test_pre_post_trace_scale_uses_both_conditions():
    pre = np.asarray([[0.0, 0.0], [1.0, 2.0], [-1.0, -2.0]])
    post = np.asarray([[0.0, 0.0], [10.0, 4.0], [-10.0, -4.0]])
    scale = _shared_trace_scale(pre, post)
    pre_only_scale = np.nanpercentile(np.abs(pre - np.nanmedian(pre, axis=0, keepdims=True)), 95, axis=0)
    assert np.all(scale > pre_only_scale)


def test_shareable_summary_png_is_written(tmp_path: Path):
    metrics = tmp_path / "quality_metrics.csv"
    metrics.write_text(
        ",num_spikes,firing_rate,presence_ratio,snr,amplitude_median,isi_violations_ratio\n"
        "0,100,1.0,1.0,8.0,-120.0,0.0\n"
        "1,200,5.0,0.8,4.0,-90.0,0.1\n"
        "2,50,0.2,0.5,2.0,-40.0,0.3\n",
        encoding="utf-8",
    )
    channel_qc = tmp_path / "channel_qc.csv"
    channel_qc.write_text(
        "channel_index,channel_id,label,detected_bad,excluded_from_sorting\n"
        "0,CH0,good,False,False\n"
        "1,CH1,noise,True,False\n"
        "2,CH2,good,False,False\n"
        "3,CH3,dead,True,True\n",
        encoding="utf-8",
    )
    labels = tmp_path / "unitrefine" / "unitrefine_full" / "unit_labels.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text(
        "unit_id,unitrefine_label,unitrefine_probability\n"
        "0,sua,0.95\n"
        "1,mua,0.71\n"
        "2,noise,0.84\n",
        encoding="utf-8",
    )
    output = write_shareable_summary_png(
        analyzer=FakeAnalyzer(),
        probe_label="ProbeA",
        output_folder=tmp_path,
        quality_metrics_csv=metrics,
        channel_qc_csv=channel_qc,
        summary={
            "session_id": "session1",
            "probe_label": "ProbeA",
            "num_units": 3,
            "num_channels": 4,
            "duration_seconds": 10,
            "quality_metrics_csv": str(metrics),
            "automated_labels_csv": str(labels),
            "automated_labels_status": "complete",
            "created_at": "2026-06-30T00:00:00+00:00",
        },
        config={"qc": {"shareable_png": {"enabled": True, "filename": "summary.png", "example_units": 3}}},
    )
    assert output.exists()
    assert output.stat().st_size > 10_000
    image = plt.imread(output)
    assert image.shape[0] > 1000
    assert image.shape[1] > 2000


def test_bad_channel_rows_support_old_and_new_formats():
    old = pd.DataFrame({"channel_index": [0, 1], "status": ["good", "noise"]})
    new = pd.DataFrame({"channel_index": [0, 1], "label": ["dead", "good"], "detected_bad": [True, False]})
    assert _bad_channel_rows(old)["channel_index"].tolist() == [1]
    assert _bad_channel_rows(new)["channel_index"].tolist() == [0]


def test_probe_channel_mismatch_fails():
    with pytest.raises(ValueError, match="does not match"):
        validate_probe(FakeProbeRecording(contacts=2))


def test_backup_verification_ignores_pipeline_outputs(tmp_path: Path):
    raw_root = tmp_path / "raw"
    backup_root = tmp_path / "backup"
    raw = raw_root / "session1"
    backup = backup_root / "session1"
    raw.mkdir(parents=True)
    backup.mkdir(parents=True)
    (raw / "continuous.dat").write_bytes(b"123")
    (backup / "continuous.dat").write_bytes(b"123")
    (raw / "spikeinterface_output").mkdir()
    (raw / "spikeinterface_output" / "quality_metrics.csv").write_text("derived", encoding="utf-8")

    config = {
        "project": {"raw_root": str(raw_root)},
        "backup": {"root": str(backup_root), "exclude_dir_prefixes": ["spikeinterface_output"]},
    }
    result = verify_raw_backup(raw, config)
    assert result.ok
    assert result.checked_files == 1
    assert backup_folder_for_raw(raw, config) == backup


def test_backup_verification_fails_on_size_mismatch(tmp_path: Path):
    raw_root = tmp_path / "raw"
    backup_root = tmp_path / "backup"
    raw = raw_root / "session1"
    backup = backup_root / "session1"
    raw.mkdir(parents=True)
    backup.mkdir(parents=True)
    (raw / "continuous.dat").write_bytes(b"123")
    (backup / "continuous.dat").write_bytes(b"12")

    config = {
        "project": {"raw_root": str(raw_root)},
        "backup": {"root": str(backup_root), "exclude_dir_prefixes": ["spikeinterface_output"]},
    }
    result = verify_raw_backup(raw, config)
    assert not result.ok
    assert result.size_mismatches


def test_derived_backup_copies_only_generated_output_folders(tmp_path: Path):
    raw_root = tmp_path / "raw"
    backup_root = tmp_path / "backup"
    raw = raw_root / "session1"
    backup = backup_root / "session1"
    probe = raw / "Record Node 101" / "experiment1" / "recording1" / "continuous" / "Neuropix-PXI-100.ProbeA"
    nwb = raw / "Record Node 101" / "experiment1" / "recording1" / "nwb"
    probe.mkdir(parents=True)
    nwb.mkdir(parents=True)
    backup.mkdir(parents=True)
    (probe / "continuous.dat").write_bytes(b"raw")
    (probe / "spikeinterface_output").mkdir()
    (probe / "spikeinterface_output" / "quality_metrics.csv").write_text("metrics", encoding="utf-8")
    (probe / "spikeinterface_output_fast").mkdir()
    (probe / "spikeinterface_output_fast" / "summary.json").write_text("{}", encoding="utf-8")
    (nwb / "nwb_trials.csv").write_text("start_time,stop_time\n0,1\n", encoding="utf-8")

    config = {
        "project": {"raw_root": str(raw_root)},
        "backup": {
            "root": str(backup_root),
            "derived_outputs": {
                "include_dir_prefixes": ["spikeinterface_output"],
                "include_dir_names": ["nwb"],
            },
        },
    }

    folders = sorted(path.name for path in find_derived_output_folders(raw, config))
    assert folders == ["nwb", "spikeinterface_output", "spikeinterface_output_fast"]
    result = backup_derived_outputs_for_raw(raw, config)
    assert result.ok
    assert result.copied_files == 3
    assert (backup / "Record Node 101" / "experiment1" / "recording1" / "nwb" / "nwb_trials.csv").exists()
    assert (
        backup
        / "Record Node 101"
        / "experiment1"
        / "recording1"
        / "continuous"
        / "Neuropix-PXI-100.ProbeA"
        / "spikeinterface_output"
        / "quality_metrics.csv"
    ).exists()
    assert not (
        backup
        / "Record Node 101"
        / "experiment1"
        / "recording1"
        / "continuous"
        / "Neuropix-PXI-100.ProbeA"
        / "continuous.dat"
    ).exists()


def test_derived_backup_skips_current_files_on_second_run(tmp_path: Path):
    raw_root = tmp_path / "raw"
    backup_root = tmp_path / "backup"
    raw = raw_root / "session1"
    backup = backup_root / "session1"
    output = raw / "spikeinterface_output"
    output.mkdir(parents=True)
    backup.mkdir(parents=True)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    config = {
        "project": {"raw_root": str(raw_root)},
        "backup": {
            "root": str(backup_root),
            "derived_outputs": {"include_dir_prefixes": ["spikeinterface_output"], "include_dir_names": ["nwb"]},
        },
    }

    first = backup_derived_outputs_for_raw(raw, config)
    second = backup_derived_outputs_for_raw(raw, config)

    assert first.copied_files == 1
    assert second.copied_files == 0
    assert second.skipped_files == 1


def test_derived_backup_updates_all_registry_rows_for_recording(tmp_path: Path):
    raw_root = tmp_path / "raw"
    backup_root = tmp_path / "backup"
    raw = raw_root / "session1"
    backup = backup_root / "session1"
    output = raw / "spikeinterface_output"
    output.mkdir(parents=True)
    backup.mkdir(parents=True)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    registry = tmp_path / "sessions.csv"
    upsert_rows(
        registry,
        [
            {"session_id": "s1_ProbeA", "raw_folder": str(raw), "stream_id": "a", "probe_label": "ProbeA", "status": "complete"},
            {"session_id": "s1_ProbeB", "raw_folder": str(raw), "stream_id": "b", "probe_label": "ProbeB", "status": "complete"},
        ],
    )
    config = {
        "project": {"raw_root": str(raw_root), "registry_csv": str(registry)},
        "backup": {
            "root": str(backup_root),
            "derived_outputs": {"include_dir_prefixes": ["spikeinterface_output"], "include_dir_names": ["nwb"]},
        },
    }

    result = backup_derived_outputs_for_recording_session(config, session_id="s1_ProbeA", update_registry=True)
    rows = read_registry(registry)

    assert result.ok
    assert {row["derived_backup_status"] for row in rows} == {"verified"}
    assert all(row["derived_backup_folder"] == str(backup) for row in rows)


def test_local_cleanup_requires_current_synology_and_full_local_archive(tmp_path: Path):
    local_root = tmp_path / "local"
    server_root = tmp_path / "server"
    archive_root = tmp_path / "archive"
    source = local_root / "session1"
    synology = server_root / "session1"
    archive = archive_root / "session1"
    (source / "probe" / "spikeinterface_output").mkdir(parents=True)
    (source / "probe" / "continuous.dat").write_bytes(b"raw-data")
    (source / "probe" / "spikeinterface_output" / "summary.json").write_text("{}", encoding="utf-8")
    shutil.copytree(source, synology)
    shutil.copytree(source, archive)
    rows = [
        {
            "session_id": "s1_ProbeA",
            "status": "complete",
            "raw_folder": str(source),
            "local_raw_folder": str(source),
            "server_raw_folder": str(synology),
            "local_recording_archive_folder": str(archive),
        },
        {
            "session_id": "s1_ProbeB",
            "status": "complete",
            "raw_folder": str(source),
            "local_raw_folder": str(source),
            "server_raw_folder": str(synology),
            "local_recording_archive_folder": str(archive),
        },
    ]
    config = {
        "project": {"raw_root": str(local_root)},
        "staging": {"local_raw_root": str(local_root), "server_raw_root": str(server_root)},
        "backup": {"local_recording_archive": {"root": str(archive_root)}},
    }

    result = assess_recording_cleanup(rows, config, source_folder=source)
    assert result.eligible
    assert result.checked_files == 2

    (archive / "probe" / "continuous.dat").write_bytes(b"wrong")
    blocked = assess_recording_cleanup(rows, config, source_folder=source)
    assert not blocked.eligible
    assert "size mismatch" in "; ".join(blocked.errors)


def test_local_cleanup_deletes_only_after_verification_and_updates_registry(tmp_path: Path):
    local_root = tmp_path / "local"
    server_root = tmp_path / "server"
    archive_root = tmp_path / "archive"
    source = local_root / "session1"
    synology = server_root / "session1"
    archive = archive_root / "session1"
    source.mkdir(parents=True)
    (source / "continuous.dat").write_bytes(b"raw-data")
    shutil.copytree(source, synology)
    shutil.copytree(source, archive)
    registry = tmp_path / "sessions.csv"
    upsert_rows(
        registry,
        [
            {
                "session_id": "s1_ProbeA",
                "stream_id": "a",
                "probe_label": "ProbeA",
                "status": "complete",
                "raw_folder": str(source),
                "local_raw_folder": str(source),
                "server_raw_folder": str(synology),
                "local_recording_archive_folder": str(archive),
            },
            {
                "session_id": "s1_ProbeB",
                "stream_id": "b",
                "probe_label": "ProbeB",
                "status": "complete",
                "raw_folder": str(source),
                "local_raw_folder": str(source),
                "server_raw_folder": str(synology),
                "local_recording_archive_folder": str(archive),
            },
        ],
    )
    config = {
        "project": {"raw_root": str(local_root), "registry_csv": str(registry)},
        "staging": {"local_raw_root": str(local_root), "server_raw_root": str(server_root)},
        "backup": {"local_recording_archive": {"root": str(archive_root)}},
    }

    result = cleanup_recording_for_session(
        config,
        session_id="s1_ProbeA",
        delete_local=True,
        update_registry=True,
    )
    assert result.status == "offloaded"
    assert not source.exists()
    rows = read_registry(registry)
    assert {row["local_cleanup_status"] for row in rows} == {"offloaded"}
    assert {row["local_stage_status"] for row in rows} == {"offloaded"}
