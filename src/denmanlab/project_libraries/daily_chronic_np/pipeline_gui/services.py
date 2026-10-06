from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pipeline.config import load_config
from pipeline.machine import discovery_root
from pipeline.mouse_arena_nwb import (
    build_nwb_digital_event_inventory,
    build_mouse_arena_units_table,
    build_mouse_arena_stimulus_tables,
    nwb_output_paths_for_row,
    save_nwb_digital_line_name,
    save_nwb_digital_line_npy,
    save_mouse_arena_digital_event_npy,
    write_mouse_arena_units_table,
    write_mouse_arena_stimulus_tables,
)
from pipeline.registry import read_registry, update_row
from pipeline.profiles import (
    default_profile_name,
    list_profiles,
    load_profile,
    resolve_profile,
    save_profile,
)
from pipeline.queue_store import QueueStore
from pipeline.queue_worker import default_queue_db
from pipeline.recording_identity import recording_group_payload, recording_identity_for_row
from pipeline.status import RUNNING_STATUSES, summarize_rows
from pipeline.timings import latest_timing_summary, timing_path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_SYSTEM_LOCK = threading.Lock()
_LAST_CPU_TIMES: tuple[int, int] | None = None
_GPU_CACHE: tuple[float, list[dict[str, Any]]] | None = None


def queue_store(repo_root: str | Path) -> QueueStore:
    return QueueStore(default_queue_db(repo_root))


def recordings_payload(config_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    registry_rows = read_registry(config["project"]["registry_csv"])
    minimum_duration = float(config.get("recordings", {}).get("minimum_duration_seconds", 0))
    groups: dict[str, dict[str, Any]] = {}
    for raw_row in registry_rows:
        identity = recording_group_payload(raw_row)
        recording_id = identity["recording_id"]
        group = groups.setdefault(
            recording_id,
            {
                **identity,
                "mouse_id": raw_row.get("mouse_id") or raw_row.get("animal_id", ""),
                "recording_date": raw_row.get("recording_date", ""),
                "task": raw_row.get("task", ""),
                "raw_folder": raw_row.get("raw_folder", ""),
                "server_raw_folder": raw_row.get("server_raw_folder", ""),
                "local_stage_status": raw_row.get("local_stage_status", ""),
                "derived_backup_status": raw_row.get("derived_backup_status", ""),
                "local_recording_archive_status": raw_row.get("local_recording_archive_status", ""),
                "local_cleanup_status": raw_row.get("local_cleanup_status", ""),
                "probes": [],
                "duration_seconds": 0.0,
            },
        )
        duration = float(identity.get("duration_seconds") or 0.0)
        group["duration_seconds"] = max(float(group["duration_seconds"]), duration)
        for status_key in (
            "local_stage_status",
            "derived_backup_status",
            "local_recording_archive_status",
            "local_cleanup_status",
        ):
            if raw_row.get(status_key) == "failed" or not group.get(status_key):
                group[status_key] = raw_row.get(status_key, "")
        group["probes"].append(
            {
                "session_id": raw_row.get("session_id", ""),
                "probe_label": raw_row.get("probe_label", ""),
                "status": raw_row.get("status", ""),
                "preprocess_status": raw_row.get("preprocess_status", ""),
                "sort_status": raw_row.get("sort_status", ""),
                "qc_status": raw_row.get("qc_status", ""),
                "phy_export_status": raw_row.get("phy_export_status", ""),
                "queue_status": raw_row.get("queue_status", ""),
                "current_step": raw_row.get("current_step", ""),
                "active_profile": raw_row.get("active_profile", ""),
                "profile_version": raw_row.get("profile_version", ""),
                "last_updated": raw_row.get("last_updated", ""),
                "error_message": raw_row.get("error_message", ""),
                "processed_folder": raw_row.get("processed_folder", ""),
            }
        )

    store = queue_store(repo_root)
    active_by_recording = {
        item["recording_id"]: item
        for item in store.list_queue(include_finished=False)
    }
    rows: list[dict[str, Any]] = []
    for group in groups.values():
        statuses = [probe["status"] for probe in group["probes"]]
        duration = float(group["duration_seconds"])
        short = duration < minimum_duration
        survey_like = _survey_like_block(group.get("experiment", ""))
        if all(status == "complete" for status in statuses):
            overall = "complete"
        elif any(status in RUNNING_STATUSES for status in statuses):
            overall = "running"
        elif any(status == "failed" for status in statuses):
            overall = "attention"
        elif all(status == "skipped" for status in statuses):
            overall = "skipped"
        else:
            overall = "pending"
        eligibility_reason = ""
        eligible = any(status in {"registered", "failed"} for status in statuses) and not short and not survey_like
        if survey_like:
            eligibility_reason = (
                f"survey block ({group.get('experiment', 'later experiment')}; "
                "excluded from the default queue)"
            )
            if short:
                eligibility_reason += f", {duration:.1f}s"
        elif short:
            eligibility_reason = (
                f"below minimum duration ({duration:.1f}s; minimum {minimum_duration:.0f}s)"
            )
        elif overall == "complete":
            eligibility_reason = "already complete"
        elif not any(status in {"registered", "failed"} for status in statuses):
            eligibility_reason = "not currently runnable"
        group.update(
            {
                "status": overall,
                "eligible": eligible,
                "short_recording": short,
                "survey_like": survey_like,
                "eligibility_reason": eligibility_reason,
                "queue": active_by_recording.get(group["recording_id"]),
                "nwb_status": _recording_nwb_status(config, registry_rows, group["recording_id"]),
            }
        )
        rows.append(group)
    rows.sort(key=lambda row: (row.get("mouse_id", ""), row.get("recording_date", ""), row.get("task", ""), row.get("recording_block", "")))
    return {
        "recordings": rows,
        "minimum_duration_seconds": minimum_duration,
        "queue": store.list_queue(include_finished=True),
        "worker": store.worker_payload(),
        "profiles": list_profiles(config_path),
        "default_profile": default_profile_name(config),
    }


def enqueue_recording_payload(
    config_path: str | Path,
    repo_root: str | Path,
    payload: dict[str, Any],
    *,
    start_worker: bool = True,
) -> dict[str, Any]:
    recording_id = str(payload.get("recording_id") or "").strip()
    if not recording_id:
        raise ValueError("recording_id is required")
    config = load_config(config_path)
    profile_name = str(payload.get("profile_name") or default_profile_name(config)).strip()
    load_profile(config_path, profile_name)
    allow_short = bool(payload.get("allow_short", False))
    allow_survey = bool(payload.get("allow_survey", False))
    target_session_id = str(payload.get("session_id") or "").strip()
    rows = _registry_rows_for_recording(config, recording_id)
    if not rows:
        raise KeyError(recording_id)
    duration = max(float(row.get("duration_seconds") or 0.0) for row in rows)
    minimum = float(config.get("recordings", {}).get("minimum_duration_seconds", 0))
    survey_like = _survey_like_block(recording_group_payload(rows[0]).get("experiment", ""))
    if duration < minimum and not allow_short:
        raise ValueError(
            f"Recording is {duration:.1f}s, below the {minimum:.0f}s minimum. "
            "Use the explicit short-recording override to enqueue it."
        )
    if survey_like and not allow_survey:
        raise ValueError("This is a survey block. Use the explicit survey override to enqueue it.")
    runnable = [
        row for row in rows
        if (not target_session_id or row.get("session_id") == target_session_id)
        and (
            row.get("status") in {"registered", "failed"}
            or _recoverable_running_row(row)
            or (allow_short and duration < minimum and row.get("status") == "skipped")
        )
    ]
    if not runnable:
        raise ValueError("This recording has no pending or recoverable probe rows.")
    store = queue_store(repo_root)
    item = store.enqueue(
        recording_id=recording_id,
        session_ids=[row["session_id"] for row in runnable],
        profile_name=profile_name,
        raw_folder=rows[0].get("raw_folder", ""),
        allow_short=allow_short or allow_survey,
    )
    for position, row in enumerate(runnable, start=1):
        updated = dict(row)
        updated.update(
            {
                "recording_id": recording_id,
                "recording_block": recording_identity_for_row(row)[1],
                "queue_status": "queued",
                "queue_position": str(item["position"]),
                "active_profile": profile_name,
                "current_step": "queued",
            }
        )
        update_row(config["project"]["registry_csv"], updated)
    worker = ensure_queue_worker(repo_root, config_path, store) if start_worker else store.worker_payload()
    return {"item": item, "worker": worker}


def enqueue_all_payload(config_path: str | Path, repo_root: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    current = recordings_payload(config_path, repo_root)
    profile_name = str(payload.get("profile_name") or current["default_profile"])
    queued: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for recording in current["recordings"]:
        if not recording.get("eligible"):
            continue
        try:
            result = enqueue_recording_payload(
                config_path,
                repo_root,
                {"recording_id": recording["recording_id"], "profile_name": profile_name},
                start_worker=False,
            )
            queued.append(result["item"])
        except Exception as exc:
            errors.append({"recording_id": recording["recording_id"], "error": str(exc)})
    store = queue_store(repo_root)
    worker = ensure_queue_worker(repo_root, config_path, store) if queued else store.worker_payload()
    return {"queued": queued, "errors": errors, "worker": worker}


def pause_queue_payload(repo_root: str | Path) -> dict[str, Any]:
    store = queue_store(repo_root)
    store.request_pause()
    store.add_event(event="pause_requested", message="Pause requested; the worker will stop after the active probe.")
    return {"worker": store.worker_payload()}


def resume_queue_payload(config_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    store = queue_store(repo_root)
    store.clear_pause()
    store.add_event(event="resume_requested", message="Queue resume requested.")
    return {"worker": ensure_queue_worker(repo_root, config_path, store)}


def queue_events_payload(repo_root: str | Path, *, queue_id: str = "", session_id: str = "", limit: int = 200) -> dict[str, Any]:
    store = queue_store(repo_root)
    worker = store.worker_payload()
    active_item: dict[str, Any] | None = None
    active_queue_id = str(worker.get("active_queue_id") or "")
    if active_queue_id:
        try:
            active_item = store.get_item(active_queue_id)
        except KeyError:
            active_item = None
    return {
        "events": store.list_events(queue_id=queue_id, session_id=session_id, limit=limit),
        "jobs": store.list_jobs(limit=limit),
        "queue": store.list_queue(include_finished=False),
        "worker": worker,
        "active_item": active_item,
    }


def queue_job_log_payload(
    repo_root: str | Path,
    *,
    job_id: str,
    stream: str = "out",
    tail: int = 200,
) -> dict[str, Any]:
    if stream not in {"out", "err", "events"}:
        raise ValueError("stream must be out, err, or events")
    job = queue_store(repo_root).get_job(job_id)
    path_key = {"out": "stdout_log", "err": "stderr_log", "events": "event_log"}[stream]
    stdout_text = tail_file(job.get("stdout_log", ""), max_lines=max(300, tail))
    stderr_text = tail_file(job.get("stderr_log", ""), max_lines=max(300, tail))
    structured_events = _read_jsonl_events(job.get("event_log", ""), max_lines=100)
    progress = _latest_progress(stderr_text)
    latest_phase = _latest_structured_phase(structured_events)
    summary = _queue_job_summary(job, latest_phase, progress)
    return {
        "job": job,
        "stream": stream,
        "path": job.get(path_key, ""),
        "text": tail_file(job.get(path_key, ""), max_lines=tail),
        "summary": summary,
        "progress": progress,
        "latest_phase": latest_phase,
        "phase_events": [
            event
            for event in structured_events
            if event.get("event") in {"phase_started", "phase_completed", "phase_failed"}
        ][-30:],
        "elapsed_seconds": _job_elapsed_seconds(job),
    }


def _read_jsonl_events(path: str | Path, *, max_lines: int) -> list[dict[str, Any]]:
    text = tail_file(path, max_lines=max_lines)
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _latest_structured_phase(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") in {"phase_started", "phase_completed", "phase_failed"}:
            return event
    return {}


def _queue_job_summary(
    job: dict[str, Any],
    phase: dict[str, Any],
    progress: dict[str, Any],
) -> str:
    if progress:
        eta = f", ETA {progress['eta']}" if progress.get("eta") else ""
        return f"{progress['task']}: {progress['percent']}%{eta}"
    if phase:
        state = str(phase.get("event", "")).removeprefix("phase_").replace("_", " ")
        return f"{phase.get('name', 'pipeline phase')} ({state})"
    return f"{job.get('kind', 'job')} {job.get('status', '')}".strip()


def _job_elapsed_seconds(job: dict[str, Any]) -> float:
    started = _parse_timestamp(str(job.get("started_at") or ""))
    ended = _parse_timestamp(str(job.get("finished_at") or ""))
    if started is None:
        return 0.0
    return max(0.0, ((ended or datetime.now(timezone.utc)) - started).total_seconds())


def profiles_payload(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = list_profiles(config_path)
    details = []
    for row in rows:
        profile = load_profile(config_path, row["name"])
        details.append({**row, "profile": profile, "yaml": yaml.safe_dump(profile, sort_keys=False)})
    return {"profiles": details, "default_profile": default_profile_name(config)}


def save_profile_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    description = str(payload.get("description") or "")
    if "yaml" in payload:
        parsed = yaml.safe_load(str(payload.get("yaml") or "")) or {}
    else:
        parsed = payload.get("profile") or {}
    if not isinstance(parsed, dict):
        raise ValueError("Profile YAML must contain a mapping.")
    overrides = payload.get("form_overrides") or {}
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("form_overrides must be a mapping.")
        _apply_profile_overrides(parsed, overrides)
    return save_profile(config_path, name, parsed, description=description)


def _apply_profile_overrides(profile: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        if dotted_key == "analyzer_extensions":
            continue
        parts = [part for part in str(dotted_key).split(".") if part]
        if not parts:
            continue
        target = profile
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = value

    extension_overrides = overrides.get("analyzer_extensions")
    if isinstance(extension_overrides, dict):
        qc = profile.setdefault("qc", {})
        extensions = qc.setdefault("analyzer_extensions", [])
        for name, enabled in extension_overrides.items():
            extension = next((item for item in extensions if item.get("name") == name), None)
            if extension is None:
                extension = {"name": name}
                extensions.append(extension)
            extension["enabled"] = bool(enabled)


def reconcile_interrupted_payload(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    legacy_active: bool,
) -> dict[str, Any]:
    if legacy_active:
        raise RuntimeError("A legacy GUI job is still active; interruption reconciliation was not applied.")
    config = load_config(config_path)
    store = queue_store(repo_root)
    worker = store.worker_payload()
    worker_live = _worker_is_live(worker)
    active_session = str(worker.get("active_session_id") or "") if worker_live else ""
    queue_reconciliation = {"jobs": 0, "probes": 0, "items": 0}
    if not worker_live:
        queue_reconciliation = store.reconcile_stale_worker(
            error="The queue worker stopped before the probe process completed."
        )
    changed: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for row in read_registry(config["project"]["registry_csv"]):
        if row.get("status") not in RUNNING_STATUSES or row.get("session_id") == active_session:
            continue
        updated = dict(row)
        updated.update(
            {
                "status": "failed",
                "qc_status": "failed" if row.get("status") == "qc_running" else row.get("qc_status", ""),
                "queue_status": "interrupted",
                "current_step": "interrupted",
                "interrupted_at": now,
                "error_message": "Processing was interrupted. Existing Kilosort output was preserved for artifact-aware recovery.",
                "last_updated": now,
            }
        )
        update_row(config["project"]["registry_csv"], updated)
        changed.append(row["session_id"])
    return {
        "reconciled_session_ids": changed,
        "queue_reconciliation": queue_reconciliation,
    }


def ensure_queue_worker(
    repo_root: str | Path,
    config_path: str | Path,
    store: QueueStore | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    store = store or queue_store(root)
    current = store.worker_payload()
    if _worker_is_live(current):
        return current
    log_dir = root / "run_state"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / "queue_worker.out.log"
    stderr = log_dir / "queue_worker.err.log"
    command = [
        sys.executable,
        "-m",
        "pipeline.queue_worker",
        "--config",
        str(Path(config_path).resolve()),
        "--repo-root",
        str(root),
        "--db",
        str(store.path),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with stdout.open("a", encoding="utf-8", errors="replace") as stdout_handle, stderr.open(
        "a", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            creationflags=creationflags,
        )
    store.set_meta("worker_pid", process.pid)
    store.set_meta("worker_status", "starting")
    store.set_meta("worker_started_at", datetime.now(timezone.utc).isoformat())
    return store.worker_payload()


def _registry_rows_for_recording(config: dict[str, Any], recording_id: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_registry(config["project"]["registry_csv"])
        if recording_identity_for_row(row)[0] == recording_id
    ]


def _recording_nwb_status(
    config: dict[str, Any],
    registry_rows: list[dict[str, str]],
    recording_id: str,
) -> str:
    representative = next(
        (row for row in registry_rows if recording_identity_for_row(row)[0] == recording_id),
        None,
    )
    if representative is None:
        return "missing"
    return "complete" if nwb_output_paths_for_row(config, representative).get("nwb") else "pending"


def _recoverable_running_row(row: dict[str, str]) -> bool:
    if row.get("status") not in RUNNING_STATUSES:
        return False
    updated = _parse_timestamp(row.get("last_updated", ""))
    if updated is None:
        return True
    return (datetime.now(timezone.utc) - updated).total_seconds() > 300


def _survey_like_block(experiment: str) -> bool:
    match = re.fullmatch(r"experiment(\d+)", str(experiment or ""), flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) > 1)


def _worker_is_live(worker: dict[str, Any]) -> bool:
    try:
        pid = int(worker.get("pid"))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    kind: str
    command: list[str]
    status: str
    created_at: str
    started_at: str
    finished_at: str
    returncode: int | None
    stdout_log: str
    stderr_log: str
    error: str


class JobManager:
    def __init__(self, *, repo_root: Path, config_path: Path, log_dir: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.config_path = config_path
        self.log_dir = (log_dir or self.repo_root / "run_logs").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        self._refresh_finished_jobs()
        with self._lock:
            return [
                _job_payload(job)
                for job in sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            ]

    def active_job(self) -> dict[str, Any] | None:
        self._refresh_finished_jobs()
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running":
                    return _job_payload(job)
        return None

    def start_job(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        self._refresh_finished_jobs()
        with self._lock:
            active = next((job for job in self._jobs.values() if job.status == "running"), None)
            if active is not None:
                raise RuntimeError(f"GUI job already running: {active.kind} ({active.job_id})")

        config = load_config(self.config_path)
        rows = summarize_rows(config)
        blocking_rows = _blocking_running_rows(kind, payload, rows)
        if blocking_rows:
            sessions = ", ".join(row.get("session_id", "") for row in blocking_rows)
            raise RuntimeError(f"Registry already has a running row ({sessions}); wait for it to finish before launching another GUI job.")

        command = build_job_command(kind, self.config_path, payload)
        safe_kind = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in kind)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{timestamp}_{safe_kind}_{uuid.uuid4().hex[:8]}"
        stdout_log = self.log_dir / f"gui_{job_id}.out.log"
        stderr_log = self.log_dir / f"gui_{job_id}.err.log"

        stdout_handle = stdout_log.open("w", encoding="utf-8", errors="replace")
        stderr_handle = stderr_log.open("w", encoding="utf-8", errors="replace")
        try:
            child_env = os.environ.copy()
            child_env["NUMBA_CACHE_DIR"] = str(self.repo_root / "run_state" / "cache" / "numba")
            child_env["MPLCONFIGDIR"] = str(self.repo_root / "run_state" / "cache" / "matplotlib")
            Path(child_env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(child_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

        now = _now()
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            command=command,
            status="running",
            created_at=now,
            started_at=now,
            finished_at="",
            returncode=None,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            error="",
        )
        with self._lock:
            self._jobs[job_id] = record
            self._processes[job_id] = process
        return _job_payload(record)

    def log_tail(self, job_id: str, stream: str = "out", tail: int = 200) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        path = Path(job.stderr_log if stream == "err" else job.stdout_log)
        return {"job_id": job_id, "stream": stream, "path": str(path), "text": tail_file(path, max_lines=tail)}

    def progress(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job_progress(job)

    def _refresh_finished_jobs(self) -> None:
        with self._lock:
            items = list(self._processes.items())
        for job_id, process in items:
            returncode = process.poll()
            if returncode is None:
                continue
            with self._lock:
                job = self._jobs[job_id]
                job.status = "complete" if returncode == 0 else "failed"
                job.returncode = int(returncode)
                job.finished_at = _now()
                if returncode != 0:
                    job.error = f"process exited with code {returncode}"
                self._processes.pop(job_id, None)


def build_job_command(kind: str, config_path: str | Path, payload: dict[str, Any] | None = None) -> list[str]:
    payload = payload or {}
    base = [sys.executable, "-m"]
    config_args = ["--config", str(config_path)]
    if kind == "discover":
        return [*base, "pipeline.discover", *config_args]
    if kind == "backup":
        return [*base, "pipeline.backup", *config_args, "--all", "--update-registry"]
    if kind == "backup-raw":
        return [*base, "pipeline.raw_backup", *config_args, "--all", "--update-registry"]
    if kind == "stage-pending":
        return [*base, "pipeline.stage_recording", *config_args, "--all-pending", "--update-registry"]
    if kind == "backup-derived":
        return [*base, "pipeline.derived_backup", *config_args, "--all", "--update-registry"]
    if kind == "run-pending":
        return [*base, "pipeline.run_pending", *config_args]
    if kind == "run-one":
        return [*base, "pipeline.run_one", *config_args, "--session-id", _required(payload, "session_id")]
    if kind == "smoke":
        return [
            *base,
            "pipeline.run_one",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
            "--duration-seconds",
            str(float(_required(payload, "duration_seconds"))),
        ]
    if kind == "rerun-complete":
        suffix = str(_required(payload, "output_suffix")).strip()
        if not suffix:
            raise ValueError("output_suffix is required")
        return [*base, "pipeline.run_pending", *config_args, "--rerun-complete", "--output-suffix", suffix]
    if kind == "unitrefine":
        command = [*base, "pipeline.unitrefine", *config_args]
        session_id = str(payload.get("session_id") or "").strip()
        processed_folder = str(payload.get("processed_folder") or "").strip()
        if session_id:
            command.extend(["--session-id", session_id])
        elif processed_folder:
            command.extend(["--processed-folder", processed_folder])
        else:
            raise ValueError("session_id or processed_folder is required")
        _append_optional(command, "--label-set", payload.get("label_set"))
        _append_optional(command, "--noise-neural-classifier", payload.get("noise_neural_classifier"))
        _append_optional(command, "--sua-mua-classifier", payload.get("sua_mua_classifier"))
        return command
    if kind == "resume-qc":
        command = [
            *base,
            "pipeline.resume_qc",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
        ]
        recompute_extensions = payload.get("recompute_extensions") or ["spike_locations"]
        for extension in recompute_extensions:
            extension = str(extension).strip()
            if extension:
                command.extend(["--recompute-extension", extension])
        if bool(payload.get("skip_phy", False)):
            command.append("--skip-phy")
        if bool(payload.get("overwrite_phy", False)):
            command.append("--overwrite-phy")
        return command
    if kind == "restart-qc":
        command = [
            *base,
            "pipeline.restart_qc",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
        ]
        if bool(payload.get("skip_phy", False)):
            command.append("--skip-phy")
        return command
    if kind == "export-phy":
        command = [
            *base,
            "pipeline.export_phy",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
        ]
        if bool(payload.get("overwrite_phy", False)):
            command.append("--overwrite-phy")
        return command
    if kind == "export-nwb":
        command = [
            *base,
            "pipeline.mouse_arena_nwb",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
        ]
        if bool(payload.get("overwrite", False)):
            command.append("--overwrite")
        if bool(payload.get("tables_only", False)):
            command.append("--tables-only")
        return command
    if kind == "qc-variant":
        command = [
            *base,
            "pipeline.qc_variant",
            *config_args,
            "--session-id",
            _required(payload, "session_id"),
            "--variant-name",
            _required(payload, "variant_name"),
        ]
        _append_optional(command, "--n-jobs", payload.get("n_jobs"))
        metric_names = str(payload.get("metric_names") or "").strip()
        if metric_names:
            command.extend(["--metric-name", metric_names])
        for payload_key, flag in [
            ("include_spike_locations", "--include-spike-locations"),
            ("include_principal_components", "--include-principal-components"),
            ("phy_compute_pc_features", "--phy-compute-pc-features"),
        ]:
            if payload_key in payload:
                command.append(flag if bool(payload[payload_key]) else f"--no-{flag[2:]}")
        if bool(payload.get("skip_phy", False)):
            command.append("--skip-phy")
        if bool(payload.get("overwrite_variant", False)):
            command.append("--overwrite-variant")
        return command
    if kind == "write-summary-png":
        command = [*base, "pipeline.reports", *config_args, "--write-summary-png"]
        if bool(payload.get("include_preprocessing_traces", False)):
            command.append("--include-preprocessing-traces")
        return command
    raise ValueError(f"Unknown job kind: {kind}")


def _blocking_running_rows(kind: str, payload: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    running = [row for row in rows if row.get("status") in RUNNING_STATUSES]
    if kind not in {"restart-qc", "export-phy"}:
        return running
    target = str(payload.get("session_id") or "").strip()
    blocking: list[dict[str, str]] = []
    for row in running:
        if row.get("session_id") != target:
            blocking.append(row)
            continue
        if row.get("status") in {"preprocessing", "sorting"}:
            blocking.append(row)
    return blocking


def job_progress(job: JobRecord) -> dict[str, Any]:
    stdout = tail_file(job.stdout_log, max_lines=220)
    stderr = tail_file(job.stderr_log, max_lines=220)
    phase = _latest_phase(stdout)
    progress = _latest_progress(stderr) or _latest_progress(stdout)
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "phase": phase,
        "progress": progress,
        "summary": _progress_summary(phase, progress, job),
    }


def status_payload(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = summarize_rows(config)
    raw_rows = {row.get("session_id", ""): row for row in read_registry(config["project"]["registry_csv"])}
    for row in rows:
        raw = raw_rows.get(row.get("session_id", ""), {})
        identity = recording_group_payload(raw or row)
        row.update(identity)
        row["raw_folder"] = raw.get("raw_folder", "")
        row["server_raw_folder"] = raw.get("server_raw_folder", "")
        row["open_ephys_experiment_name"] = raw.get("open_ephys_experiment_name", "")
        row["open_ephys_block_index"] = raw.get("open_ephys_block_index", "")
        processed = Path(row.get("processed_folder", ""))
        summary_png = processed / "report" / "summary.png"
        summary_json = processed / "summary.json"
        labels = sorted((processed / "unitrefine").glob("*/unit_labels.csv")) if (processed / "unitrefine").exists() else []
        row["summary_png"] = str(summary_png) if summary_png.exists() else ""
        row["summary_json"] = str(summary_json) if summary_json.exists() else ""
        row["unitrefine_labels"] = str(labels[-1]) if labels else ""
        nwb_paths = nwb_output_paths_for_row(config, raw or row)
        row["nwb_status"] = "complete" if nwb_paths.get("nwb") else "pending"
        row["is_running"] = row.get("status") in RUNNING_STATUSES
        for key in ("preprocess_status", "sort_status", "qc_status", "phy_export_status", "error_message"):
            row[key] = raw.get(key, "")
        for key in (
            "local_stage_status",
            "local_stage_folder",
            "local_stage_checked_at",
            "local_stage_error",
            "derived_backup_status",
            "derived_backup_folder",
            "derived_backup_checked_at",
            "derived_backup_error",
            "derived_backup_files",
            "derived_backup_bytes",
            "derived_local_backup_status",
            "derived_local_backup_folder",
            "derived_local_backup_checked_at",
            "derived_local_backup_error",
            "local_recording_archive_status",
            "local_recording_archive_folder",
            "local_recording_archive_checked_at",
            "local_recording_archive_error",
            "local_recording_archive_files",
            "local_recording_archive_bytes",
            "local_cleanup_status",
            "local_cleanup_checked_at",
            "local_cleanup_files",
            "local_cleanup_bytes",
            "local_cleanup_error",
            "nwb_backup_status",
            "nwb_backup_folder",
            "nwb_backup_path",
            "nwb_backup_checked_at",
            "nwb_backup_error",
            "nwb_source_resolution_json",
            "behavior_session_folder",
            "behavior_events_csv",
        ):
            row[key] = raw.get(key, "")
    return {
        "rows": rows,
        "has_running_rows": any(row.get("status") in RUNNING_STATUSES for row in rows),
        "undiscovered_recordings": undiscovered_recordings_payload(config),
    }


def undiscovered_recordings_payload(config: dict[str, Any]) -> list[dict[str, str]]:
    raw_root = discovery_root(config)
    if not raw_root.exists():
        return []
    registered = {
        str(Path(row.get("server_raw_folder") or row.get("raw_folder", "")).resolve()).lower()
        for row in read_registry(config["project"]["registry_csv"])
        if row.get("server_raw_folder") or row.get("raw_folder")
    }
    candidates: list[dict[str, str]] = []
    for folder in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        if folder.name.startswith("archive"):
            continue
        if not any(folder.glob("Record Node */experiment*/recording*/structure.oebin")):
            continue
        key = str(folder.resolve()).lower()
        if key in registered:
            continue
        candidates.append({"name": folder.name, "path": str(folder), "last_modified": datetime.fromtimestamp(folder.stat().st_mtime).isoformat()})
    return candidates


def probe_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    rows = read_registry(config["project"]["registry_csv"])
    row = next((item for item in rows if item.get("session_id") == session_id), None)
    if row is None:
        raise KeyError(session_id)
    processed = Path(row.get("processed_folder", ""))
    labels = sorted((processed / "unitrefine").glob("*/unit_labels.csv")) if (processed / "unitrefine").exists() else []
    outputs = {
        "processed_folder": str(processed) if processed else "",
        "sorter_output_folder": row.get("sorter_output_folder", ""),
        "sorting_analyzer_folder": row.get("sorting_analyzer_folder", ""),
        "quality_metrics_csv": row.get("quality_metrics_csv", ""),
        "channel_qc_csv": str(processed / "channel_qc.csv") if (processed / "channel_qc.csv").exists() else "",
        "summary_html": str(processed / "report" / "summary.html") if (processed / "report" / "summary.html").exists() else "",
        "summary_png": str(processed / "report" / "summary.png") if (processed / "report" / "summary.png").exists() else "",
        "unitrefine_labels": str(labels[-1]) if labels else "",
        "phy_params": str(processed / "phy" / "params.py") if (processed / "phy" / "params.py").exists() else "",
        "qc_timings": str(timing_path(processed)) if timing_path(processed).exists() else "",
        "behavior_session_folder": row.get("behavior_session_folder", ""),
        "behavior_events_csv": row.get("behavior_events_csv", ""),
        "nwb_backup_path": row.get("nwb_backup_path", ""),
        "nwb_backup_folder": row.get("nwb_backup_folder", ""),
        "nwb_source_resolution_json": row.get("nwb_source_resolution_json", ""),
        "local_recording_archive_folder": row.get("local_recording_archive_folder", ""),
    }
    outputs.update({f"nwb_{key}": value for key, value in nwb_output_paths_for_row(config, row).items()})
    return {
        "row": row,
        "workflow": workflow_payload(row, config),
        "outputs": outputs,
        "extension_status": analyzer_extension_payload(Path(row.get("sorting_analyzer_folder", ""))),
        "timings": latest_timing_summary(processed),
        "variants": qc_variant_payload(processed),
    }


def workflow_payload(row: dict[str, str], config: dict[str, Any]) -> list[dict[str, str]]:
    processed = Path(row.get("processed_folder", ""))
    labels = sorted((processed / "unitrefine").glob("*/unit_labels.csv")) if (processed / "unitrefine").exists() else []
    nwb_paths = nwb_output_paths_for_row(config, row)
    return [
        {
            "key": "backup",
            "label": "Backup",
            "status": _stage_status(row.get("backup_status", ""), complete_values={"verified"}),
            "detail": row.get("backup_folder", "") or row.get("backup_error", ""),
        },        {
            "key": "staging",
            "label": "Local Staging",
            "status": _stage_status(
                row.get("local_stage_status", ""),
                complete_values={"staged", "not_required", "offloaded"},
            ),
            "detail": row.get("local_stage_folder", "") or row.get("local_stage_error", ""),
        },
        {
            "key": "preprocessing",
            "label": "Preprocessing",
            "status": row.get("preprocess_status", "") or ("complete" if (processed / "channel_qc.csv").exists() else "pending"),
            "detail": "phase shift -> median CAR -> high-pass -> bad-channel detection",
        },
        {
            "key": "kilosort",
            "label": "Kilosort4",
            "status": row.get("sort_status", "") or _exists_status(Path(row.get("sorter_output_folder", ""))),
            "detail": f"units/status: {row.get('sorter_version', '') or 'n/a'}",
        },
        {
            "key": "analyzer",
            "label": "SortingAnalyzer",
            "status": row.get("qc_status", "") or _exists_status(Path(row.get("sorting_analyzer_folder", ""))),
            "detail": "waveforms, locations, PCs, metrics",
        },
        {
            "key": "unitrefine",
            "label": "UnitRefine",
            "status": "complete" if labels else ("failed" if _unitrefine_failed(processed) else "pending"),
            "detail": str(labels[-1]) if labels else "",
        },
        {
            "key": "phy",
            "label": "Phy Export",
            "status": row.get("phy_export_status", "") or ("complete" if (processed / "phy" / "params.py").exists() else "pending"),
            "detail": "template-gui folder",
        },
        {
            "key": "reports",
            "label": "Reports",
            "status": "complete" if (processed / "report" / "summary.html").exists() or (processed / "report" / "summary.png").exists() else "pending",
            "detail": "HTML and shareable PNG",
        },
        {
            "key": "nwb",
            "label": "Mouse Arena NWB",
            "status": "complete" if nwb_paths.get("nwb") else "pending",
            "detail": "session-level ProbeA+ProbeB package",
        },
        {
            "key": "nwb_backup",
            "label": "NWB Backup",
            "status": _stage_status(row.get("nwb_backup_status", ""), complete_values={"verified"}),
            "detail": row.get("nwb_backup_path", "") or row.get("nwb_backup_error", ""),
        },
        {
            "key": "derived_backup",
            "label": "Derived Backup",
            "status": _stage_status(row.get("derived_backup_status", ""), complete_values={"verified"}),
            "detail": row.get("derived_backup_folder", "") or row.get("derived_backup_error", ""),
        },
        {
            "key": "local_recording_archive",
            "label": "Full Local Archive",
            "status": _stage_status(row.get("local_recording_archive_status", ""), complete_values={"verified"}),
            "detail": row.get("local_recording_archive_folder", "") or row.get("local_recording_archive_error", ""),
        },
        {
            "key": "local_cleanup",
            "label": "Local C: Offload",
            "status": _stage_status(row.get("local_cleanup_status", ""), complete_values={"offloaded"}),
            "detail": row.get("local_cleanup_error", "")
            or (
                f"verified {row.get('local_cleanup_files', '0')} files before offload"
                if row.get("local_cleanup_status") == "offloaded"
                else ""
            ),
        },
    ]


def analyzer_extension_payload(analyzer_folder: Path) -> list[dict[str, str]]:
    names = [
        "random_spikes",
        "waveforms",
        "templates",
        "noise_levels",
        "spike_amplitudes",
        "unit_locations",
        "spike_locations",
        "principal_components",
        "correlograms",
        "isi_histograms",
        "quality_metrics",
        "template_metrics",
    ]
    extension_root = analyzer_folder / "extensions"
    if not analyzer_folder.exists() or not extension_root.exists():
        return [{"name": name, "status": "pending", "path": ""} for name in names]
    rows = []
    for name in names:
        folder = extension_root / name
        status = "complete" if folder.exists() and any(folder.iterdir()) else "pending"
        if folder.exists() and not (folder / "run_info.json").exists():
            status = "needs_recompute"
        rows.append({"name": name, "status": status, "path": str(folder) if folder.exists() else ""})
    return rows


def qc_variant_payload(processed: Path) -> list[dict[str, Any]]:
    root = processed / "qc_variants"
    if not root.exists():
        return []
    variants: list[dict[str, Any]] = []
    for folder in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        report = folder / "report" / "summary.html"
        png = folder / "report" / "summary.png"
        labels = sorted((folder / "unitrefine").glob("*/unit_labels.csv")) if (folder / "unitrefine").exists() else []
        variants.append(
            {
                "name": folder.name,
                "folder": str(folder),
                "summary_html": str(report) if report.exists() else "",
                "summary_png": str(png) if png.exists() else "",
                "qc_params": str(folder / "qc_params.yaml") if (folder / "qc_params.yaml").exists() else "",
                "qc_timings": str(timing_path(folder)) if timing_path(folder).exists() else "",
                "unitrefine_labels": str(labels[-1]) if labels else "",
                "timings": latest_timing_summary(folder, limit=6),
            }
        )
    return variants


def find_registry_row(config: dict[str, Any], session_id: str) -> dict[str, str]:
    for row in read_registry(config["project"]["registry_csv"]):
        if row.get("session_id") == session_id:
            return row
    raise KeyError(session_id)


def config_text_payload(config_path: str | Path) -> dict[str, str]:
    path = Path(config_path)
    return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def save_config_text(config_path: str | Path, text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    path = Path(config_path)
    path.write_text(text, encoding="utf-8")
    config = load_config(path)
    return {"path": str(path), "config": config, "effective_policy": effective_policy(config)}


def reports_payload(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    reports: list[dict[str, str]] = []
    for row in read_registry(config["project"]["registry_csv"]):
        processed = Path(row.get("processed_folder", ""))
        report = processed / "report" / "summary.html"
        if not report.exists():
            continue
        reports.append(
            {
                "session_id": row.get("session_id", ""),
                "mouse_id": row.get("mouse_id") or row.get("animal_id", ""),
                "recording_date": row.get("recording_date", ""),
                "task": row.get("task", ""),
                "probe_label": row.get("probe_label", ""),
                "status": row.get("status", ""),
                "report": str(report),
            }
        )
    for report in reports:
        report_path = Path(report["report"])
        processed = report_path.parents[1]
        png = processed / "report" / "summary.png"
        report["summary_png"] = str(png) if png.exists() else ""
        report["quality_metrics_csv"] = str(processed / "quality_metrics.csv") if (processed / "quality_metrics.csv").exists() else ""
        report["unit_summary_csv"] = str(processed / "unit_summary.csv") if (processed / "unit_summary.csv").exists() else ""
        report["channel_qc_csv"] = str(processed / "channel_qc.csv") if (processed / "channel_qc.csv").exists() else ""
        unitrefine_root = processed / "unitrefine"
        labels = sorted(unitrefine_root.glob("*/unit_labels.csv")) if unitrefine_root.exists() else []
        report["unitrefine_labels_csv"] = str(labels[-1]) if labels else ""
    return {"reports": reports}


def stimulus_behavior_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    tables = build_mouse_arena_stimulus_tables(config, session_id=session_id)
    event_counts = (
        tables.digital_events.groupby(["event_name", "line"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["event_name", "line"])
    )
    output_paths = nwb_output_paths_for_row(config, find_registry_row(config, session_id))
    saved_npys = sorted((tables.output_folder / "digital_events").glob("*.npy")) if (tables.output_folder / "digital_events").exists() else []
    return {
        "session_id": session_id,
        "output_folder": str(tables.output_folder),
        "event_counts": _records(event_counts),
        "alignment": _jsonable(tables.manifest.get("behavior_to_ephys_alignment", {})),
        "frame_sync_residual_ms": _jsonable(tables.manifest.get("frame_sync_residual_ms", {})),
        "trials_preview": _records(tables.trials.head(5)),
        "trials_count": int(len(tables.trials)),
        "session_events_count": int(len(tables.session_events)),
        "digital_events_count": int(len(tables.digital_events)),
        "paths": output_paths,
        "saved_npys": [{"name": path.stem, "path": str(path)} for path in saved_npys],
    }


def digital_event_inventory_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    return _jsonable(build_nwb_digital_event_inventory(config, session_id=session_id))


def save_digital_line_name_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    return save_nwb_digital_line_name(
        config,
        session_id=str(_required(payload, "session_id")),
        line=int(_required(payload, "line")),
        name=str(payload.get("name") or ""),
    )


def save_digital_line_npy_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    return save_nwb_digital_line_npy(
        config,
        session_id=str(_required(payload, "session_id")),
        line=int(_required(payload, "line")),
        edge=str(_required(payload, "edge")),
        output_name=str(_required(payload, "output_name")),
    )


def existing_stimulus_behavior_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    row = find_registry_row(config, session_id)
    paths = nwb_output_paths_for_row(config, row)
    output_folder = Path(paths["output_folder"])
    trials_path = output_folder / "trials_df.csv"
    if not trials_path.exists():
        trials_path = output_folder / "nwb_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"No saved trials dataframe found in {output_folder}")

    trials = pd.read_csv(trials_path)
    session_events_path = output_folder / "session_events.csv"
    session_events_count = int(len(pd.read_csv(session_events_path))) if session_events_path.exists() else 0
    digital_path = output_folder / "nwb_digital_events.csv"
    if digital_path.exists():
        digital_events = pd.read_csv(digital_path)
        event_counts = (
            digital_events.groupby(["event_name", "line"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["event_name", "line"])
        )
    else:
        digital_events = pd.DataFrame()
        event_counts = pd.DataFrame(columns=["event_name", "line", "count"])

    manifest_path = output_folder / "stimulus_behavior_manifest.json"
    if not manifest_path.exists():
        manifest_path = output_folder / "nwb_manifest.json"
    manifest = _read_json_file(manifest_path)
    saved_npys = sorted((output_folder / "digital_events").glob("*.npy")) if (output_folder / "digital_events").exists() else []
    residuals = _frame_sync_residual_ms_from_csv(output_folder / "frame_sync_residuals.csv")
    if not residuals:
        residuals = _jsonable(manifest.get("frame_sync_residual_ms", {}))
    return {
        "session_id": session_id,
        "source": "saved_files",
        "output_folder": str(output_folder),
        "event_counts": _records(event_counts),
        "alignment": _jsonable(manifest.get("behavior_to_ephys_alignment", {})),
        "frame_sync_residual_ms": residuals,
        "trials_preview": _records(trials.head(5)),
        "trials_count": int(len(trials)),
        "session_events_count": session_events_count,
        "digital_events_count": int(len(digital_events)),
        "paths": nwb_output_paths_for_row(config, row),
        "saved_npys": [{"name": path.stem, "path": str(path)} for path in saved_npys],
    }


def save_nwb_input_csv_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    session_id = str(_required(payload, "session_id"))
    kind = str(_required(payload, "kind")).strip().lower()
    if kind not in {"trials", "units"}:
        raise ValueError("kind must be 'trials' or 'units'")
    filename = str(_required(payload, "filename")).strip()
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are accepted")
    csv_text = str(_required(payload, "csv_text"))
    if not csv_text.strip():
        raise ValueError("CSV file is empty")

    row = find_registry_row(config, session_id)
    paths = nwb_output_paths_for_row(config, row)
    output_folder = Path(paths["output_folder"])
    imported_folder = output_folder / "imported_csv"
    imported_folder.mkdir(parents=True, exist_ok=True)
    safe_filename = _safe_filename(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = imported_folder / f"{kind}_{timestamp}_{safe_filename}"
    destination.write_text(csv_text, encoding="utf-8", newline="")

    overrides_path = output_folder / "input_overrides.json"
    overrides = _read_json_file(overrides_path)
    active = dict(overrides.get("active_overrides", {}))
    active[f"{kind}_csv"] = str(destination)
    overrides["active_overrides"] = active
    overrides.setdefault("history", []).append(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "kind": kind,
            "source_filename": filename,
            "path": str(destination),
        }
    )
    overrides_path.write_text(json.dumps(overrides, indent=2, default=str), encoding="utf-8")

    if kind == "trials":
        payload_out = csv_trials_payload(config_path, session_id=session_id, csv_path=destination, source=f"uploaded CSV: {filename}")
    else:
        payload_out = csv_units_payload(config_path, session_id=session_id, csv_path=destination, source=f"uploaded CSV: {filename}")
    payload_out["saved_csv"] = str(destination)
    payload_out["overrides_json"] = str(overrides_path)
    return payload_out


def clear_nwb_input_override_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    session_id = str(_required(payload, "session_id"))
    kind = str(_required(payload, "kind")).strip().lower()
    if kind not in {"trials", "units"}:
        raise ValueError("kind must be 'trials' or 'units'")

    row = find_registry_row(config, session_id)
    paths = nwb_output_paths_for_row(config, row)
    output_folder = Path(paths["output_folder"])
    overrides_path = output_folder / "input_overrides.json"
    overrides = _read_json_file(overrides_path)
    active = dict(overrides.get("active_overrides", {}))
    removed = active.pop(f"{kind}_csv", "")
    overrides["active_overrides"] = active
    overrides.setdefault("history", []).append(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "kind": kind,
            "action": "clear_active_override",
            "removed_path": removed,
        }
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(json.dumps(overrides, indent=2, default=str), encoding="utf-8")
    refreshed_paths = nwb_output_paths_for_row(config, row)
    return {
        "session_id": session_id,
        "kind": kind,
        "removed_path": removed,
        "overrides_json": str(overrides_path),
        "paths": refreshed_paths,
        "warnings": [] if removed else [f"No active {kind} CSV override was set."],
    }


def csv_trials_payload(config_path: str | Path, *, session_id: str, csv_path: str | Path, source: str) -> dict[str, Any]:
    config = load_config(config_path)
    row = find_registry_row(config, session_id)
    trials = pd.read_csv(csv_path)
    paths = nwb_output_paths_for_row(config, row)
    return {
        "session_id": session_id,
        "source": source,
        "output_folder": paths["output_folder"],
        "event_counts": [],
        "alignment": {},
        "frame_sync_residual_ms": {},
        "trials_preview": _records(trials.head(5)),
        "trials_count": int(len(trials)),
        "session_events_count": 0,
        "digital_events_count": 0,
        "paths": paths,
        "saved_npys": [],
        "warnings": _trial_csv_warnings(trials),
    }


def csv_units_payload(config_path: str | Path, *, session_id: str, csv_path: str | Path, source: str) -> dict[str, Any]:
    config = load_config(config_path)
    row = find_registry_row(config, session_id)
    units = pd.read_csv(csv_path)
    paths = nwb_output_paths_for_row(config, row)
    return _units_dataframe_payload(
        session_id=session_id,
        output_folder=paths["output_folder"],
        units=units,
        output_paths=paths,
        source=source,
        warnings=_unit_csv_warnings(units),
    )


def units_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    table = build_mouse_arena_units_table(config, session_id=session_id)
    output_paths = nwb_output_paths_for_row(config, find_registry_row(config, session_id))
    return _units_dataframe_payload(
        session_id=session_id,
        output_folder=str(table.output_folder),
        units=table.units,
        output_paths=output_paths,
        source="rebuilt_from_pipeline_outputs",
    )


def _units_dataframe_payload(
    *,
    session_id: str,
    output_folder: str,
    units: pd.DataFrame,
    output_paths: dict[str, str],
    source: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    probe_counts = (
        units.groupby("probe_label", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("probe_label")
        if "probe_label" in units
        else []
    )
    label_counts = (
        units.groupby("unitrefine_label", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("unitrefine_label")
        if "unitrefine_label" in units
        else []
    )
    preview_columns = [
        "nwb_unit_id",
        "unique_unit_id",
        "probe_label",
        "source_unit_id",
        "n_spikes",
        "firing_rate",
        "snr",
        "presence_ratio",
        "amplitude_cutoff",
        "unitrefine_label",
        "unitrefine_probability",
        "kilosort_label",
        "kilosort_ks_label",
    ]
    preview = units[[column for column in preview_columns if column in units.columns]].head(8)
    return {
        "session_id": session_id,
        "source": source,
        "output_folder": output_folder,
        "unit_count": int(len(units)),
        "probe_counts": _records(probe_counts) if hasattr(probe_counts, "to_dict") else [],
        "label_counts": _records(label_counts) if hasattr(label_counts, "to_dict") else [],
        "columns": list(units.columns),
        "preview": _records(preview),
        "paths": output_paths,
        "warnings": warnings or [],
    }


def existing_units_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    row = find_registry_row(config, session_id)
    paths = nwb_output_paths_for_row(config, row)
    units_path = Path(paths["output_folder"]) / "nwb_units.csv"
    if not units_path.exists():
        raise FileNotFoundError(f"No saved units dataframe found: {units_path}")
    units = pd.read_csv(units_path)
    return _units_dataframe_payload(
        session_id=session_id,
        output_folder=str(units_path.parent),
        units=units,
        output_paths=paths,
        source="saved_files",
    )


def write_units_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    outputs = write_mouse_arena_units_table(config, session_id=session_id)
    return {"session_id": session_id, "outputs": outputs}


def write_stimulus_behavior_payload(config_path: str | Path, session_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    outputs = write_mouse_arena_stimulus_tables(config, session_id=session_id)
    return {"session_id": session_id, "outputs": outputs}


def save_digital_event_payload(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    return save_mouse_arena_digital_event_npy(
        config,
        session_id=str(_required(payload, "session_id")),
        event_name=str(_required(payload, "event_name")),
        output_name=str(_required(payload, "output_name")),
    )


def config_payload(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {"config": config, "machine_role": config.get("machine", {}).get("role", "analysis"), "effective_policy": effective_policy(config)}


def nwb_metadata_payload(config_path: str | Path, session_id: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    nwb_config = config.get("mouse_arena_nwb", {})
    row = None
    mouse_id = ""
    if session_id:
        try:
            row = find_registry_row(config, session_id)
        except KeyError:
            row = None
    if row:
        mouse_id = row.get("mouse_id") or row.get("animal_id", "")
    subjects = nwb_config.get("subjects", {}) or {}
    subject = dict(subjects.get(mouse_id, {}) or nwb_config.get("subject", {}) or {})
    if mouse_id and not subject.get("subject_id"):
        subject["subject_id"] = mouse_id
    if row:
        subject = _subject_with_computed_age(subject, row)
    return {
        "mouse_id": mouse_id,
        "subject": subject,
        "session": dict(nwb_config.get("session", {}) or {}),
    }


def save_nwb_metadata(config_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    nwb_config = config.setdefault("mouse_arena_nwb", {})
    session_id = str(payload.get("session_id") or "")
    row = find_registry_row(config, session_id) if session_id else {}
    mouse_id = row.get("mouse_id") or row.get("animal_id") or str(payload.get("mouse_id") or "")
    if "subject" in payload:
        subject = _clean_metadata_dict(payload.get("subject") or {})
        if mouse_id and not subject.get("subject_id"):
            subject["subject_id"] = mouse_id
        if row:
            subject = _subject_with_computed_age(subject, row)
        if mouse_id:
            subjects = nwb_config.setdefault("subjects", {})
            subjects[mouse_id] = subject
            nwb_config.pop("subject", None)
        else:
            nwb_config["subject"] = subject
    if "session" in payload:
        existing_session = dict(nwb_config.get("session", {}) or {})
        session = _clean_metadata_dict(payload.get("session") or {})
        if isinstance(session.get("keywords"), str):
            session["keywords"] = [item.strip() for item in session["keywords"].split(",") if item.strip()]
        existing_session.update(session)
        nwb_config["session"] = existing_session
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return nwb_metadata_payload(path, session_id or None)


def system_payload(config_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "timestamp": _now(),
        "cpu": _cpu_payload(),
        "memory": _memory_payload(),
        "disks": _disk_payload(config, repo_root),
        "gpus": _gpu_payload(),
    }


def effective_policy(config: dict[str, Any]) -> list[str]:
    preprocessing = config.get("preprocessing", {})
    sorting_params = config.get("sorting", {}).get("params", {})
    bad = preprocessing.get("bad_channels", {})
    car = preprocessing.get("car", {})
    highpass = preprocessing.get("highpass_filter", {})
    labels = config.get("qc", {}).get("automated_labels", {})
    qc = config.get("qc", {})
    analyzer = qc.get("sorting_analyzer", {})
    phy_dat = qc.get("phy_dat_path", {})
    derived_backup = config.get("backup", {}).get("derived_outputs", {})
    local_archive = config.get("backup", {}).get("local_recording_archive", {})
    staging = config.get("staging", {})
    nwb_config = config.get("mouse_arena_nwb", {})
    return [
        f"Machine role: {config.get('machine', {}).get('role', 'analysis')}",
        f"Staging: {'on' if staging.get('enabled', False) else 'off'} ({staging.get('server_raw_root', 'n/a')} -> {staging.get('local_raw_root', 'n/a')})",
        f"phase shift: {'on' if preprocessing.get('phase_shift', {}).get('enabled', True) else 'off'}",
        f"SpikeInterface CAR: {'on' if car.get('enabled', True) else 'off'} ({car.get('operator', 'median')})",
        f"high-pass: {'on' if highpass.get('enabled', True) else 'off'} at {highpass.get('freq_min', 300)} Hz",
        f"bad channel detection: {'on' if bad.get('enabled', True) else 'off'} ({bad.get('method', 'coherence+psd')})",
        f"exclude bad channels from sorting: {bool(bad.get('exclude_from_sorting', False))}",
        f"Kilosort4 CAR: {bool(sorting_params.get('do_CAR', False))}",
        f"Kilosort4 drift correction: {bool(sorting_params.get('do_correction', True))}",
        f"delete temporary recording.dat: {bool(sorting_params.get('delete_recording_dat', True))}",
        f"SortingAnalyzer sparse: {bool(analyzer.get('sparse', True))}",
        (
            f"jobs: n_jobs={config.get('jobs', {}).get('n_jobs', 'default')}, "
            f"fallback_n_jobs={config.get('jobs', {}).get('fallback_n_jobs', 2)}, "
            f"chunk={config.get('jobs', {}).get('chunk_duration', 'default')}"
        ),
        f"Phy PC features: {bool(qc.get('phy_compute_pc_features', True))}",
        (
            f"Phy dat_path: {'on' if phy_dat.get('enabled', False) else 'off'} "
            f"({phy_dat.get('source', 'n/a')}, hp_filtered={bool(phy_dat.get('hp_filtered', False))})"
        ),
        (
            f"Phy metrics columns: quality={bool(qc.get('phy_add_quality_metrics', False))}, "
            f"template={bool(qc.get('phy_add_template_metrics', False))}, "
            f"prune_metric_tsvs={bool(qc.get('phy_prune_metric_tsvs', True))}"
        ),
        f"Quality metrics: {', '.join(qc.get('quality_metrics', {}).get('metric_names', [])) or 'SpikeInterface defaults'}",
        f"Analyzer extensions: {', '.join(_configured_extension_names(qc))}",
        f"Phy UnitRefine labels: {bool(qc.get('phy_add_unitrefine_labels', True))}",
        f"UnitRefine: {'on' if labels.get('enabled', False) else 'off'} ({labels.get('label_set', 'n/a')})",
        f"Mouse arena NWB: {'on' if nwb_config.get('enabled', False) else 'manual/optional'}",
        f"NWB source fallback: {'on' if nwb_config.get('source_resolution', {}).get('enabled', True) else 'off'}; log={nwb_config.get('source_resolution', {}).get('log_filename', 'nwb_source_resolution.json')}",
        f"NWB backup root: {nwb_config.get('backup_root', '') or 'off'}",
        (
            f"Derived-output backup: {'on' if derived_backup.get('enabled', False) else 'off'}; "
            f"auto after processing={bool(derived_backup.get('auto_after_processing', False))}, "
            f"auto after NWB={bool(derived_backup.get('auto_after_nwb_export', False))}, "
            f"local archive={derived_backup.get('local_archive_root', '') or 'off'}"
        ),
        (
            f"Full recording archive: {'on' if local_archive.get('enabled', False) else 'off'}; "
            f"auto after recording={bool(local_archive.get('auto_after_recording', False))}, "
            f"root={local_archive.get('root', '') or 'off'}"
        ),
    ]


def _configured_extension_names(qc: dict[str, Any]) -> list[str]:
    configured = qc.get("analyzer_extensions") or []
    names: list[str] = []
    for item in configured:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("enabled", True):
            names.append(str(item.get("name", "")))
    return [name for name in names if name]


def _stage_status(value: str, *, complete_values: set[str]) -> str:
    if value in complete_values:
        return "complete"
    if value:
        return value
    return "pending"


def _exists_status(path: Path) -> str:
    return "complete" if path.exists() else "pending"


def _unitrefine_failed(processed: Path) -> bool:
    root = processed / "unitrefine"
    if not root.exists():
        return False
    for path in root.iterdir():
        summary = path / "summary.json"
        if not summary.exists():
            continue
        try:
            if '"status": "failed"' in summary.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def _cpu_payload() -> dict[str, Any]:
    percent = _windows_cpu_percent()
    return {"percent": percent, "available": percent is not None}


def _windows_cpu_percent() -> float | None:
    if os.name != "nt":
        return None
    idle = ctypes.wintypes.FILETIME()
    kernel = ctypes.wintypes.FILETIME()
    user = ctypes.wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    idle_ticks = _filetime_to_int(idle)
    total_ticks = _filetime_to_int(kernel) + _filetime_to_int(user)
    global _LAST_CPU_TIMES
    with _SYSTEM_LOCK:
        previous = _LAST_CPU_TIMES
        _LAST_CPU_TIMES = (idle_ticks, total_ticks)
    if previous is None:
        return None
    idle_delta = idle_ticks - previous[0]
    total_delta = total_ticks - previous[1]
    if total_delta <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), 1)


def _filetime_to_int(value) -> int:
    return (int(value.dwHighDateTime) << 32) + int(value.dwLowDateTime)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_payload() -> dict[str, Any]:
    if os.name != "nt":
        return {"available": False}
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"available": False}
    total = int(status.ullTotalPhys)
    available = int(status.ullAvailPhys)
    used = max(0, total - available)
    return {
        "available": True,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": available,
        "percent": round(float(status.dwMemoryLoad), 1),
    }


def _disk_payload(config: dict[str, Any], repo_root: str | Path) -> list[dict[str, Any]]:
    candidates = [
        ("repo", Path(repo_root)),
        ("raw", Path(config.get("project", {}).get("raw_root", ""))),
        ("processed", Path(config.get("project", {}).get("processed_root", ""))),
    ]
    seen: set[str] = set()
    disks: list[dict[str, Any]] = []
    for label, path in candidates:
        usage_path = _nearest_existing_path(path)
        if usage_path is None:
            continue
        key = str(usage_path.anchor or usage_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        usage = shutil.disk_usage(usage_path)
        used = usage.total - usage.free
        disks.append(
            {
                "label": label,
                "path": str(usage_path),
                "total_bytes": usage.total,
                "used_bytes": used,
                "free_bytes": usage.free,
                "percent": round(100.0 * used / usage.total, 1) if usage.total else None,
            }
        )
    return disks


def _nearest_existing_path(path: Path) -> Path | None:
    if not str(path):
        return None
    candidate = path.expanduser()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    return candidate


def _gpu_payload() -> list[dict[str, Any]]:
    global _GPU_CACHE
    now = time.monotonic()
    with _SYSTEM_LOCK:
        if _GPU_CACHE and now - _GPU_CACHE[0] < 10.0:
            return _GPU_CACHE[1]
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        gpus: list[dict[str, Any]] = []
    else:
        gpus = _parse_nvidia_smi(completed.stdout) if completed.returncode == 0 else []
    with _SYSTEM_LOCK:
        _GPU_CACHE = (now, gpus)
    return gpus


def _parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, util, mem_used, mem_total, temp = parts
        gpus.append(
            {
                "index": _to_int(index),
                "name": name,
                "utilization_percent": _to_float(util),
                "memory_used_mib": _to_float(mem_used),
                "memory_total_mib": _to_float(mem_total),
                "temperature_c": _to_float(temp),
            }
        )
    return gpus


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _frame_sync_residual_ms_from_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    candidates = [
        column
        for column in ["fit_residual_seconds", "residual_seconds", "nearest_residual_seconds"]
        if column in frame.columns
    ]
    if not candidates:
        candidates = [column for column in frame.columns if "residual" in column and "seconds" in column]
    if not candidates:
        return {}
    residual = pd.to_numeric(frame[candidates[0]], errors="coerce").dropna().to_numpy(dtype=float) * 1000.0
    if residual.size == 0:
        return {}
    return {
        "mean": float(residual.mean()),
        "median": float(np.median(residual)),
        "median_abs": float(np.median(np.abs(residual))),
        "q05": float(np.percentile(residual, 5)),
        "q95": float(np.percentile(residual, 95)),
    }


def _trial_csv_warnings(trials: pd.DataFrame) -> list[str]:
    columns = set(trials.columns)
    if {"start_time", "stop_time"} <= columns or {"t_start", "t_end"} <= columns:
        return []
    return ["Trials CSV should include start_time/stop_time or t_start/t_end before NWB export."]


def _unit_csv_warnings(units: pd.DataFrame) -> list[str]:
    columns = set(units.columns)
    has_match_key = "unique_unit_id" in columns or {"probe_label", "source_unit_id"} <= columns or {"probe_id", "unit_id"} <= columns
    has_spike_times = "spike_times" in columns or "times" in columns
    warnings: list[str] = []
    if not has_match_key and not has_spike_times:
        warnings.append(
            "Units CSV does not include unique_unit_id, probe/source unit columns, or spike_times/times. "
            "NWB export can include the rows, but unit spike times may be empty."
        )
    if "nwb_unit_id" in columns:
        warnings.append("Uploaded nwb_unit_id values are used for matching only; NWB row ids are reassigned sequentially on export.")
    return warnings


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_metadata_dict(payload: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            clean[key] = value.strip()
        else:
            clean[key] = value
    return clean


def _safe_filename(value: str) -> str:
    name = Path(value).name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "uploaded.csv"


def _subject_with_computed_age(subject: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    updated = dict(subject)
    dob = str(updated.get("date_of_birth") or "").strip()
    if dob:
        age = _age_from_dob_for_row(dob, row)
        if age:
            updated["age"] = age
    return updated


def _age_from_dob_for_row(date_of_birth: str, row: dict[str, str]) -> str:
    dob = _parse_date(date_of_birth)
    session_dt = _recording_datetime_for_row(row)
    if dob is None or session_dt is None:
        return ""
    days = max(0, int((session_dt.date() - dob.date()).days))
    return f"P{days}D"


def _parse_date(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    return parsed


def _recording_datetime_for_row(row: dict[str, str]) -> datetime | None:
    raw_folder = str(row.get("raw_folder") or "")
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", raw_folder)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
    date_value = str(row.get("recording_date") or "")
    if date_value:
        return _parse_date(date_value)
    return None


def allowed_file_roots(config_path: str | Path, repo_root: str | Path) -> list[Path]:
    config = load_config(config_path)
    roots = [Path(repo_root).resolve()]
    for key in ("raw_root", "processed_root"):
        value = config.get("project", {}).get(key)
        if value:
            roots.append(Path(value).resolve())
    rows = read_registry(config["project"]["registry_csv"])
    for row in rows:
        for key in ("processed_folder", "sorter_output_folder", "sorting_analyzer_folder"):
            value = row.get(key)
            if value:
                roots.append(Path(value).resolve())
    return _dedupe_existing_roots(roots)


def is_allowed_file(path: str | Path, config_path: str | Path, repo_root: str | Path) -> bool:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        return False
    return any(_is_relative_to(candidate, root) for root in allowed_file_roots(config_path, repo_root))


def tail_file(path: str | Path, max_lines: int = 200) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-max(1, int(max_lines)) :])


def _job_payload(job: JobRecord) -> dict[str, Any]:
    payload = asdict(job)
    payload["progress"] = job_progress(job)
    return payload


def _latest_phase(text: str) -> dict[str, str]:
    latest: dict[str, str] = {}
    pattern = re.compile(r"\[(?P<elapsed>[^\]]+)\]\s+(?P<label>.+?):\s+(?P<event>START|DONE|FAILED)\s+(?P<phase>.+)")
    for raw_line in text.splitlines():
        line = _clean_console_line(raw_line)
        match = pattern.search(line)
        if not match:
            continue
        latest = {
            "elapsed": match.group("elapsed").strip(),
            "label": match.group("label").strip(),
            "event": match.group("event").strip(),
            "phase": match.group("phase").strip(),
            "line": line,
        }
    return latest


def _latest_progress(text: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    pattern = re.compile(
        r"(?P<task>[^:\r\n]+):\s+"
        r"(?P<percent>\d+)%\|.*?\|\s+"
        r"(?P<current>\d+)/(?P<total>\d+)\s+"
        r"\[(?P<elapsed>[^<\]]+)(?:<(?P<eta>[^,\]]+))?"
    )
    for raw_line in text.splitlines():
        line = _clean_console_line(raw_line)
        match = pattern.search(line)
        if not match:
            continue
        current = int(match.group("current"))
        total = int(match.group("total"))
        latest = {
            "task": match.group("task").strip(),
            "percent": int(match.group("percent")),
            "current": current,
            "total": total,
            "elapsed": (match.group("elapsed") or "").strip(),
            "eta": (match.group("eta") or "").strip(),
            "line": line,
        }
    return latest


def _progress_summary(phase: dict[str, str], progress: dict[str, Any], job: JobRecord) -> str:
    if progress:
        eta = f", ETA {progress['eta']}" if progress.get("eta") else ""
        return (
            f"{progress['task']}: {progress['current']}/{progress['total']} "
            f"({progress['percent']}%{eta})"
        )
    if phase:
        return f"{phase.get('event', '')} {phase.get('phase', '')} at {phase.get('elapsed', '')}".strip()
    return f"{job.kind} {job.status}"


def _clean_console_line(line: str) -> str:
    line = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)
    line = line.replace("\r", "")
    return line.strip()


def _append_optional(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    value = str(value).strip()
    if value:
        command.extend([flag, value])


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is required")
    return str(value)


def _dedupe_existing_roots(paths: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        if any(resolved == existing for existing in roots):
            continue
        roots.append(resolved)
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()










