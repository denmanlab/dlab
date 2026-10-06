from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .derived_backup import backup_derived_outputs_for_recording_session
from .file_sync import SyncProgress
from .local_cleanup import cleanup_recording_for_session
from .local_recording_archive import archive_recording_for_session
from .profiles import resolve_profile
from .queue_store import QueueStore
from .recording_identity import recording_identity_for_row
from .registry import read_registry, update_row
from .stage_recording import stage_recording_for_session


def default_queue_db(repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "run_state" / "pipeline_queue.sqlite3"


class QueueWorker:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        config_path: str | Path,
        db_path: str | Path | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.store = QueueStore(db_path or default_queue_db(self.repo_root))
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.jobs_root = self.repo_root / "run_state" / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def run(self, *, once: bool = False) -> None:
        self.store.set_meta("worker_pid", os.getpid())
        self.store.set_meta("worker_status", "running")
        self.store.set_meta("worker_started_at", _now())
        self.store.add_event(event="worker_started", message=f"Queue worker started with PID {os.getpid()}.")
        try:
            while True:
                self.store.set_meta("worker_heartbeat_at", _now())
                if self.store.pause_requested():
                    self.store.set_meta("worker_status", "paused")
                    if once:
                        return
                    time.sleep(self.poll_seconds)
                    continue
                item = self.store.next_item()
                if item is None:
                    self.store.set_meta("worker_status", "idle")
                    self.store.set_meta("active_queue_id", "")
                    self.store.set_meta("active_session_id", "")
                    if once:
                        return
                    time.sleep(self.poll_seconds)
                    continue
                self.store.set_meta("worker_status", "running")
                self._process_recording(item)
                if once:
                    return
        except BaseException as exc:
            self.store.set_meta("worker_status", "failed")
            self.store.add_event(level="error", event="worker_failed", message=_failure_message(exc))
            raise
        finally:
            self.store.set_meta("worker_heartbeat_at", _now())
            self.store.set_meta("active_queue_id", "")
            self.store.set_meta("active_session_id", "")

    def _process_recording(self, item: dict[str, Any]) -> None:
        queue_id = item["queue_id"]
        self.store.set_meta("active_queue_id", queue_id)
        self.store.update_item(queue_id, status="running", started_at=item.get("started_at") or _now(), error="")
        self.store.add_event(
            queue_id=queue_id,
            event="recording_started",
            message=f"Started recording {item['recording_id']}.",
        )

        base_config = load_config(self.config_path)
        rows = self._rows_for_item(item, base_config)
        if not rows:
            self.store.update_item(queue_id, status="attention", error="No matching registry rows.", finished_at=_now())
            self.store.add_event(queue_id=queue_id, level="error", event="recording_failed", message="No matching registry rows.")
            return

        representative = rows[0]
        try:
            self.store.set_meta("active_session_id", representative["session_id"])
            self.store.add_event(
                queue_id=queue_id,
                session_id=representative["session_id"],
                event="staging_started",
                message="Staging the acquisition session locally.",
            )
            stage_recording_for_session(
                base_config,
                session_id=representative["session_id"],
                update_registry=True,
                progress_callback=self._staging_progress_callback(
                    queue_id=queue_id,
                    session_id=representative["session_id"],
                ),
            )
            self.store.add_event(
                queue_id=queue_id,
                session_id=representative["session_id"],
                event="staging_complete",
                message="Local staging is complete.",
            )
            rows = self._rows_for_item(item, base_config)
            representative = rows[0]
        except BaseException as exc:
            message = _failure_message(exc)
            self.store.update_item(queue_id, status="attention", error=message, finished_at=_now())
            self.store.add_event(queue_id=queue_id, level="error", event="staging_failed", message=message)
            return

        any_failed = False
        for probe in item.get("probes", []):
            session_id = probe["session_id"]
            if probe.get("status") == "complete":
                continue
            row = next((candidate for candidate in rows if candidate.get("session_id") == session_id), None)
            if row is None:
                self.store.update_probe(queue_id, session_id, status="failed", error="Registry row missing.", finished_at=_now())
                any_failed = True
                continue
            if row.get("status") == "complete":
                self.store.update_probe(queue_id, session_id, status="complete", current_step="complete", finished_at=_now())
                continue

            success = self._process_probe(item, row)
            any_failed = any_failed or not success
            if self.store.pause_requested():
                self.store.update_item(queue_id, status="paused", current_probe="", error="")
                self.store.set_meta("worker_status", "paused")
                self.store.add_event(
                    queue_id=queue_id,
                    session_id=session_id,
                    event="queue_paused",
                    message="Pause requested; stopped after the completed probe attempt.",
                )
                return

        backup_errors = self._finalize_recording(item, representative, base_config)
        any_failed = any_failed or bool(backup_errors)
        final_status = "attention" if any_failed else "complete"
        self.store.update_item(
            queue_id,
            status=final_status,
            current_probe="",
            error="; ".join(backup_errors),
            finished_at=_now(),
        )
        self.store.add_event(
            queue_id=queue_id,
            level="warning" if any_failed else "info",
            event="recording_finished",
            message=f"Recording finished with status {final_status}.",
        )

    def _staging_progress_callback(self, *, queue_id: str, session_id: str):
        return self._sync_progress_callback(
            queue_id=queue_id,
            session_id=session_id,
            event="staging_progress",
            label="Staging",
        )

    def _sync_progress_callback(
        self,
        *,
        queue_id: str,
        session_id: str,
        event: str,
        label: str,
    ):
        last_heartbeat = 0.0
        last_event = 0.0
        last_file = ""
        last_percent = -1

        def report(progress: SyncProgress) -> None:
            nonlocal last_heartbeat, last_event, last_file, last_percent
            now = time.monotonic()
            if now - last_heartbeat >= 2.0:
                self.store.set_meta("worker_heartbeat_at", _now())
                last_heartbeat = now

            percent_whole = int(progress.percent)
            file_changed = str(progress.relative_path) != last_file
            finished = progress.status in {"complete", "skipped", "dry_run"}
            should_emit = (
                last_event == 0.0
                or now - last_event >= 15.0
                or file_changed
                or finished
                or percent_whole >= last_percent + 5
            )
            if not should_emit:
                return
            payload = {
                "percent": round(progress.percent, 1),
                "completed_bytes": progress.completed_bytes,
                "total_bytes": progress.total_bytes,
                "file_index": progress.file_index,
                "total_files": progress.total_files,
                "relative_path": str(progress.relative_path),
                "file_bytes_copied": progress.file_bytes_copied,
                "file_bytes_total": progress.file_bytes_total,
                "status": progress.status,
            }
            message = (
                f"{label} {progress.percent:.1f}% "
                f"({_format_bytes(progress.completed_bytes)} / {_format_bytes(progress.total_bytes)}); "
                f"file {progress.file_index}/{progress.total_files}: {progress.relative_path} "
                f"({_format_bytes(progress.file_bytes_copied)} / {_format_bytes(progress.file_bytes_total)})."
            )
            self.store.add_event(
                queue_id=queue_id,
                session_id=session_id,
                event=event,
                message=message,
                payload=payload,
            )
            last_event = now
            last_file = str(progress.relative_path)
            last_percent = percent_whole

        return report

    def _process_probe(self, item: dict[str, Any], row: dict[str, str]) -> bool:
        queue_id = item["queue_id"]
        session_id = row["session_id"]
        profile_name = str(item.get("profile_name") or "default")
        effective_config, profile_meta = resolve_profile(self.config_path, profile_name)
        _disable_probe_level_transport(effective_config)
        restored = _restore_recoverable_interrupted_output(row)
        if restored is not None:
            self.store.add_event(
                queue_id=queue_id,
                session_id=session_id,
                event="recoverable_output_restored",
                message=f"Restored completed Kilosort output from {restored}; QC will restart without sorting.",
            )
        action = _next_probe_action(row)

        if action == "run-one" and _has_partial_outputs(row):
            archived = _archive_interrupted_output(row)
            self.store.add_event(
                queue_id=queue_id,
                session_id=session_id,
                level="warning",
                event="partial_output_archived",
                message=f"Archived interrupted output before restarting: {archived}",
            )

        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        source_root = job_dir / "source"
        source_hash = _snapshot_source(self.repo_root, source_root)
        config_snapshot = job_dir / "effective_config.yaml"
        config_snapshot.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
        config_hash = _file_hash(config_snapshot)
        stdout_log = job_dir / "stdout.log"
        stderr_log = job_dir / "stderr.log"
        event_log = job_dir / "events.jsonl"
        command = _probe_command(
            source_root=source_root,
            config_snapshot=config_snapshot,
            session_id=session_id,
            action=action,
        )

        self.store.set_meta("active_session_id", session_id)
        self.store.update_item(queue_id, current_probe=session_id)
        self.store.update_probe(
            queue_id,
            session_id,
            status="running",
            current_step=action,
            profile_name=profile_name,
            profile_version=int(profile_meta.get("version") or 1),
            config_hash=config_hash,
            source_hash=source_hash,
            job_id=job_id,
            error="",
            started_at=_now(),
        )
        self._update_registry_runtime(
            row,
            queue_status="running",
            current_step=action,
            active_profile=profile_name,
            profile_version=str(profile_meta.get("version") or 1),
            active_config_hash=config_hash,
            active_source_hash=source_hash,
            active_job_id=job_id,
            error_message="",
        )
        self.store.add_event(
            queue_id=queue_id,
            session_id=session_id,
            event="probe_started",
            message=f"{action} started with profile {profile_name} v{profile_meta.get('version', 1)}.",
            payload={"config_hash": config_hash, "source_hash": source_hash},
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root)
        env["PIPELINE_EVENT_LOG"] = str(event_log)
        env["NUMBA_CACHE_DIR"] = str(self.repo_root / "run_state" / "cache" / "numba")
        env["MPLCONFIGDIR"] = str(self.repo_root / "run_state" / "cache" / "matplotlib")
        Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with stdout_log.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_log.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                creationflags=creationflags,
            )
            self.store.add_job(
                job_id=job_id,
                queue_id=queue_id,
                session_id=session_id,
                kind=action,
                command=command,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                event_log=str(event_log),
                pid=process.pid,
                profile_name=profile_name,
                profile_version=int(profile_meta.get("version") or 1),
                config_hash=config_hash,
                source_hash=source_hash,
            )
            while process.poll() is None:
                self.store.set_meta("worker_heartbeat_at", _now())
                latest = _latest_phase_event(event_log)
                if latest:
                    self.store.update_probe(queue_id, session_id, current_step=str(latest.get("name") or action))
                time.sleep(self.poll_seconds)
            returncode = int(process.returncode or 0)

        error = "" if returncode == 0 else _last_error(stderr_log, stdout_log)
        self.store.finish_job(job_id, returncode=returncode, error=error)
        status = "complete" if returncode == 0 else "failed"
        self.store.update_probe(
            queue_id,
            session_id,
            status=status,
            current_step="complete" if returncode == 0 else "failed",
            error=error,
            finished_at=_now(),
        )
        refreshed = _registry_row(effective_config, session_id) or row
        self._update_registry_runtime(
            refreshed,
            queue_status=status,
            current_step="complete" if returncode == 0 else "failed",
            active_job_id=job_id,
            error_message="" if returncode == 0 else error,
        )
        self.store.add_event(
            queue_id=queue_id,
            session_id=session_id,
            level="info" if returncode == 0 else "error",
            event="probe_finished",
            message=f"Probe finished with status {status}.",
            payload={"returncode": returncode, "error": error},
        )
        return returncode == 0

    def _finalize_recording(
        self,
        item: dict[str, Any],
        representative: dict[str, str],
        config: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        queue_id = item["queue_id"]
        try:
            self.store.add_event(queue_id=queue_id, event="derived_backup_started", message="Backing up generated outputs.")
            result = backup_derived_outputs_for_recording_session(
                config,
                session_id=representative["session_id"],
                update_registry=True,
            )
            if not result.ok:
                errors.append(result.error or "Generated-output backup failed.")
            self.store.add_event(
                queue_id=queue_id,
                level="warning" if not result.ok else "info",
                event="derived_backup_finished",
                message=f"Generated-output backup status: {result.status}.",
            )
        except BaseException as exc:
            errors.append(f"generated backup: {_failure_message(exc)}")

        if not self._has_later_item_for_raw(item):
            try:
                self.store.add_event(queue_id=queue_id, event="local_archive_started", message="Archiving full local session to D:.")
                result = archive_recording_for_session(
                    config,
                    session_id=representative["session_id"],
                    update_registry=True,
                    progress_callback=self._sync_progress_callback(
                        queue_id=queue_id,
                        session_id=representative["session_id"],
                        event="local_archive_progress",
                        label="D: archive",
                    ),
                )
                if not result.ok:
                    errors.append(result.error or "Local recording archive failed.")
                self.store.add_event(
                    queue_id=queue_id,
                    level="warning" if not result.ok else "info",
                    event="local_archive_finished",
                    message=f"Local archive status: {result.status}.",
                )
            except BaseException as exc:
                errors.append(f"local archive: {_failure_message(exc)}")
        else:
            self.store.add_event(
                queue_id=queue_id,
                event="local_archive_deferred",
                message="Full local archive deferred until the last queued block in this acquisition session.",
            )
        if not errors and config.get("staging", {}).get("cleanup_after_verified_backups", False):
            try:
                self.store.add_event(
                    queue_id=queue_id,
                    event="local_cleanup_started",
                    message="Re-verifying Synology and D: copies before removing the staged local recording.",
                )
                result = cleanup_recording_for_session(
                    config,
                    session_id=representative["session_id"],
                    delete_local=True,
                    update_registry=True,
                )
                self.store.add_event(
                    queue_id=queue_id,
                    event="local_cleanup_finished",
                    message=(
                        f"Local recording offloaded after verifying {result.checked_files} files "
                        f"({result.checked_bytes} bytes) at both destinations."
                    ),
                )
            except BaseException as exc:
                message = f"local cleanup: {_failure_message(exc)}"
                errors.append(message)
                self.store.add_event(
                    queue_id=queue_id,
                    level="error",
                    event="local_cleanup_blocked",
                    message=message,
                )
        return errors

    def _has_later_item_for_raw(self, item: dict[str, Any]) -> bool:
        raw = str(Path(item.get("raw_folder") or "").resolve()).lower()
        for candidate in self.store.list_queue(include_finished=False):
            if candidate["queue_id"] == item["queue_id"]:
                continue
            if str(Path(candidate.get("raw_folder") or "").resolve()).lower() == raw:
                return True
        return False

    def _rows_for_item(self, item: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
        session_ids = {probe["session_id"] for probe in item.get("probes", [])}
        rows = [row for row in read_registry(config["project"]["registry_csv"]) if row.get("session_id") in session_ids]
        return sorted(rows, key=lambda row: (row.get("probe_label", ""), row.get("session_id", "")))

    def _update_registry_runtime(self, row: dict[str, str], **fields: str) -> None:
        updated = dict(row)
        updated.update(fields)
        updated["last_updated"] = _now()
        update_row(load_config(self.config_path)["project"]["registry_csv"], updated)


def _next_probe_action(row: dict[str, str]) -> str:
    if _valid_sorter_output(row):
        return "restart-qc"
    return "run-one"


def _valid_sorter_output(row: dict[str, str]) -> bool:
    folder = Path(row.get("sorter_output_folder") or "")
    if not folder.exists():
        return False
    return any(folder.rglob("spike_times.npy"))


def _restore_recoverable_interrupted_output(row: dict[str, str]) -> Path | None:
    if _valid_sorter_output(row):
        return None
    processed = Path(row.get("processed_folder") or "")
    if not processed.parent.exists():
        return None
    candidates = sorted(
        processed.parent.glob(f"{processed.name}_interrupted_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    recoverable = next(
        (candidate for candidate in candidates if any((candidate / "kilosort4").rglob("spike_times.npy"))),
        None,
    )
    if recoverable is None:
        return None
    if processed.exists() and any(processed.iterdir()):
        destination = processed.with_name(f"{processed.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(processed), str(destination))
    elif processed.exists():
        processed.rmdir()
    shutil.move(str(recoverable), str(processed))
    return recoverable


def _has_partial_outputs(row: dict[str, str]) -> bool:
    processed = Path(row.get("processed_folder") or "")
    return processed.exists() and any(processed.iterdir())


def _archive_interrupted_output(row: dict[str, str]) -> Path:
    processed = Path(row["processed_folder"])
    destination = processed.with_name(f"{processed.name}_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.move(str(processed), str(destination))
    return destination


def _disable_probe_level_transport(config: dict[str, Any]) -> None:
    config.setdefault("staging", {})["auto_stage_before_run_one"] = False
    derived = config.setdefault("backup", {}).setdefault("derived_outputs", {})
    derived["auto_after_processing"] = False
    archive = config.setdefault("backup", {}).setdefault("local_recording_archive", {})
    archive["auto_after_recording"] = False


def _snapshot_source(repo_root: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    source = repo_root / "pipeline"
    shutil.copytree(
        source,
        destination / "pipeline",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return _tree_hash(destination / "pipeline")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_command(
    *,
    source_root: Path,
    config_snapshot: Path,
    session_id: str,
    action: str,
) -> list[str]:
    module = "pipeline.restart_qc" if action == "restart-qc" else "pipeline.run_one"
    return [
        sys.executable,
        "-m",
        module,
        "--config",
        str(config_snapshot),
        "--session-id",
        session_id,
    ]


def _latest_phase_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-50:]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") in {"phase_started", "phase_completed", "phase_failed"}:
            return payload
    return None


def _last_error(stderr_log: Path, stdout_log: Path) -> str:
    for path in (stderr_log, stdout_log):
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError:
            continue
        if lines:
            return lines[-1][-1000:]
    return "Probe process failed."


def _registry_row(config: dict[str, Any], session_id: str) -> dict[str, str] | None:
    return next(
        (row for row in read_registry(config["project"]["registry_csv"]) if row.get("session_id") == session_id),
        None,
    )


def _failure_message(exc: BaseException) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persistent recording-first processing queue.")
    parser.add_argument("--config", default="pipeline/config_analysis.yaml")
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--db")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = QueueWorker(
        repo_root=args.repo_root,
        config_path=args.config,
        db_path=args.db,
    )
    worker.run(once=args.once)


if __name__ == "__main__":
    main()
