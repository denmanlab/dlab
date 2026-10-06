from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_QUEUE_STATUSES = {"queued", "running", "paused"}
FINAL_PROBE_STATUSES = {"complete", "failed", "skipped", "interrupted"}


class QueueStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def enqueue(
        self,
        *,
        recording_id: str,
        session_ids: list[str],
        profile_name: str,
        raw_folder: str,
        allow_short: bool = False,
    ) -> dict[str, Any]:
        if not session_ids:
            raise ValueError("A queue item requires at least one probe session_id.")
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM queue_items
                WHERE recording_id = ? AND status IN ('queued', 'running', 'paused')
                ORDER BY created_at DESC LIMIT 1
                """,
                (recording_id,),
            ).fetchone()
            if existing is not None:
                return self._item_payload(connection, existing)
            position = int(connection.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM queue_items").fetchone()[0])
            queue_id = f"queue_{uuid.uuid4().hex[:12]}"
            now = _now()
            connection.execute(
                """
                INSERT INTO queue_items
                    (queue_id, recording_id, raw_folder, profile_name, position, status,
                     allow_short, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (queue_id, recording_id, raw_folder, profile_name, position, int(allow_short), now, now),
            )
            for session_id in session_ids:
                connection.execute(
                    """
                    INSERT INTO queue_probes
                        (queue_id, session_id, probe_label, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, ?)
                    """,
                    (queue_id, session_id, _probe_label(session_id), now, now),
                )
            row = connection.execute("SELECT * FROM queue_items WHERE queue_id = ?", (queue_id,)).fetchone()
            return self._item_payload(connection, row)

    def list_queue(self, *, include_finished: bool = True) -> list[dict[str, Any]]:
        with self.connect() as connection:
            query = "SELECT * FROM queue_items"
            params: tuple[Any, ...] = ()
            if not include_finished:
                query += " WHERE status IN ('queued', 'running', 'paused')"
            query += " ORDER BY position, created_at"
            return [self._item_payload(connection, row) for row in connection.execute(query, params).fetchall()]

    def get_item(self, queue_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM queue_items WHERE queue_id = ?", (queue_id,)).fetchone()
            if row is None:
                raise KeyError(queue_id)
            return self._item_payload(connection, row)

    def next_item(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM queue_items
                WHERE status IN ('queued', 'running')
                ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, position, created_at
                LIMIT 1
                """
            ).fetchone()
            return self._item_payload(connection, row) if row is not None else None

    def update_item(self, queue_id: str, **fields: Any) -> None:
        self._update("queue_items", "queue_id", queue_id, fields)

    def update_probe(self, queue_id: str, session_id: str, **fields: Any) -> None:
        fields = {**fields, "updated_at": _now()}
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [_db_value(value) for value in fields.values()]
        with self.connect() as connection:
            connection.execute(
                f"UPDATE queue_probes SET {assignments} WHERE queue_id = ? AND session_id = ?",
                (*values, queue_id, session_id),
            )

    def add_job(
        self,
        *,
        job_id: str,
        queue_id: str,
        session_id: str,
        kind: str,
        command: list[str],
        stdout_log: str,
        stderr_log: str,
        event_log: str,
        pid: int,
        profile_name: str,
        profile_version: int,
        config_hash: str,
        source_hash: str,
    ) -> None:
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs
                    (job_id, queue_id, session_id, kind, command_json, status, pid,
                     stdout_log, stderr_log, event_log, profile_name, profile_version,
                     config_hash, source_hash, created_at, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    queue_id,
                    session_id,
                    kind,
                    json.dumps(command),
                    int(pid),
                    stdout_log,
                    stderr_log,
                    event_log,
                    profile_name,
                    int(profile_version),
                    config_hash,
                    source_hash,
                    now,
                    now,
                    now,
                ),
            )

    def finish_job(self, job_id: str, *, returncode: int, error: str = "") -> None:
        status = "complete" if int(returncode) == 0 else "failed"
        self._update(
            "jobs",
            "job_id",
            job_id,
            {
                "status": status,
                "returncode": int(returncode),
                "error": error,
                "finished_at": _now(),
                "updated_at": _now(),
            },
        )

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [_row_payload(row) for row in rows]

    def add_event(
        self,
        *,
        queue_id: str = "",
        session_id: str = "",
        level: str = "info",
        event: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events (queue_id, session_id, level, event, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (queue_id, session_id, level, event, message, json.dumps(payload or {}, default=str), _now()),
            )

    def list_events(self, *, queue_id: str = "", session_id: str = "", limit: int = 200) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if queue_id:
            where.append("queue_id = ?")
            params.append(queue_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        query = "SELECT * FROM events"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY event_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
            return [_row_payload(row) for row in reversed(rows)]

    def set_meta(self, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, json.dumps(value, default=str), _now()),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return row["value"]

    def request_pause(self) -> None:
        self.set_meta("pause_after_probe", True)

    def clear_pause(self) -> None:
        self.set_meta("pause_after_probe", False)
        with self.connect() as connection:
            connection.execute(
                "UPDATE queue_items SET status = 'queued', updated_at = ? WHERE status = 'paused'",
                (_now(),),
            )

    def pause_requested(self) -> bool:
        return bool(self.get_meta("pause_after_probe", False))

    def reconcile_stale_worker(self, *, error: str) -> dict[str, int]:
        now = _now()
        with self.connect() as connection:
            jobs = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', returncode = -1, error = ?, finished_at = ?, updated_at = ?
                WHERE status = 'running'
                """,
                (error, now, now),
            ).rowcount
            probes = connection.execute(
                """
                UPDATE queue_probes
                SET status = 'queued', current_step = 'interrupted', error = ?, updated_at = ?
                WHERE status = 'running'
                """,
                (error, now),
            ).rowcount
            items = connection.execute(
                """
                UPDATE queue_items
                SET status = 'paused', current_probe = '', error = ?, updated_at = ?
                WHERE status = 'running'
                """,
                (error, now),
            ).rowcount
        self.set_meta("worker_status", "failed")
        self.set_meta("active_queue_id", "")
        self.set_meta("active_session_id", "")
        return {"jobs": jobs, "probes": probes, "items": items}

    def worker_payload(self) -> dict[str, Any]:
        return {
            "status": self.get_meta("worker_status", "stopped"),
            "pid": self.get_meta("worker_pid", None),
            "started_at": self.get_meta("worker_started_at", ""),
            "heartbeat_at": self.get_meta("worker_heartbeat_at", ""),
            "pause_after_probe": self.pause_requested(),
            "active_queue_id": self.get_meta("active_queue_id", ""),
            "active_session_id": self.get_meta("active_session_id", ""),
        }

    def _update(self, table: str, key_column: str, key_value: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        if table == "queue_items" and "updated_at" not in fields:
            fields = {**fields, "updated_at": _now()}
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [_db_value(value) for value in fields.values()]
        with self.connect() as connection:
            connection.execute(
                f"UPDATE {table} SET {assignments} WHERE {key_column} = ?",
                (*values, key_value),
            )

    def _item_payload(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        item = _row_payload(row)
        probes = connection.execute(
            "SELECT * FROM queue_probes WHERE queue_id = ? ORDER BY probe_label, session_id",
            (row["queue_id"],),
        ).fetchall()
        item["probes"] = [_row_payload(probe) for probe in probes]
        return item

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS queue_items (
                    queue_id TEXT PRIMARY KEY,
                    recording_id TEXT NOT NULL,
                    raw_folder TEXT NOT NULL DEFAULT '',
                    profile_name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    allow_short INTEGER NOT NULL DEFAULT 0,
                    current_probe TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS queue_probes (
                    queue_probe_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id TEXT NOT NULL REFERENCES queue_items(queue_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    probe_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step TEXT NOT NULL DEFAULT '',
                    profile_name TEXT NOT NULL DEFAULT '',
                    profile_version INTEGER,
                    config_hash TEXT NOT NULL DEFAULT '',
                    source_hash TEXT NOT NULL DEFAULT '',
                    job_id TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(queue_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    queue_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    returncode INTEGER,
                    stdout_log TEXT NOT NULL,
                    stderr_log TEXT NOT NULL,
                    event_log TEXT NOT NULL,
                    profile_name TEXT NOT NULL DEFAULT '',
                    profile_version INTEGER,
                    config_hash TEXT NOT NULL DEFAULT '',
                    source_hash TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(status, position);
                CREATE INDEX IF NOT EXISTS idx_probe_status ON queue_probes(status);
                CREATE INDEX IF NOT EXISTS idx_event_queue ON events(queue_id, event_id);
                """
            )


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    for key in ("allow_short",):
        if key in payload:
            payload[key] = bool(payload[key])
    for key in ("command_json", "payload_json"):
        if key in payload:
            try:
                default_json = "[]" if key == "command_json" else "{}"
                payload[key.removesuffix("_json")] = json.loads(payload[key] or default_json)
            except json.JSONDecodeError:
                payload[key.removesuffix("_json")] = payload[key]
    return payload


def _db_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def _probe_label(session_id: str) -> str:
    if session_id.endswith("_ProbeA"):
        return "ProbeA"
    if session_id.endswith("_ProbeB"):
        return "ProbeB"
    return session_id.rsplit("_", 1)[-1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
