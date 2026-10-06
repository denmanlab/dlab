from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


class QCTimingRecorder:
    def __init__(self, path: str | Path, *, context: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.context = dict(context or {})
        self.events: list[dict[str, Any]] = []
        self.created_at = _now()

    @contextmanager
    def phase(self, name: str, *, category: str = "phase", metadata: dict[str, Any] | None = None):
        event = self._start_event(name, category=category, metadata=metadata)
        try:
            yield event
        except BaseException as exc:
            self._finish_event(event, status="failed", error=_failure_message(exc))
            raise
        else:
            status = str(event.get("status") or "complete")
            self._finish_event(event, status=status)

    def record(
        self,
        name: str,
        *,
        category: str = "event",
        status: str = "complete",
        metadata: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
        error: str = "",
    ) -> None:
        started_at = _now()
        event = {
            "name": name,
            "category": category,
            "status": status,
            "started_at": started_at,
            "ended_at": started_at,
            "elapsed_seconds": float(elapsed_seconds),
            "metadata": dict(metadata or {}),
        }
        if error:
            event["error"] = error
        self.events.append(event)
        self.write()

    def write(self) -> None:
        payload = {
            "created_at": self.created_at,
            "updated_at": _now(),
            "context": self.context,
            "events": self.events,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _start_event(self, name: str, *, category: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "status": "running",
            "started_at": _now(),
            "ended_at": "",
            "elapsed_seconds": None,
            "metadata": dict(metadata or {}),
            "_perf_start": perf_counter(),
        }

    def _finish_event(self, event: dict[str, Any], *, status: str, error: str = "") -> None:
        started = float(event.pop("_perf_start", perf_counter()))
        event["status"] = status
        event["ended_at"] = _now()
        event["elapsed_seconds"] = round(perf_counter() - started, 3)
        if error:
            event["error"] = error
        self.events.append(event)
        self.write()


def timing_path(output_folder: str | Path) -> Path:
    return Path(output_folder) / "qc_timings.json"


def latest_timing_summary(output_folder: str | Path, *, limit: int = 12) -> dict[str, Any]:
    path = timing_path(output_folder)
    if not path.exists():
        return {"path": str(path), "exists": False, "events": [], "total_elapsed_seconds": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "exists": False, "events": [], "error": _failure_message(exc)}
    events = payload.get("events", [])
    total = sum(float(event.get("elapsed_seconds") or 0.0) for event in events if event.get("category") == "phase")
    return {
        "path": str(path),
        "exists": True,
        "events": events[-max(1, int(limit)) :],
        "total_elapsed_seconds": round(total, 3),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
