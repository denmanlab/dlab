from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


class PhaseTracker:
    def __init__(self, label: str, total: int, *, enabled: bool = True, timing_recorder: Any = None) -> None:
        self.label = label
        self.enabled = enabled
        self.timing_recorder = timing_recorder
        self._start = perf_counter()
        self._pbar = None
        if enabled and tqdm is not None:
            self._pbar = tqdm(total=total, desc=label, unit="phase", dynamic_ncols=True)
        elif enabled:
            print(f"{label}: starting ({total} phases)")

    @contextmanager
    def phase(self, name: str):
        start = perf_counter()
        self._emit_event("phase_started", name=name)
        self.message(f"START {name}")
        timing_context = (
            self.timing_recorder.phase(name, category="phase")
            if self.timing_recorder is not None
            else _null_timing_context()
        )
        try:
            with timing_context:
                yield
        except Exception:
            self._emit_event(
                "phase_failed",
                name=name,
                elapsed_seconds=round(perf_counter() - start, 3),
            )
            self.message(f"FAILED {name} after {_format_seconds(perf_counter() - start)}")
            raise
        else:
            self._emit_event(
                "phase_completed",
                name=name,
                elapsed_seconds=round(perf_counter() - start, 3),
            )
            self.message(f"DONE {name} in {_format_seconds(perf_counter() - start)}")
            if self._pbar is not None:
                self._pbar.update(1)

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        elapsed = _format_seconds(perf_counter() - self._start)
        payload = f"[{elapsed}] {self.label}: {text}"
        self._emit_event("message", message=text, elapsed=elapsed)
        if tqdm is not None:
            tqdm.write(payload)
        else:
            print(payload)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()

    def _emit_event(self, event: str, **payload: Any) -> None:
        path = str(os.environ.get("PIPELINE_EVENT_LOG") or "").strip()
        if not path:
            return
        event_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "label": self.label,
            **payload,
        }
        try:
            event_path = Path(path)
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, default=str) + "\n")
        except OSError:
            return


def progress_enabled(config: dict, cli_enabled: bool = True) -> bool:
    return bool(cli_enabled and config.get("progress", {}).get("enabled", True))


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(remainder):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{int(remainder):02d}s"


@contextmanager
def _null_timing_context():
    yield
