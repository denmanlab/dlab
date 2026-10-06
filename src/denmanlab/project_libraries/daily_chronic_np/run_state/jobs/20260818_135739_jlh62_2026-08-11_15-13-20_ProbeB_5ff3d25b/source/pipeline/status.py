from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .registry import read_registry


RUNNING_STATUSES = {"preprocessing", "sorting", "qc_running"}


def summarize_rows(config: dict) -> list[dict[str, str]]:
    rows = read_registry(config["project"]["registry_csv"])
    summaries: list[dict[str, str]] = []
    for row in rows:
        processed = Path(row.get("processed_folder", ""))
        report = processed / "report" / "summary.html"
        phase = _phase(row)
        summaries.append(
            {
                "session_id": row.get("session_id", ""),
                "mouse_id": row.get("mouse_id") or row.get("animal_id", ""),
                "recording_date": row.get("recording_date", ""),
                "task": row.get("task", ""),
                "probe_label": row.get("probe_label", ""),
                "status": row.get("status", ""),
                "backup_status": row.get("backup_status", ""),
                "local_stage_status": row.get("local_stage_status", ""),
                "derived_backup_status": row.get("derived_backup_status", ""),
                "derived_local_backup_status": row.get("derived_local_backup_status", ""),
                "local_recording_archive_status": row.get("local_recording_archive_status", ""),
                "nwb_backup_status": row.get("nwb_backup_status", ""),
                "behavior_session_folder": row.get("behavior_session_folder", ""),
                "phase": phase,
                "last_updated": row.get("last_updated", ""),
                "processed_folder": str(processed) if processed else "",
                "report": str(report) if report.exists() else "",
                "error_message": row.get("error_message", ""),
            }
        )
    return summaries


def has_running_rows(rows: list[dict[str, str]]) -> bool:
    return any(row.get("status") in RUNNING_STATUSES for row in rows)


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No registry rows found.")
        return
    headers = [
        "mouse_id",
        "recording_date",
        "probe_label",
        "status",
        "backup_status",
        "local_stage_status",
        "derived_backup_status",
        "derived_local_backup_status",
        "local_recording_archive_status",
        "nwb_backup_status",
        "phase",
        "last_updated",
        "report",
        "error_message",
    ]
    widths = {
        header: min(
            max(len(header), *(len(_shorten(row.get(header, ""), 72)) for row in rows)),
            72,
        )
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(_shorten(row.get(header, ""), widths[header]).ljust(widths[header]) for header in headers))


def _phase(row: dict[str, str]) -> str:
    status = row.get("status", "")
    if status == "failed":
        return "failed"
    if status == "complete":
        return "complete"
    for key in ("preprocess_status", "sort_status", "qc_status", "phy_export_status"):
        value = row.get(key, "")
        if value in {"running", "pending", "blocked"}:
            return f"{key.replace('_status', '')}:{value}"
    return status


def _shorten(value: str, width: int) -> str:
    value = str(value)
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Show SpikeInterface pipeline registry status.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--running-exit-code", action="store_true", help="Exit with code 2 if any row is currently running.")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = summarize_rows(config)
    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
    if args.running_exit_code and has_running_rows(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

