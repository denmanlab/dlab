from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .config import load_config
from .derived_backup import backup_derived_outputs_for_recording_session
from .discover import discover
from .local_recording_archive import archive_recording_for_session
from .local_cleanup import cleanup_recording_for_session
from .progress import PhaseTracker, progress_enabled
from .registry import read_registry, upsert_rows
from .recording_identity import recording_identity_for_row
from .run_one import _sanitize_output_suffix, run_registry_row
from .stage_recording import stage_recording_for_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and run registered pending probe rows.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--output-suffix",
        help="Write each run to a sibling output folder named spikeinterface_output_<suffix> and separate registry rows.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="With --output-suffix, rerun completed base rows into new output variant folders.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    auto_backup = bool(config.get("backup", {}).get("derived_outputs", {}).get("auto_after_processing", False))
    auto_archive = bool(config.get("backup", {}).get("local_recording_archive", {}).get("auto_after_recording", True))
    probe_config = copy.deepcopy(config)
    probe_config.setdefault("backup", {}).setdefault("derived_outputs", {})["auto_after_processing"] = False
    probe_config.setdefault("backup", {}).setdefault("local_recording_archive", {})["auto_after_recording"] = False
    tracker = PhaseTracker("run_pending", total=3, enabled=progress_enabled(config, not args.no_progress))
    try:
        with tracker.phase("discover and refresh registry"):
            rows = discover(config, dry_run=False)
            upsert_rows(config["project"]["registry_csv"], rows, preserve_completed=True)
            rows = read_registry(config["project"]["registry_csv"])
            runnable = [row for row in rows if row.get("status") in {"registered", "failed"}]
            if args.rerun_complete and args.output_suffix:
                suffix_marker = f"_{_sanitize_output_suffix(args.output_suffix)}"
                complete_base_rows = [
                    row
                    for row in rows
                    if row.get("status") == "complete"
                    and suffix_marker not in row.get("session_id", "")
                    and _is_completed_base_row(row, config)
                ]
                runnable.extend(complete_base_rows)
            tracker.message(f"runnable rows: {len(runnable)}")
        grouped = _group_rows_by_recording(runnable)
        with tracker.phase("stage/run/archive recordings"):
            total_probes = len(runnable)
            probe_index = 0
            for recording_index, group in enumerate(grouped, start=1):
                representative = group[0]
                recording_key = representative.get("server_raw_folder") or representative.get("raw_folder") or representative.get("session_id", "")
                tracker.message(f"recording {recording_index}/{len(grouped)}: {recording_key}")
                if config.get("staging", {}).get("enabled", False) and config.get("staging", {}).get("auto_stage_before_run_pending", True):
                    tracker.message(f"stage recording: {recording_key}")
                    stage_recording_for_session(config, session_id=representative["session_id"], update_registry=True)
                    refreshed = {row.get("session_id", ""): row for row in read_registry(config["project"]["registry_csv"])}
                    group = [refreshed.get(row.get("session_id", ""), row) for row in group]
                else:
                    tracker.message("staging disabled")

                completed_any = False
                for row in group:
                    probe_index += 1
                    tracker.message(f"probe {probe_index}/{total_probes}: {row['session_id']} {row['probe_label']}")
                    completed = run_registry_row(
                        row,
                        probe_config,
                        force=args.force,
                        duration_seconds=args.duration_seconds,
                        output_suffix=args.output_suffix,
                        show_progress=not args.no_progress,
                    )
                    print(f"{completed['session_id']} | {completed['probe_label']} | {completed['status']}")
                    completed_any = completed_any or completed.get("status") == "complete"
                if completed_any and auto_backup:
                    tracker.message(f"backup generated outputs once for recording: {recording_key}")
                    backup_derived_outputs_for_recording_session(
                        config,
                        session_id=representative["session_id"],
                        update_registry=True,
                    )

                archive_config = config.get("backup", {}).get("local_recording_archive", {})
                if archive_config.get("enabled", False) and auto_archive:
                    tracker.message(f"archive full local recording to D: {recording_key}")
                    result = archive_recording_for_session(
                        config,
                        session_id=representative["session_id"],
                        update_registry=True,
                    )
                    print(
                        f"local recording archive | {representative['session_id']} | {result.status} | "
                        f"copied={result.copied_files} skipped={result.skipped_files}"
                    )
                if config.get("staging", {}).get("cleanup_after_verified_backups", False):
                    result = cleanup_recording_for_session(
                        config,
                        session_id=representative["session_id"],
                        delete_local=True,
                        update_registry=True,
                    )
                    print(
                        f"local cleanup | {representative['session_id']} | {result.status} | "
                        f"verified_files={result.checked_files}"
                    )
    finally:
        tracker.close()


def _is_completed_base_row(row: dict[str, str], config: dict) -> bool:
    """Return True only for canonical full-output rows, not smoke/variant rows."""
    notes = str(row.get("notes", "")).lower()
    session_id = str(row.get("session_id", "")).lower()
    if "output variant:" in notes or "duration-limited smoke run:" in notes or "_smoke_" in session_id:
        return False

    if config.get("project", {}).get("output_layout") == "recording_local_probe_folder":
        output_name = config.get("project", {}).get("recording_local_output_name", "spikeinterface_output")
        return Path(row.get("processed_folder", "")).name == output_name

    return True


def _group_rows_by_recording(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = recording_identity_for_row(row)[0]
        grouped.setdefault(key, []).append(row)
    return list(grouped.values())


if __name__ == "__main__":
    main()


