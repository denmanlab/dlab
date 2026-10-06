from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import spikeinterface.full as si

from .config import load_config
from .output_paths import sorting_analyzer_folder_name
from .probe_maps import load_recording_with_probe
from .progress import PhaseTracker, progress_enabled
from .registry import find_row, update_row
from .run_one import (
    _apply_common_reference,
    _apply_highpass_filter,
    _apply_motion_correction,
    _apply_phase_shift,
    _compute_analyzer,
    _export_metrics_and_summary,
    _failed_status_update,
    _failure_message,
    _patch_phy_dat_path,
    _phy_export_kwargs,
    _phy_metric_export_kwargs,
    _prune_phy_metric_tsvs,
    _write_phy_unitrefine_tsvs,
    _prepare_phy_extensions,
    _preprocessing_manifest,
    _status_update,
)
from .resume_qc import _preprocessing_figures
from .timings import QCTimingRecorder, timing_path


QC_ARCHIVE_NAMES = [
    "sorting_analyzer",
    "analyzer",
    "quality_metrics.csv",
    "template_metrics.csv",
    "unit_summary.csv",
    "summary.json",
    "manifest.json",
    "phy",
    "unitrefine",
    "report",
]


def restart_qc_for_row(
    row: dict[str, str],
    config: dict[str, Any],
    *,
    skip_phy: bool = False,
    show_progress: bool = True,
) -> dict[str, str]:
    row = dict(row)
    registry_csv = config["project"]["registry_csv"]
    processed_folder = Path(row["processed_folder"])
    sorter_folder = Path(row["sorter_output_folder"])
    analyzer_folder = processed_folder / sorting_analyzer_folder_name(config)
    row["sorting_analyzer_folder"] = str(analyzer_folder)
    channel_qc_csv = processed_folder / "channel_qc.csv"
    timing = QCTimingRecorder(
        timing_path(processed_folder),
        context={"session_id": row.get("session_id", ""), "probe_label": row.get("probe_label", ""), "run_type": "restart_qc"},
    )
    tracker = PhaseTracker(
        f"{row['session_id']}:{row['probe_label']}:restart_qc",
        total=6,
        enabled=progress_enabled(config, show_progress),
        timing_recorder=timing,
    )

    try:
        with tracker.phase("validate existing Kilosort output"):
            if not sorter_folder.exists():
                raise FileNotFoundError(f"Kilosort output folder not found: {sorter_folder}")
            if not channel_qc_csv.exists():
                raise FileNotFoundError(f"channel_qc.csv not found: {channel_qc_csv}")
            tracker.message(f"Kilosort output: {sorter_folder}")

        preprocessing_figures: dict[str, str] = {}
        with tracker.phase("archive existing QC outputs"):
            archive_folder = _archive_qc_outputs(processed_folder, tracker=tracker)
            preprocessing_figures = _restore_preprocessing_figures(processed_folder, archive_folder)
            row.update(
                _status_update(
                    "qc_running",
                    sort_status="complete",
                    qc_status="running",
                    phy_export_status="pending" if not skip_phy else "skipped",
                    error_message="",
                )
            )
            update_row(registry_csv, row)
            if archive_folder is not None:
                tracker.message(f"archived previous QC outputs: {archive_folder}")
            else:
                tracker.message("no previous QC outputs to archive")

        with tracker.phase("rebuild preprocessed recording view"):
            recording, _probe_json = load_recording_with_probe(row, config)
            recording = _apply_phase_shift(recording, config)
            recording = _apply_common_reference(recording, config)
            recording = _apply_highpass_filter(recording, config)
            recording = _apply_motion_correction(recording, processed_folder, config, force=False)
            tracker.message(
                f"recording: {recording.get_num_channels()} channels, "
                f"{recording.get_total_duration():.1f}s, {recording.get_sampling_frequency():.1f} Hz"
            )

        with tracker.phase("load Kilosort sorting and compute analyzer"):
            sorting = si.read_sorter_folder(sorter_folder)
            tracker.message(f"Kilosort units: {len(sorting.get_unit_ids())}")
            analyzer = _compute_analyzer(sorting, recording, analyzer_folder, config, overwrite=True, tracker=tracker)

        with tracker.phase("export metrics and reports"):
            outputs = _export_metrics_and_summary(
                analyzer,
                row,
                config,
                channel_qc_csv,
                _preprocessing_manifest(config),
                config.get("sorting", {}).get("params", {}),
                preprocessing_figures or _preprocessing_figures(processed_folder),
                tracker=tracker,
            )
            row.update({"quality_metrics_csv": str(outputs["quality_metrics_csv"])})

        with tracker.phase("export Phy"):
            if skip_phy or not config.get("qc", {}).get("export_phy", True):
                tracker.message("Phy export skipped")
                phy_status = "skipped"
            else:
                row.update(_status_update("qc_running", qc_status="complete", phy_export_status="running"))
                update_row(registry_csv, row)
                _prepare_phy_extensions(analyzer, config, tracker=tracker)
                si.export_to_phy(
                    analyzer,
                    output_folder=processed_folder / "phy",
                    **_phy_export_kwargs(config),
                    **_phy_metric_export_kwargs(config),
                    remove_if_exists=True,
                    verbose=True,
                )
                _patch_phy_dat_path(processed_folder / "phy", row, config, tracker=tracker)
                _write_phy_unitrefine_tsvs(processed_folder / "phy", processed_folder, config, tracker=tracker)
                _prune_phy_metric_tsvs(processed_folder / "phy", config, tracker=tracker)
                phy_status = "complete"

        row.update(
            _status_update(
                "complete",
                sort_status="complete",
                qc_status="complete",
                phy_export_status=phy_status,
                error_message="",
            )
        )
        update_row(registry_csv, row)
        return row
    except BaseException as exc:
        row.update(_failed_status_update(row, error_message=_failure_message(exc)))
        update_row(registry_csv, row)
        raise
    finally:
        tracker.close()


def _archive_qc_outputs(processed_folder: Path, *, tracker: PhaseTracker | None = None) -> Path | None:
    existing = [processed_folder / name for name in QC_ARCHIVE_NAMES if (processed_folder / name).exists()]
    if not existing:
        return None
    archive_folder = processed_folder / "qc_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_folder.mkdir(parents=True, exist_ok=False)
    for source in existing:
        destination = archive_folder / source.name
        if tracker is not None:
            tracker.message(f"archive {source.name}")
        shutil.move(str(source), str(destination))
    return archive_folder


def _restore_preprocessing_figures(processed_folder: Path, archive_folder: Path | None) -> dict[str, str]:
    if archive_folder is None:
        return _preprocessing_figures(processed_folder)
    archived_trace = archive_folder / "report" / "figures" / "pre_post_car_traces.png"
    if not archived_trace.exists():
        return {}
    restored_trace = processed_folder / "report" / "figures" / "pre_post_car_traces.png"
    restored_trace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archived_trace, restored_trace)
    return {"pre_post_car_traces_png": str(restored_trace)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive and recompute QC from an existing Kilosort output.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skip-phy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    row = find_row(config["project"]["registry_csv"], session_id=args.session_id)
    completed = restart_qc_for_row(row, config, skip_phy=args.skip_phy, show_progress=not args.no_progress)
    print(f"{completed['session_id']} | {completed.get('probe_label', '')} | {completed['status']}")


if __name__ == "__main__":
    main()
