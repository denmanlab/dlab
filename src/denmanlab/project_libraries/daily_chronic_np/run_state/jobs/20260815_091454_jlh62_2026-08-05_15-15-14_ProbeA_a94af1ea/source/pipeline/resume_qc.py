from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import spikeinterface.full as si

from .config import load_config
from .progress import PhaseTracker, progress_enabled
from .registry import find_row, update_row
from .run_one import (
    ANALYZER_EXTENSIONS,
    _compute_analyzer_extensions,
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
from .timings import QCTimingRecorder, timing_path


def resume_qc_for_row(
    row: dict[str, str],
    config: dict[str, Any],
    *,
    recompute_extensions: list[str] | None = None,
    skip_phy: bool = False,
    overwrite_phy: bool = False,
    show_progress: bool = True,
) -> dict[str, str]:
    row = dict(row)
    registry_csv = config["project"]["registry_csv"]
    processed_folder = Path(row["processed_folder"])
    analyzer_folder = Path(row["sorting_analyzer_folder"])
    channel_qc_csv = processed_folder / "channel_qc.csv"
    timing = QCTimingRecorder(
        timing_path(processed_folder),
        context={"session_id": row.get("session_id", ""), "probe_label": row.get("probe_label", ""), "run_type": "resume_qc"},
    )
    tracker = PhaseTracker(
        f"{row['session_id']}:{row['probe_label']}:resume_qc",
        total=4,
        enabled=progress_enabled(config, show_progress),
        timing_recorder=timing,
    )

    try:
        with tracker.phase("load existing SortingAnalyzer"):
            if not analyzer_folder.exists():
                raise FileNotFoundError(f"SortingAnalyzer folder not found: {analyzer_folder}")
            if not channel_qc_csv.exists():
                raise FileNotFoundError(f"channel_qc.csv not found: {channel_qc_csv}")
            analyzer = si.load_sorting_analyzer(analyzer_folder)
            tracker.message(f"loaded analyzer with saved extensions: {', '.join(_saved_extensions(analyzer)) or 'none'}")
            row.update(
                _status_update(
                    "qc_running",
                    sort_status="complete",
                    qc_status="running",
                    phy_export_status=row.get("phy_export_status", "pending") or "pending",
                    error_message="",
                )
            )
            update_row(registry_csv, row)

        with tracker.phase("resume SortingAnalyzer extensions"):
            _validate_extension_names(recompute_extensions or [])
            _compute_analyzer_extensions(
                analyzer,
                config,
                tracker=tracker,
                skip_existing=True,
                recompute_extensions=recompute_extensions or [],
            )

        with tracker.phase("export metrics and static report"):
            outputs = _export_metrics_and_summary(
                analyzer,
                row,
                config,
                channel_qc_csv,
                _preprocessing_manifest(config),
                config.get("sorting", {}).get("params", {}),
                _preprocessing_figures(processed_folder),
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
                    remove_if_exists=overwrite_phy,
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


def _saved_extensions(analyzer) -> list[str]:
    try:
        return sorted(analyzer.get_saved_extension_names())
    except Exception:
        return []


def _validate_extension_names(extension_names: list[str]) -> None:
    unknown = sorted(set(extension_names) - set(ANALYZER_EXTENSIONS))
    if unknown:
        raise ValueError(f"Unknown analyzer extension(s): {', '.join(unknown)}")


def _preprocessing_figures(processed_folder: Path) -> dict[str, str]:
    figure = processed_folder / "report" / "figures" / "pre_post_car_traces.png"
    return {"pre_post_car_traces_png": str(figure)} if figure.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume QC/report export from an existing SortingAnalyzer.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--recompute-extension",
        action="append",
        default=[],
        help="Delete and recompute a saved analyzer extension before continuing. Can be used multiple times.",
    )
    parser.add_argument("--skip-phy", action="store_true")
    parser.add_argument("--overwrite-phy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    row = find_row(config["project"]["registry_csv"], session_id=args.session_id)
    completed = resume_qc_for_row(
        row,
        config,
        recompute_extensions=args.recompute_extension,
        skip_phy=args.skip_phy,
        overwrite_phy=args.overwrite_phy,
        show_progress=not args.no_progress,
    )
    print(f"{completed['session_id']} | {completed.get('probe_label', '')} | {completed['status']}")


if __name__ == "__main__":
    main()
