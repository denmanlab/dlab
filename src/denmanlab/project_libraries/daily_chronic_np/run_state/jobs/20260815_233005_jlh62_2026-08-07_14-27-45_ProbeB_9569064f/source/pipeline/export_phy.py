from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import spikeinterface.full as si

from .config import load_config
from .progress import PhaseTracker, progress_enabled
from .registry import find_row, update_row
from .run_one import (
    _attach_automated_label_properties,
    _failed_status_update,
    _failure_message,
    _patch_phy_dat_path,
    _phy_export_kwargs,
    _phy_metric_export_kwargs,
    _prune_phy_metric_tsvs,
    _write_phy_unitrefine_tsvs,
    _prepare_phy_extensions,
    _status_update,
)
from .timings import QCTimingRecorder, timing_path


def export_phy_for_row(
    row: dict[str, str],
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, str]:
    row = dict(row)
    registry_csv = config["project"]["registry_csv"]
    processed_folder = Path(row["processed_folder"])
    analyzer_folder = Path(row["sorting_analyzer_folder"])
    phy_folder = processed_folder / "phy"
    timing = QCTimingRecorder(
        timing_path(processed_folder),
        context={"session_id": row.get("session_id", ""), "probe_label": row.get("probe_label", ""), "run_type": "export_phy"},
    )
    tracker = PhaseTracker(
        f"{row['session_id']}:{row['probe_label']}:export_phy",
        total=2,
        enabled=progress_enabled(config, show_progress),
        timing_recorder=timing,
    )

    try:
        with tracker.phase("load existing SortingAnalyzer"):
            if not analyzer_folder.exists():
                raise FileNotFoundError(f"SortingAnalyzer folder not found: {analyzer_folder}")
            if phy_folder.exists() and not overwrite:
                raise FileExistsError(f"Phy folder already exists; use --overwrite-phy to replace it: {phy_folder}")
            analyzer = si.load_sorting_analyzer(analyzer_folder)
            labels_csv = _latest_unitrefine_labels(processed_folder)
            if labels_csv:
                _attach_automated_label_properties(
                    analyzer,
                    {"unit_labels_csv": str(labels_csv)},
                    config,
                )
                tracker.message(f"attached UnitRefine labels: {labels_csv}")
            row.update(
                _status_update(
                    "qc_running",
                    sort_status="complete",
                    qc_status=row.get("qc_status", "complete") or "complete",
                    phy_export_status="running",
                    error_message="",
                )
            )
            update_row(registry_csv, row)

        with tracker.phase("export Phy"):
            _prepare_phy_extensions(analyzer, config, tracker=tracker)
            si.export_to_phy(
                analyzer,
                output_folder=phy_folder,
                **_phy_export_kwargs(config),
                **_phy_metric_export_kwargs(config),
                remove_if_exists=overwrite,
                verbose=True,
            )
            _patch_phy_dat_path(phy_folder, row, config, tracker=tracker)
            _write_phy_unitrefine_tsvs(phy_folder, processed_folder, config, tracker=tracker)
            _prune_phy_metric_tsvs(phy_folder, config, tracker=tracker)
            tracker.message(f"Phy exported: {phy_folder}")

        row.update(
            _status_update(
                "complete",
                sort_status="complete",
                qc_status=row.get("qc_status", "complete") or "complete",
                phy_export_status="complete",
                error_message="",
            )
        )
        update_row(registry_csv, row)
        return row
    except BaseException as exc:
        row.update(_failed_status_update(row, error_message=_failure_message(exc)))
        row["phy_export_status"] = "failed"
        update_row(registry_csv, row)
        raise
    finally:
        tracker.close()


def _latest_unitrefine_labels(processed_folder: Path) -> Path | None:
    root = processed_folder / "unitrefine"
    if not root.exists():
        return None
    labels = sorted(root.glob("*/unit_labels.csv"), key=lambda path: path.stat().st_mtime)
    return labels[-1] if labels else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phy from an existing SortingAnalyzer.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--overwrite-phy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    row = find_row(config["project"]["registry_csv"], session_id=args.session_id)
    completed = export_phy_for_row(
        row,
        config,
        overwrite=args.overwrite_phy,
        show_progress=not args.no_progress,
    )
    print(f"{completed['session_id']} | {completed.get('probe_label', '')} | {completed['phy_export_status']}")


if __name__ == "__main__":
    main()
