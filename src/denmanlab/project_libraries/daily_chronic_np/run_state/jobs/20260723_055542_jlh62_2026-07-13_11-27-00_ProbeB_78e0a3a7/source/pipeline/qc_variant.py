from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

import spikeinterface.full as si
import yaml

from .config import load_config
from .output_paths import sorting_analyzer_folder_name
from .probe_maps import load_recording_with_probe
from .progress import PhaseTracker, progress_enabled
from .registry import find_row
from .run_one import (
    DEFAULT_ANALYZER_EXTENSIONS,
    _apply_common_reference,
    _apply_highpass_filter,
    _apply_motion_correction,
    _apply_phase_shift,
    _compute_analyzer,
    _export_metrics_and_summary,
    _patch_phy_dat_path,
    _phy_export_kwargs,
    _phy_metric_export_kwargs,
    _prune_phy_metric_tsvs,
    _write_phy_unitrefine_tsvs,
    _prepare_phy_extensions,
    _preprocessing_manifest,
    _sanitize_output_suffix,
)
from .resume_qc import _preprocessing_figures
from .timings import QCTimingRecorder, timing_path


def run_qc_variant_for_row(
    row: dict[str, str],
    config: dict[str, Any],
    *,
    variant_name: str,
    overwrite: bool = False,
    skip_phy: bool = False,
    show_progress: bool = True,
) -> Path:
    row = dict(row)
    variant = _sanitize_output_suffix(variant_name)
    base_processed = Path(row["processed_folder"])
    variant_folder = base_processed / "qc_variants" / variant
    if variant_folder.exists() and not overwrite:
        raise FileExistsError(f"QC variant already exists; use --overwrite-variant to replace it: {variant_folder}")
    if variant_folder.exists() and overwrite:
        shutil.rmtree(variant_folder)
    variant_folder.mkdir(parents=True, exist_ok=False)

    variant_row = dict(row)
    variant_row["session_id"] = f"{row['session_id']}__qc_{variant}"
    variant_row["processed_folder"] = str(variant_folder)
    variant_row["sorting_analyzer_folder"] = str(variant_folder / sorting_analyzer_folder_name(config))
    variant_row["quality_metrics_csv"] = str(variant_folder / "quality_metrics.csv")

    timing = QCTimingRecorder(
        timing_path(variant_folder),
        context={
            "session_id": row.get("session_id", ""),
            "probe_label": row.get("probe_label", ""),
            "run_type": "qc_variant",
            "variant_name": variant,
        },
    )
    tracker = PhaseTracker(
        f"{row['session_id']}:{row['probe_label']}:qc_variant:{variant}",
        total=5,
        enabled=progress_enabled(config, show_progress),
        timing_recorder=timing,
    )

    try:
        with tracker.phase("write variant QC params"):
            (variant_folder / "qc_params.yaml").write_text(
                yaml.safe_dump(_variant_params_payload(config, variant), sort_keys=False),
                encoding="utf-8",
            )

        with tracker.phase("rebuild preprocessed recording view"):
            recording, _probe_json = load_recording_with_probe(row, config)
            recording = _apply_phase_shift(recording, config)
            recording = _apply_common_reference(recording, config)
            recording = _apply_highpass_filter(recording, config)
            recording = _apply_motion_correction(recording, variant_folder, config, force=overwrite)
            tracker.message(
                f"recording: {recording.get_num_channels()} channels, "
                f"{recording.get_total_duration():.1f}s, {recording.get_sampling_frequency():.1f} Hz"
            )

        with tracker.phase("load Kilosort sorting and compute analyzer"):
            sorter_folder = Path(row["sorter_output_folder"])
            if not sorter_folder.exists():
                raise FileNotFoundError(f"Kilosort output folder not found: {sorter_folder}")
            sorting = si.read_sorter_folder(sorter_folder)
            tracker.message(f"Kilosort units: {len(sorting.get_unit_ids())}")
            analyzer = _compute_analyzer(
                sorting,
                recording,
                Path(variant_row["sorting_analyzer_folder"]),
                config,
                overwrite=True,
                tracker=tracker,
            )

        with tracker.phase("export metrics and reports"):
            channel_qc_csv = _variant_channel_qc(base_processed, variant_folder)
            _export_metrics_and_summary(
                analyzer,
                variant_row,
                config,
                channel_qc_csv,
                _preprocessing_manifest(config),
                config.get("sorting", {}).get("params", {}),
                _preprocessing_figures(base_processed),
                tracker=tracker,
            )

        with tracker.phase("export Phy"):
            if skip_phy or not config.get("qc", {}).get("export_phy", True):
                tracker.message("Phy export skipped")
            else:
                _prepare_phy_extensions(analyzer, config, tracker=tracker)
                si.export_to_phy(
                    analyzer,
                    output_folder=variant_folder / "phy",
                    **_phy_export_kwargs(config),
                    **_phy_metric_export_kwargs(config),
                    remove_if_exists=overwrite,
                    verbose=True,
                )
                _patch_phy_dat_path(variant_folder / "phy", row, config, tracker=tracker)
                _write_phy_unitrefine_tsvs(variant_folder / "phy", variant_folder, config, tracker=tracker)
                _prune_phy_metric_tsvs(variant_folder / "phy", config, tracker=tracker)

        tracker.message(f"QC variant complete: {variant_folder}")
        return variant_folder
    finally:
        tracker.close()


def config_with_variant_overrides(
    config: dict[str, Any],
    *,
    n_jobs: int | None = None,
    metric_names: list[str] | None = None,
    include_spike_locations: bool | None = None,
    include_principal_components: bool | None = None,
    phy_compute_pc_features: bool | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if n_jobs is not None:
        updated.setdefault("jobs", {})["n_jobs"] = int(n_jobs)
    if metric_names is not None:
        updated.setdefault("qc", {}).setdefault("quality_metrics", {})["metric_names"] = metric_names
    if phy_compute_pc_features is not None:
        updated.setdefault("qc", {})["phy_compute_pc_features"] = bool(phy_compute_pc_features)
    for name, enabled in [
        ("spike_locations", include_spike_locations),
        ("principal_components", include_principal_components),
    ]:
        if enabled is not None:
            _set_extension_enabled(updated, name, bool(enabled))
    return updated


def _set_extension_enabled(config: dict[str, Any], extension_name: str, enabled: bool) -> None:
    qc_config = config.setdefault("qc", {})
    if not qc_config.get("analyzer_extensions"):
        qc_config["analyzer_extensions"] = list(DEFAULT_ANALYZER_EXTENSIONS)
    extensions = qc_config["analyzer_extensions"]
    for index, item in enumerate(list(extensions)):
        if item == extension_name:
            extensions[index] = {"name": extension_name, "enabled": enabled}
            return
        if isinstance(item, dict) and item.get("name") == extension_name:
            item["enabled"] = enabled
            return
    extensions.append({"name": extension_name, "enabled": enabled})


def _variant_params_payload(config: dict[str, Any], variant_name: str) -> dict[str, Any]:
    return {
        "variant_name": variant_name,
        "qc": config.get("qc", {}),
        "jobs": config.get("jobs", {}),
    }


def _variant_channel_qc(base_processed: Path, variant_folder: Path) -> Path:
    source = base_processed / "channel_qc.csv"
    destination = variant_folder / "channel_qc.csv"
    if not source.exists():
        raise FileNotFoundError(f"channel_qc.csv not found: {source}")
    shutil.copy2(source, destination)
    return destination


def _metric_names_from_cli(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        for name in str(value).split(","):
            name = name.strip()
            if name:
                names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QC-only variant from an existing Kilosort output.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--overwrite-variant", action="store_true")
    parser.add_argument("--skip-phy", action="store_true")
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--metric-name", action="append", default=[])
    parser.add_argument("--include-spike-locations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-principal-components", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phy-compute-pc-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = config_with_variant_overrides(
        load_config(args.config),
        n_jobs=args.n_jobs,
        metric_names=_metric_names_from_cli(args.metric_name),
        include_spike_locations=args.include_spike_locations,
        include_principal_components=args.include_principal_components,
        phy_compute_pc_features=args.phy_compute_pc_features,
    )
    row = find_row(config["project"]["registry_csv"], session_id=args.session_id)
    folder = run_qc_variant_for_row(
        row,
        config,
        variant_name=args.variant_name,
        overwrite=args.overwrite_variant,
        skip_phy=args.skip_phy,
        show_progress=not args.no_progress,
    )
    print(f"{row['session_id']} | {row.get('probe_label', '')} | qc variant | {folder}")


if __name__ == "__main__":
    main()
