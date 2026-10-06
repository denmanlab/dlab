from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kilosort
import pandas as pd
import spikeinterface
import spikeinterface.full as si

from .backup import require_backup_verified
from .config import load_config
from .derived_backup import backup_derived_outputs_for_recording_session
from .hashing import hash_jsonable
from .local_recording_archive import archive_recording_for_session
from .output_paths import output_paths_for_processed_folder, suffixed_processed_folder
from .probe_maps import channel_ids_to_indices, compute_probe_hashes, load_recording_with_probe
from .progress import PhaseTracker, progress_enabled
from .registry import find_row, update_row
from .reports import write_preprocessing_trace_plots, write_summary_report
from .stage_recording import stage_recording_for_session
from .timings import QCTimingRecorder, timing_path
from .unitrefine import run_unitrefine_for_analyzer


DEFAULT_ANALYZER_EXTENSIONS = [
    "random_spikes",
    "waveforms",
    "templates",
    "noise_levels",
    "spike_amplitudes",
    "unit_locations",
    "spike_locations",
    "principal_components",
    "correlograms",
    "isi_histograms",
    "quality_metrics",
    "template_metrics",
]
ANALYZER_EXTENSIONS = DEFAULT_ANALYZER_EXTENSIONS
JOB_KWARG_KEYS = {"n_jobs", "chunk_duration", "progress_bar", "mp_context", "max_threads_per_process"}
QUALITY_METRIC_ALIASES = {
    "isi_violations_ratio": "isi_violation",
    "isi_violations_count": "isi_violation",
    "rp_contamination": "rp_violation",
    "rp_violations": "rp_violation",
}
DEFAULT_UNITREFINE_QUALITY_METRICS = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "synchrony",
    "firing_range",
    "amplitude_cv",
    "amplitude_cutoff",
    "noise_cutoff",
    "amplitude_median",
    "drift",
    "sd_ratio",
    "mahalanobis",
    "d_prime",
    "nearest_neighbor",
    "silhouette",
    "nn_advanced",
]


def run_registry_row(
    row: dict[str, str],
    config: dict[str, Any],
    *,
    force: bool = False,
    duration_seconds: float | None = None,
    output_suffix: str | None = None,
    show_progress: bool = True,
) -> dict[str, str]:
    registry_csv = config["project"]["registry_csv"]
    row = dict(row)
    suffix_parts: list[str] = []
    base = Path(row["processed_folder"])
    if output_suffix:
        suffix_parts.append(_sanitize_output_suffix(output_suffix))
    if duration_seconds is not None:
        suffix_parts.append(f"smoke_{int(duration_seconds)}s")
    if suffix_parts:
        suffix = "_".join(part for part in suffix_parts if part)
        processed_variant = suffixed_processed_folder(base, suffix)
        row["session_id"] = f"{row['session_id']}_{suffix}"
        row["notes"] = _append_note(row.get("notes", ""), f"output variant: {suffix}")
        if duration_seconds is not None:
            row["notes"] = _append_note(row.get("notes", ""), f"duration-limited smoke run: {duration_seconds}s")
        row.update({key: str(value) for key, value in output_paths_for_processed_folder(processed_variant).items()})
    processed_folder = Path(row["processed_folder"])
    manifest_path = processed_folder / "manifest.json"

    if row.get("status") == "complete" and manifest_path.exists() and not force:
        return row
    if processed_folder.exists() and not force and _has_processing_outputs(processed_folder):
        raise RuntimeError(f"Processed outputs already exist; use --force only if you intend to rerun: {processed_folder}")

    timing = QCTimingRecorder(
        timing_path(processed_folder),
        context={"session_id": row.get("session_id", ""), "probe_label": row.get("probe_label", ""), "run_type": "base"},
    )
    tracker = PhaseTracker(
        f"{row['session_id']}:{row['probe_label']}",
        total=16,
        enabled=progress_enabled(config, show_progress),
        timing_recorder=timing,
    )
    try:
        with tracker.phase("verify raw backup"):
            backup_fields = require_backup_verified(row, config)
            row.update(backup_fields)
            if backup_fields.get("backup_status") == "verified":
                tracker.message(f"raw backup verified: {backup_fields['backup_folder']}")
            update_row(registry_csv, row)

        with tracker.phase("initialize output and registry"):
            processed_folder.mkdir(parents=True, exist_ok=True)
            row.update(_status_update("preprocessing", preprocess_status="running", error_message=""))
            update_row(registry_csv, row)

        with tracker.phase("load Open Ephys recording and probe map"):
            recording, probe_json = load_recording_with_probe(row, config)
            if duration_seconds is not None:
                frames = min(recording.get_num_frames(), int(float(duration_seconds) * recording.get_sampling_frequency()))
                recording = recording.frame_slice(start_frame=0, end_frame=frames)
            tracker.message(
                f"recording: {recording.get_num_channels()} channels, "
                f"{recording.get_total_duration():.1f}s, {recording.get_sampling_frequency():.1f} Hz"
            )

        with tracker.phase("validate geometry and compute hashes"):
            probe_hashes = compute_probe_hashes(recording, probe_json)
            row.update(probe_hashes)
            preprocessing_config = _preprocessing_manifest(config)
            row["preprocessing_config_hash"] = hash_jsonable(preprocessing_config)

        with tracker.phase("apply Neuropixels phase shift"):
            recording = _apply_phase_shift(recording, config)
            if config.get("preprocessing", {}).get("phase_shift", {}).get("enabled", True):
                tracker.message("phase_shift applied")
            else:
                tracker.message("phase_shift disabled")

        with tracker.phase("apply SpikeInterface CAR and plot traces"):
            pre_car_recording = recording
            recording = _apply_common_reference(recording, config)
            preprocessing_figures = _write_preprocessing_figures(pre_car_recording, recording, processed_folder, config)
            car_config = config.get("preprocessing", {}).get("car", {})
            if preprocessing_figures:
                tracker.message(
                    f"CAR operator={car_config.get('operator', 'median')}; "
                    f"pre/post CAR plot: {preprocessing_figures['pre_post_car_traces_png']}"
                )
            else:
                tracker.message(f"CAR operator={car_config.get('operator', 'median')}; pre/post CAR plot skipped")

        with tracker.phase("apply high-pass filter"):
            recording = _apply_highpass_filter(recording, config)
            highpass_config = config.get("preprocessing", {}).get("highpass_filter", {})
            if highpass_config.get("enabled", True):
                tracker.message(f"highpass_filter applied: freq_min={highpass_config.get('freq_min', 300)} Hz")
            else:
                tracker.message("highpass_filter disabled")

        with tracker.phase("detect bad channels"):
            bad_channel_ids, channel_labels = _detect_bad_channels(recording, config)
            excluded_bad_channel_ids = _bad_channels_for_sorting(recording, bad_channel_ids, channel_labels, config)
            channel_qc_csv = processed_folder / "channel_qc.csv"
            _write_channel_qc(recording, bad_channel_ids, channel_labels, excluded_bad_channel_ids, channel_qc_csv)
            tracker.message(
                f"bad channels detected: {len(bad_channel_ids)}; "
                f"excluded from sorting: {len(excluded_bad_channel_ids)}"
            )

        with tracker.phase("optional SpikeInterface motion correction"):
            recording = _apply_motion_correction(recording, processed_folder, config, force=force)
            if _motion_correction_enabled(config):
                tracker.message("SpikeInterface motion correction applied before sorting")
            else:
                tracker.message("using Kilosort4 drift correction")

        with tracker.phase("prepare Kilosort4 parameters"):
            sorter_params = _kilosort4_params(config, recording, excluded_bad_channel_ids)
            sorter_output = Path(row["sorter_output_folder"])
            tracker.message(
                "Kilosort4 preprocessing: "
                f"do_CAR={sorter_params['do_CAR']}, "
                f"do_correction={sorter_params['do_correction']}, "
                f"bad_channels={len(sorter_params['bad_channels'])}"
            )
            row.update(
                {
                    "status": "sorting",
                    "preprocess_status": "complete",
                    "sort_status": "running",
                    "sorter_version": getattr(kilosort, "__version__", ""),
                    "spikeinterface_version": spikeinterface.__version__,
                    "kilosort_version": getattr(kilosort, "__version__", ""),
                    "last_updated": _now(),
                }
            )
            update_row(registry_csv, row)

        with tracker.phase("run Kilosort4"):
            sorting = si.run_sorter(
                sorter_name=config.get("sorting", {}).get("sorter_name", "kilosort4"),
                recording=recording,
                folder=sorter_output,
                remove_existing_folder=force or bool(config.get("sorting", {}).get("remove_existing_folder", False)),
                docker_image=config.get("sorting", {}).get("docker_image", False),
                singularity_image=config.get("sorting", {}).get("singularity_image", False),
                verbose=True,
                **sorter_params,
            )
            tracker.message(f"Kilosort4 units: {len(sorting.get_unit_ids())}")

        with tracker.phase("compute SortingAnalyzer extensions"):
            row.update(_status_update("qc_running", sort_status="complete", qc_status="running"))
            update_row(registry_csv, row)
            analyzer = _compute_analyzer(
                sorting,
                recording,
                Path(row["sorting_analyzer_folder"]),
                config,
                overwrite=force,
                tracker=tracker,
            )

        with tracker.phase("export metrics and static report"):
            outputs = _export_metrics_and_summary(
                analyzer,
                row,
                config,
                channel_qc_csv,
                preprocessing_config,
                sorter_params,
                preprocessing_figures,
                tracker=tracker,
            )

        with tracker.phase("export Phy"):
            if config.get("qc", {}).get("export_phy", True):
                _prepare_phy_extensions(analyzer, config, tracker=tracker)
                si.export_to_phy(
                    analyzer,
                    output_folder=processed_folder / "phy",
                    **_phy_export_kwargs(config),
                    **_phy_metric_export_kwargs(config),
                    remove_if_exists=force,
                    verbose=True,
                )
                _patch_phy_dat_path(processed_folder / "phy", row, config, tracker=tracker)
                _write_phy_unitrefine_tsvs(processed_folder / "phy", processed_folder, config, tracker=tracker)
                _prune_phy_metric_tsvs(processed_folder / "phy", config, tracker=tracker)
            else:
                tracker.message("Phy export disabled")

        with tracker.phase("finalize manifest and registry"):
            row.update(
                {
                    **probe_hashes,
                    "quality_metrics_csv": str(outputs["quality_metrics_csv"]),
                    "status": "complete",
                    "preprocess_status": "complete",
                    "sort_status": "complete",
                    "qc_status": "complete",
                    "phy_export_status": "complete" if config.get("qc", {}).get("export_phy", True) else "skipped",
                    "error_message": "",
                    "last_updated": _now(),
                }
            )
            update_row(registry_csv, row)
        with tracker.phase("backup completed outputs"):
            if config.get("backup", {}).get("derived_outputs", {}).get("enabled", False) and config.get("backup", {}).get("derived_outputs", {}).get("auto_after_processing", False):
                tracker.message("backup generated outputs to Synology/D:")
                backup_derived_outputs_for_recording_session(config, session_id=row["session_id"], update_registry=True)
            else:
                tracker.message("generated-output backup disabled")

            archive_config = config.get("backup", {}).get("local_recording_archive", {})
            if archive_config.get("enabled", False) and archive_config.get("auto_after_recording", True):
                tracker.message("archive full local recording to D:")
                archive_recording_for_session(config, session_id=row["session_id"], update_registry=True)
            else:
                tracker.message("full local recording archive disabled")
        return row
    except BaseException as exc:
        row.update(_failed_status_update(row, error_message=_failure_message(exc)))
        update_row(registry_csv, row)
        raise
    finally:
        tracker.close()


def _detect_bad_channels(recording, config: dict[str, Any]):
    bad_config = config.get("preprocessing", {}).get("bad_channels", {})
    if not bad_config.get("enabled", True):
        return [], []
    return si.detect_bad_channels(recording, method=bad_config.get("method", "coherence+psd"))


def _bad_channels_for_sorting(recording, bad_channel_ids, channel_labels, config: dict[str, Any]) -> list:
    bad_config = config.get("preprocessing", {}).get("bad_channels", {})
    if not bad_config.get("exclude_from_sorting", False):
        return []

    label_lookup = _channel_label_lookup(recording, bad_channel_ids, channel_labels)
    exclude_labels = {str(label).lower() for label in bad_config.get("exclude_labels", ["dead", "noise", "out"])}
    return [
        channel_id
        for channel_id in bad_channel_ids
        if str(label_lookup.get(str(channel_id), "bad")).lower() in exclude_labels
    ]


def _apply_phase_shift(recording, config: dict[str, Any]):
    phase_config = config.get("preprocessing", {}).get("phase_shift", {})
    if not phase_config.get("enabled", True):
        return recording
    if not _recording_has_property(recording, "inter_sample_shift"):
        raise RuntimeError(
            "preprocessing.phase_shift is enabled, but the recording does not expose "
            "the Neuropixels 'inter_sample_shift' property."
        )
    return si.phase_shift(recording)


def _apply_highpass_filter(recording, config: dict[str, Any]):
    highpass_config = config.get("preprocessing", {}).get("highpass_filter", {})
    if not highpass_config.get("enabled", True):
        return recording
    return si.highpass_filter(recording, freq_min=float(highpass_config.get("freq_min", 300)))


def _apply_common_reference(recording, config: dict[str, Any]):
    car_config = config.get("preprocessing", {}).get("car", {})
    if not car_config.get("enabled", False):
        return recording
    return si.common_reference(
        recording,
        reference=car_config.get("reference", "global"),
        operator=car_config.get("operator", "median"),
    )


def _write_preprocessing_figures(pre_car_recording, post_car_recording, processed_folder: Path, config: dict[str, Any]) -> dict[str, str]:
    plot_config = config.get("preprocessing", {}).get("trace_plots", {})
    if not plot_config.get("enabled", True):
        return {}
    return write_preprocessing_trace_plots(
        pre_car_recording=pre_car_recording,
        post_car_recording=post_car_recording,
        output_folder=processed_folder,
        start_seconds=float(plot_config.get("start_seconds", 0)),
        seconds=float(plot_config.get("seconds", 2)),
        max_channels=int(plot_config.get("max_channels", 24)),
    )


def _apply_motion_correction(recording, processed_folder: Path, config: dict[str, Any], *, force: bool = False):
    motion_config = config.get("preprocessing", {}).get("motion_correction", {})
    if not motion_config.get("enabled", False):
        return recording

    folder = processed_folder / "motion_correction"
    kwargs = {
        key: value
        for key, value in motion_config.items()
        if key not in {"enabled", "save_motion_info"} and value is not None
    }
    preset = kwargs.pop("preset", "dredge_fast")
    save_motion_info = bool(motion_config.get("save_motion_info", True))
    try:
        result = si.correct_motion(
            recording,
            preset=preset,
            folder=folder,
            overwrite=force,
            output_motion_info=save_motion_info,
            **kwargs,
        )
    except TypeError:
        result = si.correct_motion(recording, preset=preset, folder=folder, overwrite=force, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


def _recording_has_property(recording, property_name: str) -> bool:
    if hasattr(recording, "get_property_keys"):
        return property_name in recording.get_property_keys()
    try:
        return recording.get_property(property_name) is not None
    except Exception:
        return False


def _has_processing_outputs(processed_folder: Path) -> bool:
    return any(
        (processed_folder / name).exists()
        for name in ["kilosort4", "sorting_analyzer", "quality_metrics.csv", "summary.json", "manifest.json", "phy"]
    )


def _write_channel_qc(recording, bad_channel_ids, channel_labels, excluded_bad_channel_ids, path: Path) -> None:
    label_lookup = _channel_label_lookup(recording, bad_channel_ids, channel_labels)
    bad_ids = {str(channel_id) for channel_id in bad_channel_ids}
    excluded_ids = {str(channel_id) for channel_id in excluded_bad_channel_ids}
    rows = []
    for index, channel_id in enumerate(recording.get_channel_ids()):
        channel_id = str(channel_id)
        rows.append(
            {
                "channel_index": index,
                "channel_id": channel_id,
                "label": label_lookup.get(channel_id, "good"),
                "detected_bad": channel_id in bad_ids,
                "excluded_from_sorting": channel_id in excluded_ids,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _channel_label_lookup(recording, bad_channel_ids, channel_labels) -> dict[str, str]:
    channel_ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
    labels = [str(label) for label in channel_labels]
    if len(labels) == len(channel_ids):
        return dict(zip(channel_ids, labels))
    if len(labels) == len(bad_channel_ids):
        lookup = {channel_id: "good" for channel_id in channel_ids}
        lookup.update({str(channel_id): label for channel_id, label in zip(bad_channel_ids, labels)})
        return lookup
    lookup = {channel_id: "good" for channel_id in channel_ids}
    lookup.update({str(channel_id): "bad" for channel_id in bad_channel_ids})
    return lookup


def _kilosort4_params(config: dict[str, Any], recording, bad_channel_ids) -> dict[str, Any]:
    params = si.get_default_sorter_params("kilosort4")
    params.update(config.get("sorting", {}).get("params", {}))
    params.update(
        {
            "do_CAR": False,
            "do_correction": not _motion_correction_enabled(config),
            "skip_kilosort_preprocessing": False,
            "save_preprocessed_copy": False,
            "delete_recording_dat": True,
            "nblocks": int(params.get("nblocks", 1)),
            "bad_channels": channel_ids_to_indices(recording, bad_channel_ids),
        }
    )
    return params


def _compute_analyzer(
    sorting,
    recording,
    folder: Path,
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    tracker: PhaseTracker | None = None,
):
    timing = getattr(tracker, "timing_recorder", None)
    with _timed(
        timing,
        "create SortingAnalyzer",
        category="analyzer_creation",
        metadata={"folder": str(folder), **_sorting_analyzer_kwargs(config)},
    ):
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            format="binary_folder",
            folder=folder,
            overwrite=overwrite,
            **_sorting_analyzer_kwargs(config),
        )
    _compute_analyzer_extensions(analyzer, config, tracker=tracker, skip_existing=False)
    return analyzer


def _compute_analyzer_extensions(
    analyzer,
    config: dict[str, Any],
    *,
    tracker: PhaseTracker | None = None,
    skip_existing: bool = False,
    recompute_extensions: list[str] | None = None,
):
    job_kwargs = _job_kwargs(config)
    fallback_n_jobs = _fallback_n_jobs(config)
    current_job_kwargs = dict(job_kwargs)
    if tracker is not None and current_job_kwargs:
        tracker.message(
            f"SortingAnalyzer job kwargs: {_format_job_kwargs(current_job_kwargs)}; "
            f"multiprocessing fallback_n_jobs={fallback_n_jobs}"
        )
    timing = getattr(tracker, "timing_recorder", None)
    recompute = set(recompute_extensions or [])
    for extension, extension_kwargs in _analyzer_extension_specs(config):
        if extension in recompute and _analyzer_has_extension(analyzer, extension):
            if tracker is not None:
                tracker.message(f"delete analyzer extension before recompute: {extension}")
            _delete_analyzer_extension(analyzer, extension, tracker=tracker)
        if skip_existing and _analyzer_has_extension(analyzer, extension):
            if tracker is not None:
                tracker.message(f"skip existing analyzer extension: {extension}")
            _record_timing(
                timing,
                f"analyzer extension: {extension}",
                category="analyzer_extension",
                status="skipped",
                metadata={"extension": extension, "reason": "already exists"},
            )
            continue
        if tracker is not None:
            tracker.message(f"compute analyzer extension: {extension}")
        metadata = {
            "extension": extension,
            "job_kwargs": dict(current_job_kwargs),
            "extension_kwargs": dict(extension_kwargs),
            "action": "recompute" if extension in recompute else "compute",
        }
        with _timed(timing, f"analyzer extension: {extension}", category="analyzer_extension", metadata=metadata) as event:
            fallback_kwargs = _compute_analyzer_extension(
                analyzer,
                extension,
                current_job_kwargs,
                extension_kwargs=extension_kwargs,
                fallback_n_jobs=fallback_n_jobs,
                tracker=tracker,
            )
            if fallback_kwargs is not None:
                event["metadata"]["final_job_kwargs"] = dict(fallback_kwargs)
                event["metadata"]["retry"] = "fallback_jobs"
        if fallback_kwargs is not None and fallback_kwargs != current_job_kwargs:
            current_job_kwargs = fallback_kwargs
            if tracker is not None:
                tracker.message(
                    "using fallback SortingAnalyzer job kwargs for remaining extensions: "
                    f"{_format_job_kwargs(current_job_kwargs)}"
                )
    return analyzer


def _analyzer_has_extension(analyzer, extension: str) -> bool:
    try:
        return bool(analyzer.has_extension(extension))
    except Exception:
        return False


def _delete_analyzer_extension(analyzer, extension: str, *, tracker: PhaseTracker | None = None) -> None:
    try:
        analyzer.delete_extension(extension)
        return
    except AttributeError as exc:
        if "delete" not in str(exc):
            raise
        if tracker is not None:
            tracker.message(f"SpikeInterface could not delete partial {extension}; removing extension folder")
    except ValueError as exc:
        if not _extension_folder_exists(analyzer, extension):
            raise
        if tracker is not None:
            tracker.message(f"SpikeInterface does not list partial {extension}; removing extension folder ({exc})")
    _remove_extension_folder(analyzer, extension)


def _extension_folder_exists(analyzer, extension: str) -> bool:
    analyzer_folder = Path(getattr(analyzer, "folder", ""))
    return bool(analyzer_folder) and (analyzer_folder / "extensions" / extension).exists()


def _remove_extension_folder(analyzer, extension: str) -> None:
    analyzer_folder = Path(getattr(analyzer, "folder", ""))
    if not analyzer_folder:
        raise RuntimeError(f"Cannot repair analyzer extension without analyzer.folder: {extension}")
    extension_folder = analyzer_folder / "extensions" / extension
    resolved_extension = extension_folder.resolve()
    resolved_root = (analyzer_folder / "extensions").resolve()
    if not _is_relative_to(resolved_extension, resolved_root):
        raise RuntimeError(f"Refusing to remove analyzer extension outside analyzer folder: {resolved_extension}")
    if resolved_extension.exists():
        shutil.rmtree(resolved_extension)
    extensions = getattr(analyzer, "extensions", None)
    if isinstance(extensions, dict):
        extensions.pop(extension, None)


def _compute_analyzer_extension(
    analyzer,
    extension: str,
    job_kwargs: dict[str, Any],
    *,
    extension_kwargs: dict[str, Any] | None = None,
    fallback_n_jobs: int = 2,
    tracker: PhaseTracker | None = None,
) -> dict[str, Any] | None:
    extension_kwargs = extension_kwargs or {}
    try:
        analyzer.compute(extension, **extension_kwargs, **job_kwargs)
    except TypeError as exc:
        if not _looks_like_unexpected_job_kwarg(exc, job_kwargs):
            raise
        if tracker is not None:
            tracker.message(
                f"{extension} rejected job kwargs ({exc}); retrying without {_format_job_kwargs(job_kwargs)}"
            )
        analyzer.compute(extension, **extension_kwargs)
        return None
    except (OSError, EOFError, BrokenProcessPool) as exc:
        if not _looks_like_multiprocessing_spawn_failure(exc, job_kwargs):
            raise
        fallback_kwargs = _fallback_job_kwargs(job_kwargs, fallback_n_jobs)
        if tracker is not None:
            tracker.message(
                f"{extension} failed with multiprocessing error ({_failure_message(exc)}); "
                f"retrying with {_format_job_kwargs(fallback_kwargs)}"
            )
        _delete_analyzer_extension(analyzer, extension, tracker=tracker)
        try:
            analyzer.compute(extension, **extension_kwargs, **fallback_kwargs)
        except (OSError, EOFError, BrokenProcessPool) as fallback_exc:
            if not _looks_like_multiprocessing_spawn_failure(fallback_exc, fallback_kwargs):
                raise
            serial_kwargs = _fallback_job_kwargs(job_kwargs, 1)
            if tracker is not None:
                tracker.message(
                    f"{extension} also failed with fallback workers ({_failure_message(fallback_exc)}); "
                    f"retrying serially with {_format_job_kwargs(serial_kwargs)}"
                )
            _delete_analyzer_extension(analyzer, extension, tracker=tracker)
            analyzer.compute(extension, **extension_kwargs, **serial_kwargs)
            return serial_kwargs
        return fallback_kwargs
    return None


def _looks_like_unexpected_job_kwarg(exc: TypeError, job_kwargs: dict[str, Any]) -> bool:
    if not job_kwargs:
        return False
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in [
            "unexpected keyword argument",
            "got an unexpected",
            "invalid keyword",
            "unexpected keyword",
        ]
    )


def _looks_like_multiprocessing_spawn_failure(exc: BaseException, job_kwargs: dict[str, Any]) -> bool:
    if int(job_kwargs.get("n_jobs", 1) or 1) <= 1:
        return False
    if isinstance(exc, BrokenProcessPool):
        return True
    if isinstance(exc, EOFError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 22:
        return True
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in [
            "pickle data was truncated",
            "broken process pool",
            "invalid argument",
            "process was terminated abruptly",
        ]
    )


def _fallback_job_kwargs(job_kwargs: dict[str, Any], fallback_n_jobs: int) -> dict[str, Any]:
    fallback = dict(job_kwargs)
    fallback["n_jobs"] = max(1, int(fallback_n_jobs))
    return fallback


def _job_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.get("jobs", {}).items() if key in JOB_KWARG_KEYS}


def _fallback_n_jobs(config: dict[str, Any]) -> int:
    return max(1, int(config.get("jobs", {}).get("fallback_n_jobs", 2)))


def _format_job_kwargs(job_kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(job_kwargs.items()))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _export_metrics_and_summary(
    analyzer,
    row: dict[str, str],
    config: dict[str, Any],
    channel_qc_csv: Path,
    preprocessing_config: dict[str, Any],
    sorter_params: dict[str, Any],
    preprocessing_figures: dict[str, str] | None = None,
    tracker: PhaseTracker | None = None,
) -> dict[str, Path]:
    processed_folder = Path(row["processed_folder"])
    quality_metrics_csv = processed_folder / "quality_metrics.csv"
    template_metrics_csv = processed_folder / "template_metrics.csv"
    unit_summary_csv = processed_folder / "unit_summary.csv"
    summary_json = processed_folder / "summary.json"
    manifest_json = processed_folder / "manifest.json"

    timing = getattr(tracker, "timing_recorder", None)
    with _timed(timing, "export metric CSVs", category="metrics_export"):
        qm = analyzer.get_extension("quality_metrics").get_data()
        qm.to_csv(quality_metrics_csv)
        try:
            tm = analyzer.get_extension("template_metrics").get_data()
            tm.to_csv(template_metrics_csv)
        except Exception:
            tm = pd.DataFrame()

        unit_summary = _unit_summary(analyzer, qm)
        unit_summary.to_csv(unit_summary_csv, index=False)
    with _timed(timing, "run UnitRefine", category="unitrefine"):
        automated_labels = run_unitrefine_for_analyzer(
            analyzer,
            processed_folder,
            config,
            overwrite=True,
        )
        _attach_automated_label_properties(analyzer, automated_labels, config)

    summary = {
        "session_id": row["session_id"],
        "probe_label": row["probe_label"],
        "raw_folder": row["raw_folder"],
        "stream_name": row["stream_name"],
        "num_units": int(len(analyzer.sorting.get_unit_ids())),
        "num_channels": int(analyzer.recording.get_num_channels()),
        "duration_seconds": float(analyzer.recording.get_total_duration()),
        "bad_channels_csv": str(channel_qc_csv),
        "quality_metrics_csv": str(quality_metrics_csv),
        "template_metrics_csv": str(template_metrics_csv) if not tm.empty else "",
        "automated_labels_csv": automated_labels.get("unit_labels_csv", ""),
        "automated_labels_summary": automated_labels,
        "automated_labels_status": automated_labels.get("status", ""),
        "preprocessing_figures": preprocessing_figures or {},
        "qc_timings_json": str(timing_path(processed_folder)),
        "phy_compute_pc_features": bool(config.get("qc", {}).get("phy_compute_pc_features", True)),
        "created_at": _now(),
    }
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    manifest = {
        **summary,
        "probeinterface_json": row.get("probeinterface_json"),
        "probeinterface_hash": row.get("probeinterface_hash"),
        "channel_ids_hash": row.get("channel_ids_hash"),
        "geometry_hash": row.get("geometry_hash"),
        "site_hash": row.get("site_hash"),
        "spikeinterface_version": spikeinterface.__version__,
        "kilosort_version": getattr(kilosort, "__version__", ""),
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_hash": hash_jsonable(preprocessing_config),
        "sorter_name": config.get("sorting", {}).get("sorter_name", "kilosort4"),
        "sorter_params": sorter_params,
        "status": "complete",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if config.get("qc", {}).get("export_report", True):
        with _timed(timing, "write static reports", category="report"):
            write_summary_report(
                analyzer=analyzer,
                probe_label=row["probe_label"],
                output_folder=processed_folder,
                quality_metrics_csv=quality_metrics_csv,
                channel_qc_csv=channel_qc_csv,
                summary=summary,
                config=config,
            )
    return {"quality_metrics_csv": quality_metrics_csv}


def _unit_summary(analyzer, quality_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unit_ids = analyzer.sorting.get_unit_ids()
    for unit_id in unit_ids:
        row = {"unit_id": unit_id, "num_spikes": int(analyzer.sorting.get_unit_spike_train(unit_id).size)}
        if unit_id in quality_metrics.index:
            for column in ["firing_rate", "snr", "presence_ratio", "amplitude_cutoff", "amplitude_median"]:
                if column in quality_metrics:
                    row[column] = quality_metrics.loc[unit_id, column]
        rows.append(row)
    return pd.DataFrame(rows)


def _sorting_analyzer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    analyzer_config = config.get("qc", {}).get("sorting_analyzer", {})
    return {
        "sparse": bool(analyzer_config.get("sparse", True)),
        "num_spikes_for_sparsity": int(analyzer_config.get("num_spikes_for_sparsity", 200)),
    }


def _analyzer_extension_specs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    configured = config.get("qc", {}).get("analyzer_extensions")
    if not configured:
        return [(name, _extension_params_from_config(config, name, {})) for name in DEFAULT_ANALYZER_EXTENSIONS]

    specs: list[tuple[str, dict[str, Any]]] = []
    for item in configured:
        if isinstance(item, str):
            name = item
            params: dict[str, Any] = {}
            enabled = True
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            params = dict(item.get("params") or {})
            enabled = bool(item.get("enabled", True))
        else:
            raise TypeError(f"Invalid analyzer extension config entry: {item!r}")
        if not enabled:
            continue
        if name not in DEFAULT_ANALYZER_EXTENSIONS:
            raise ValueError(f"Unknown analyzer extension: {name}")
        specs.append((name, _extension_params_from_config(config, name, params)))
    return specs


def _extension_params_from_config(config: dict[str, Any], extension: str, params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if extension == "quality_metrics":
        metric_names = _quality_metric_compute_names(config, base_metric_names=params.get("metric_names"))
        if metric_names:
            params["metric_names"] = metric_names
    return params


def _quality_metric_names_from_value(value: Any) -> list[str]:
    names = value
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",")]
    return [str(name).strip() for name in names if str(name).strip()]


def _quality_metric_names(config: dict[str, Any]) -> list[str]:
    metric_config = config.get("qc", {}).get("quality_metrics", {})
    return _quality_metric_names_from_value(metric_config.get("metric_names", []))


def _unitrefine_quality_metric_names(config: dict[str, Any]) -> list[str]:
    label_config = config.get("qc", {}).get("automated_labels", {})
    if not label_config.get("enabled", False):
        return []
    metric_config = config.get("qc", {}).get("quality_metrics", {})
    names = metric_config.get("unitrefine_metric_names", DEFAULT_UNITREFINE_QUALITY_METRICS)
    if names in (None, False):
        return []
    if isinstance(names, str) and names.strip().lower() in {"default", "unitrefine_default", "all"}:
        return list(DEFAULT_UNITREFINE_QUALITY_METRICS)
    return _quality_metric_names_from_value(names)


def _quality_metric_compute_names(config: dict[str, Any], *, base_metric_names: Any = None) -> list[str]:
    compute_names: list[str] = []
    base_names = _quality_metric_names(config) if base_metric_names is None else _quality_metric_names_from_value(base_metric_names)
    for name in [*base_names, *_unitrefine_quality_metric_names(config)]:
        compute_name = QUALITY_METRIC_ALIASES.get(name, name)
        if compute_name not in compute_names:
            compute_names.append(compute_name)
    return compute_names


def _phy_export_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    qc_config = config.get("qc", {})
    kwargs = {
        "compute_pc_features": bool(qc_config.get("phy_compute_pc_features", True)),
        "compute_amplitudes": bool(qc_config.get("phy_compute_amplitudes", True)),
        "copy_binary": bool(qc_config.get("phy_copy_binary", False)),
        **_job_kwargs(config),
    }
    if qc_config.get("phy_add_unitrefine_labels", True):
        kwargs["additional_properties"] = ["unitrefine_label", "unitrefine_probability"]
    return kwargs


def _phy_metric_export_kwargs(config: dict[str, Any]) -> dict[str, bool]:
    qc_config = config.get("qc", {})
    return {
        "add_quality_metrics": bool(qc_config.get("phy_add_quality_metrics", False)),
        "add_template_metrics": bool(qc_config.get("phy_add_template_metrics", False)),
    }


def _prune_phy_metric_tsvs(phy_folder: str | Path, config: dict[str, Any], *, tracker: PhaseTracker | None = None) -> list[Path]:
    qc_config = config.get("qc", {})
    if not qc_config.get("phy_prune_metric_tsvs", True):
        return []
    if qc_config.get("phy_add_quality_metrics", False) or qc_config.get("phy_add_template_metrics", False):
        return []
    keep = {
        "cluster_channel_group.tsv",
        "cluster_group.tsv",
        "cluster_si_unit_ids.tsv",
        "cluster_unitrefine_label.tsv",
        "cluster_unitrefine_probability.tsv",
    }
    removed: list[Path] = []
    for path in Path(phy_folder).glob("cluster_*.tsv"):
        if path.name in keep:
            continue
        path.unlink()
        removed.append(path)
    if tracker is not None and removed:
        tracker.message(f"pruned {len(removed)} Phy metric TSV files")
    return removed


def _write_phy_unitrefine_tsvs(
    phy_folder: str | Path,
    processed_folder: str | Path,
    config: dict[str, Any],
    *,
    tracker: PhaseTracker | None = None,
) -> list[Path]:
    if not config.get("qc", {}).get("phy_add_unitrefine_labels", True):
        return []
    phy = Path(phy_folder)
    unit_labels_csv = _latest_unitrefine_labels_csv(processed_folder)
    cluster_map_tsv = phy / "cluster_si_unit_ids.tsv"
    if unit_labels_csv is None or not cluster_map_tsv.exists():
        return []
    labels = pd.read_csv(unit_labels_csv)
    if labels.empty:
        return []
    if "unit_id" not in labels.columns:
        labels = labels.rename(columns={labels.columns[0]: "unit_id"})
    labels["unit_id"] = labels["unit_id"].astype(str)
    label_lookup = labels.set_index("unit_id").to_dict(orient="index")
    cluster_map = pd.read_csv(cluster_map_tsv, sep="\t")
    if "cluster_id" not in cluster_map.columns or "si_unit_id" not in cluster_map.columns:
        return []
    rows = []
    for _, cluster in cluster_map.iterrows():
        unit_id = str(cluster["si_unit_id"])
        label_row = label_lookup.get(unit_id, {})
        rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "unitrefine_label": label_row.get("unitrefine_label", ""),
                "unitrefine_probability": label_row.get("unitrefine_probability", ""),
            }
        )
    label_path = phy / "cluster_unitrefine_label.tsv"
    probability_path = phy / "cluster_unitrefine_probability.tsv"
    label_frame = pd.DataFrame({"cluster_id": [row["cluster_id"] for row in rows], "unitrefine_label": [row["unitrefine_label"] for row in rows]})
    probability_frame = pd.DataFrame(
        {"cluster_id": [row["cluster_id"] for row in rows], "unitrefine_probability": [row["unitrefine_probability"] for row in rows]}
    )
    label_frame.to_csv(label_path, sep="\t", index=False)
    probability_frame.to_csv(probability_path, sep="\t", index=False)
    if tracker is not None:
        tracker.message(f"wrote Phy UnitRefine TSVs: {label_path.name}, {probability_path.name}")
    return [label_path, probability_path]


def _latest_unitrefine_labels_csv(processed_folder: str | Path) -> Path | None:
    root = Path(processed_folder) / "unitrefine"
    if not root.exists():
        return None
    labels = sorted(root.glob("*/unit_labels.csv"), key=lambda path: path.stat().st_mtime)
    return labels[-1] if labels else None


def _prepare_phy_extensions(analyzer, config: dict[str, Any], *, tracker: PhaseTracker | None = None) -> None:
    qc_config = config.get("qc", {})
    job_kwargs = _job_kwargs(config)
    fallback_n_jobs = _fallback_n_jobs(config)
    if qc_config.get("phy_compute_amplitudes", True):
        _ensure_phy_extension(
            analyzer,
            "spike_amplitudes",
            job_kwargs=job_kwargs,
            fallback_n_jobs=fallback_n_jobs,
            tracker=tracker,
        )
    if qc_config.get("phy_compute_pc_features", True):
        _ensure_phy_extension(
            analyzer,
            "principal_components",
            job_kwargs=job_kwargs,
            fallback_n_jobs=fallback_n_jobs,
            extension_kwargs=_extension_params_from_config(config, "principal_components", {"n_components": 5, "mode": "by_channel_local"}),
            tracker=tracker,
        )
    elif tracker is not None:
        tracker.message("Phy export will skip pc_features.npy because qc.phy_compute_pc_features=false")


def _patch_phy_dat_path(
    phy_folder: str | Path,
    row: dict[str, str],
    config: dict[str, Any],
    *,
    tracker: PhaseTracker | None = None,
) -> Path | None:
    dat_config = config.get("qc", {}).get("phy_dat_path", {})
    if not dat_config.get("enabled", False):
        return None
    params_py = Path(phy_folder) / "params.py"
    if not params_py.exists():
        if tracker is not None:
            tracker.message(f"Phy dat_path patch skipped; params.py missing: {params_py}")
        return None
    dat_path = _resolve_phy_dat_path(row, config)
    if dat_path is None:
        if tracker is not None:
            tracker.message("Phy dat_path patch skipped; original continuous.dat not found")
        return None
    text = params_py.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    hp_filtered = bool(dat_config.get("hp_filtered", False))
    updated: list[str] = []
    saw_dat_path = False
    saw_hp_filtered = False
    dat_literal = repr(str(dat_path))
    for line in lines:
        if line.startswith("dat_path"):
            updated.append(f"dat_path = {dat_literal}")
            saw_dat_path = True
        elif line.startswith("hp_filtered"):
            updated.append(f"hp_filtered = {hp_filtered}")
            saw_hp_filtered = True
        else:
            updated.append(line)
    if not saw_dat_path:
        updated.insert(0, f"dat_path = {dat_literal}")
    if not saw_hp_filtered:
        updated.append(f"hp_filtered = {hp_filtered}")
    params_py.write_text("\n".join(updated) + "\n", encoding="utf-8")
    if tracker is not None:
        tracker.message(f"Phy dat_path patched to original recording: {dat_path}")
    return dat_path


def _resolve_phy_dat_path(row: dict[str, str], config: dict[str, Any]) -> Path | None:
    dat_config = config.get("qc", {}).get("phy_dat_path", {})
    explicit = str(dat_config.get("path") or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    source = str(dat_config.get("source", "original_continuous_dat")).lower()
    if source != "original_continuous_dat":
        return None
    processed = Path(row.get("processed_folder", ""))
    for candidate_root in [processed, *processed.parents]:
        candidate = candidate_root / "continuous.dat"
        if candidate.exists():
            return candidate
    return None


def _ensure_phy_extension(
    analyzer,
    extension: str,
    *,
    job_kwargs: dict[str, Any],
    fallback_n_jobs: int,
    extension_kwargs: dict[str, Any] | None = None,
    tracker: PhaseTracker | None = None,
) -> None:
    if _analyzer_has_extension(analyzer, extension):
        if tracker is not None:
            tracker.message(f"Phy export will reuse existing analyzer extension: {extension}")
        return
    if tracker is not None:
        tracker.message(f"Phy export needs missing analyzer extension: {extension}")
    _compute_analyzer_extension(
        analyzer,
        extension,
        job_kwargs,
        extension_kwargs=extension_kwargs,
        fallback_n_jobs=fallback_n_jobs,
        tracker=tracker,
    )


def _attach_automated_label_properties(analyzer, automated_labels: dict[str, Any], config: dict[str, Any]) -> None:
    if not config.get("qc", {}).get("phy_add_unitrefine_labels", True):
        return
    labels_csv = automated_labels.get("unit_labels_csv")
    if not labels_csv:
        return
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        return
    labels = pd.read_csv(labels_path)
    if labels.empty:
        return
    if "unit_id" in labels.columns:
        labels = labels.set_index("unit_id")
    elif str(labels.columns[0]).startswith("Unnamed"):
        labels = labels.set_index(labels.columns[0])

    unit_ids = list(analyzer.sorting.get_unit_ids())
    label_lookup = {str(index): row for index, row in labels.iterrows()}
    if "unitrefine_label" in labels:
        analyzer.sorting.set_property(
            "unitrefine_label",
            [label_lookup.get(str(unit_id), pd.Series(dtype=object)).get("unitrefine_label", "") for unit_id in unit_ids],
        )
    if "unitrefine_probability" in labels:
        analyzer.sorting.set_property(
            "unitrefine_probability",
            [
                label_lookup.get(str(unit_id), pd.Series(dtype=object)).get("unitrefine_probability", float("nan"))
                for unit_id in unit_ids
            ],
        )


def _preprocessing_manifest(config: dict[str, Any]) -> dict[str, Any]:
    preprocessing = config.get("preprocessing", {})
    return {
        "strategy": "spikeinterface_phase_shift_median_car_highpass_then_kilosort4",
        "phase_shift": preprocessing.get("phase_shift", {}),
        "spikeinterface_common_reference": preprocessing.get("car", {}),
        "highpass_filter": preprocessing.get("highpass_filter", {}),
        "spikeinterface_motion_correction": preprocessing.get("motion_correction", {}),
        "trace_plots": preprocessing.get("trace_plots", {}),
        "save_preprocessed_recording": False,
        "bad_channel_detection": preprocessing.get("bad_channels", {}),
        "kilosort4_internal_car": False,
        "kilosort4_internal_drift_correction": not _motion_correction_enabled(config),
    }


def _motion_correction_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("preprocessing", {}).get("motion_correction", {}).get("enabled", False))


def _status_update(status: str, **fields: str) -> dict[str, str]:
    return {"status": status, "last_updated": _now(), **fields}


def _failed_status_update(row: dict[str, str], *, error_message: str) -> dict[str, str]:
    fields = _status_update("failed", error_message=error_message)
    for key in ("preprocess_status", "sort_status", "qc_status", "phy_export_status"):
        if row.get(key) == "running":
            fields[key] = "failed"
    return fields


def _failure_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _append_note(existing: str, note: str) -> str:
    return f"{existing}; {note}" if existing else note


def _sanitize_output_suffix(suffix: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("--output-suffix must contain at least one letter or number")
    return sanitized


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timed(timing_recorder, name: str, *, category: str, metadata: dict[str, Any] | None = None):
    if timing_recorder is None:
        return _null_timing_context()
    return timing_recorder.phase(name, category=category, metadata=metadata)


def _record_timing(
    timing_recorder,
    name: str,
    *,
    category: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if timing_recorder is not None:
        timing_recorder.record(name, category=category, status=status, metadata=metadata)


def _null_timing_context():
    class _NullTimingContext:
        def __enter__(self):
            return {"metadata": {}}

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullTimingContext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one registered probe through Kilosort4 and QC.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id")
    parser.add_argument("--raw-folder")
    parser.add_argument("--probe-label")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--output-suffix",
        help="Write to a sibling output folder named spikeinterface_output_<suffix> and create a separate registry row.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    row = find_row(
        config["project"]["registry_csv"],
        session_id=args.session_id,
        raw_folder=args.raw_folder,
        probe_label=args.probe_label,
    )
    if _auto_stage_before_run_one(config):
        print(f"staging recording locally before run_one: {row.get('server_raw_folder') or row.get('raw_folder')}")
        stage_recording_for_session(config, session_id=row["session_id"], update_registry=True)
        row = find_row(config["project"]["registry_csv"], session_id=row["session_id"])
    completed = run_registry_row(
        row,
        config,
        force=args.force,
        duration_seconds=args.duration_seconds,
        output_suffix=args.output_suffix,
        show_progress=not args.no_progress,
    )
    print(f"{completed['session_id']} | {completed['probe_label']} | {completed['status']}")

def _auto_stage_before_run_one(config: dict[str, Any]) -> bool:
    staging = config.get("staging", {})
    return bool(staging.get("enabled", False)) and bool(staging.get("auto_stage_before_run_one", True))


if __name__ == "__main__":
    main()
