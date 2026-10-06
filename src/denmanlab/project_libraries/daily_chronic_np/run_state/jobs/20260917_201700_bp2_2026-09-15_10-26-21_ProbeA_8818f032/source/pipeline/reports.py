from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.full as si

from .config import load_config
from .probe_maps import load_recording_with_probe
from .registry import read_registry


HIGH_YIELD_METRICS = [
    "isi_violations_ratio",
    "rp_contamination",
    "sliding_rp_violation",
    "snr",
    "presence_ratio",
    "amplitude_cutoff",
    "amplitude_median",
    "firing_rate",
]


def write_summary_report(
    *,
    analyzer,
    probe_label: str,
    output_folder: str | Path,
    quality_metrics_csv: str | Path,
    channel_qc_csv: str | Path,
    summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_folder)
    report_dir = output / "report"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    preprocessing_trace_plot = figures_dir / "pre_post_car_traces.png"
    if preprocessing_trace_plot.exists():
        figure_paths.append(preprocessing_trace_plot)
    figure_paths.append(_plot_probe_geometry(analyzer.recording, channel_qc_csv, figures_dir / "probe_geometry.png"))
    qm = _read_metrics_csv(quality_metrics_csv)
    labels = _read_unitrefine_labels_for_output(output, summary)
    qm = _metrics_with_unitrefine_labels(qm, labels)
    if not qm.empty:
        figure_paths.append(_plot_metric_histograms(qm, figures_dir / "quality_metric_histograms.png"))
    figure_paths.append(_plot_sorting_summary(analyzer, figures_dir / "sorting_summary.png"))
    shareable_png_path = None
    if _shareable_png_enabled(config):
        shareable_png_path = write_shareable_summary_png(
            analyzer=analyzer,
            probe_label=probe_label,
            output_folder=output,
            quality_metrics_csv=quality_metrics_csv,
            channel_qc_csv=channel_qc_csv,
            summary=summary,
            config=config or {},
        )

    report_path = report_dir / "summary.html"
    escaped_summary = html.escape(json.dumps(summary, indent=2, default=str))
    if shareable_png_path is not None and shareable_png_path.exists():
        figure_paths.insert(0, shareable_png_path)
    images = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2><img src="{html.escape(os.path.relpath(path, report_dir).replace(os.sep, "/"))}" /></section>'
        for path in figure_paths
        if path.exists()
    )
    report_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>Spike sorting summary</title>",
                "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.4} img{max-width:100%;border:1px solid #ddd} pre{background:#f6f6f6;padding:12px;overflow:auto} table{border-collapse:collapse;margin:12px 0 24px 0;font-size:13px} th,td{border:1px solid #ddd;padding:5px 8px;text-align:right} th:first-child,td:first-child{text-align:left} h2{margin-top:28px}</style>",
                "</head><body>",
                f"<h1>{html.escape(probe_label)} spike sorting summary</h1>",
                f"<pre>{escaped_summary}</pre>",
                _expanded_qc_html(qm, labels),
                images,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def write_shareable_summary_png(
    *,
    analyzer,
    probe_label: str,
    output_folder: str | Path,
    quality_metrics_csv: str | Path,
    channel_qc_csv: str | Path,
    summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_folder)
    report_dir = output / "report"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_config = (config or {}).get("qc", {}).get("shareable_png", {})
    png_path = report_dir / str(png_config.get("filename", "summary.png"))

    metrics = _read_metrics_csv(quality_metrics_csv)
    labels = _read_unitrefine_labels_for_output(output, summary)
    metrics = _metrics_with_unitrefine_labels(metrics, labels)
    channel_qc = _read_csv(channel_qc_csv)
    preprocessing_trace_png = figures_dir / "pre_post_car_traces.png"

    fig = plt.figure(figsize=(16, 10), dpi=190)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[0.58, 2.2, 2.5, 2.7],
        width_ratios=[1.15, 1.25, 1.25, 1.1],
        hspace=0.48,
        wspace=0.35,
    )
    _draw_header(fig.add_subplot(gs[0, :]), summary, probe_label)
    _draw_probe_geometry(fig.add_subplot(gs[1:3, 0]), _safe_recording(analyzer), channel_qc)
    _draw_image_panel(fig.add_subplot(gs[1, 1:3]), preprocessing_trace_png, "Pre/post CAR examples")
    _draw_policy_box(fig.add_subplot(gs[1, 3]), summary, channel_qc, config or {}, labels)
    _draw_metric_histograms(fig.add_subplot(gs[2, 1:3]), metrics)
    _draw_unit_qc_scatter(fig.add_subplot(gs[2, 3]), metrics)
    _draw_unit_locations(fig.add_subplot(gs[3, 0]), analyzer)
    _draw_example_templates(fig.add_subplot(gs[3, 1:4]), analyzer, metrics, int(png_config.get("example_units", 6)))

    fig.savefig(png_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def write_preprocessing_trace_plots(
    *,
    pre_car_recording,
    post_car_recording,
    output_folder: str | Path,
    start_seconds: float = 0.0,
    seconds: float = 2.0,
    max_channels: int = 24,
) -> dict[str, str]:
    output = Path(output_folder)
    figures_dir = output / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "pre_post_car_traces.png"

    if seconds <= 0 or max_channels <= 0:
        return {}

    channel_ids = _evenly_spaced_channel_ids(pre_car_recording, max_channels)
    if len(channel_ids) == 0:
        return {}

    sampling_frequency = float(pre_car_recording.get_sampling_frequency())
    num_frames = int(pre_car_recording.get_num_frames())
    start_frame = max(0, min(int(float(start_seconds) * sampling_frequency), max(num_frames - 1, 0)))
    end_frame = min(num_frames, start_frame + max(1, int(float(seconds) * sampling_frequency)))
    if end_frame <= start_frame:
        return {}

    pre_traces = _get_traces(pre_car_recording, start_frame, end_frame, channel_ids)
    post_traces = _get_traces(post_car_recording, start_frame, end_frame, channel_ids)
    if pre_traces.size == 0 or post_traces.size == 0:
        return {}

    scale = _shared_trace_scale(pre_traces, post_traces)
    time = np.arange(pre_traces.shape[0]) / sampling_frequency
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    _plot_stacked_traces(axes[0], time, pre_traces, channel_ids, "Before CAR", scale=scale)
    _plot_stacked_traces(axes[1], time, post_traces, channel_ids, "After SpikeInterface CAR", scale=scale)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Preprocessing trace comparison; pre/post panels use matched per-channel voltage scales", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"pre_post_car_traces_png": str(path)}


def _evenly_spaced_channel_ids(recording, max_channels: int):
    channel_ids = list(recording.get_channel_ids())
    if len(channel_ids) <= max_channels:
        return channel_ids
    indices = np.linspace(0, len(channel_ids) - 1, max_channels, dtype=int)
    return [channel_ids[index] for index in indices]


def _get_traces(recording, start_frame: int, end_frame: int, channel_ids) -> np.ndarray:
    try:
        traces = recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=channel_ids,
            return_in_uV=True,
        )
    except TypeError:
        try:
            traces = recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                channel_ids=channel_ids,
                return_scaled=True,
            )
        except TypeError:
            traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids)
    traces = np.asarray(traces, dtype=float)
    if traces.ndim == 1:
        traces = traces[:, np.newaxis]
    return traces


def _shared_trace_scale(pre_traces: np.ndarray, post_traces: np.ndarray) -> np.ndarray:
    pre_centered = pre_traces - np.nanmedian(pre_traces, axis=0, keepdims=True)
    post_centered = post_traces - np.nanmedian(post_traces, axis=0, keepdims=True)
    combined = np.concatenate([np.abs(pre_centered), np.abs(post_centered)], axis=0)
    scale = np.nanpercentile(combined, 95, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return scale


def _plot_stacked_traces(ax, time: np.ndarray, traces: np.ndarray, channel_ids, title: str, *, scale: np.ndarray | None = None) -> None:
    centered = traces - np.nanmedian(traces, axis=0, keepdims=True)
    if scale is None:
        scale = np.nanpercentile(np.abs(centered), 95, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    normalized = centered / scale
    offsets = np.arange(normalized.shape[1], dtype=float) * 4.0
    for index in range(normalized.shape[1]):
        ax.plot(time, normalized[:, index] + offsets[index], linewidth=0.55, color="#334155")
    ax.set_title(title)
    ax.set_ylabel("channels\n(shared scale)")
    tick_step = max(1, len(channel_ids) // 8)
    ticks = offsets[::tick_step]
    labels = [str(channel_id) for channel_id in channel_ids[::tick_step]]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)


def _shareable_png_enabled(config: dict[str, Any] | None) -> bool:
    if config is None:
        return True
    return bool(config.get("qc", {}).get("shareable_png", {}).get("enabled", True))


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_metrics_csv(path: str | Path) -> pd.DataFrame:
    metrics = _read_csv(path)
    if not metrics.empty and str(metrics.columns[0]).startswith("Unnamed"):
        metrics = metrics.set_index(metrics.columns[0])
    elif not metrics.empty and "unit_id" in metrics.columns:
        metrics = metrics.set_index("unit_id")
    return metrics


def _read_unitrefine_labels_for_output(output_folder: str | Path, summary: dict[str, Any]) -> pd.DataFrame:
    candidates: list[Path] = []
    summary_csv = summary.get("automated_labels_csv")
    if summary_csv:
        candidates.append(Path(str(summary_csv)))
    unitrefine_root = Path(output_folder) / "unitrefine"
    if unitrefine_root.exists():
        candidates.extend(
            sorted(unitrefine_root.glob("*/unit_labels.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    for candidate in candidates:
        labels = _read_unitrefine_labels(candidate)
        if not labels.empty:
            return labels
    return pd.DataFrame()


def _read_unitrefine_labels(path: str | Path) -> pd.DataFrame:
    labels = _read_csv(path)
    if labels.empty:
        return labels
    if "unit_id" in labels.columns:
        labels = labels.set_index("unit_id")
    elif str(labels.columns[0]).startswith("Unnamed"):
        labels = labels.set_index(labels.columns[0])
    return labels


def _metrics_with_unitrefine_labels(metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or labels.empty:
        return metrics
    merged = metrics.copy()
    label_lookup = {str(index): row for index, row in labels.iterrows()}
    for column in ["unitrefine_label", "unitrefine_probability"]:
        if column not in labels:
            continue
        merged[column] = [
            label_lookup.get(str(index), pd.Series(dtype=object)).get(column, np.nan) for index in merged.index
        ]
    return merged


def _draw_header(ax, summary: dict[str, Any], probe_label: str) -> None:
    ax.axis("off")
    session = summary.get("session_id", "")
    duration = float(summary.get("duration_seconds", 0) or 0)
    hours = duration / 3600
    title = f"{session} | {probe_label}"
    details = (
        f"units: {summary.get('num_units', 'n/a')}    "
        f"channels: {summary.get('num_channels', 'n/a')}    "
        f"duration: {duration:.0f}s ({hours:.2f}h)    "
        f"created: {summary.get('created_at', 'n/a')}"
    )
    ax.text(0.0, 0.72, title, fontsize=15, fontweight="bold", ha="left", va="center")
    ax.text(0.0, 0.2, details, fontsize=9.5, color="#334155", ha="left", va="center")


def _draw_unavailable(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="#64748b")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")


def _draw_image_panel(ax, image_path: Path, title: str) -> None:
    if not image_path.exists():
        _draw_unavailable(ax, title, "pre/post CAR trace image unavailable")
        return
    try:
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    except Exception as exc:
        _draw_unavailable(ax, title, f"image unavailable: {exc}")


def _draw_probe_geometry(ax, recording, channel_qc: pd.DataFrame) -> None:
    if recording is None:
        _draw_unavailable(ax, "Probe geometry", "recording unavailable")
        return
    try:
        locations = recording.get_channel_locations()
    except Exception as exc:
        _draw_unavailable(ax, "Probe geometry", f"locations unavailable: {exc}")
        return
    ax.scatter(locations[:, 0], locations[:, 1], s=8, c="#64748b", alpha=0.65, label="channel")
    bad = _bad_channel_rows(channel_qc)
    if not bad.empty and "channel_index" in bad:
        indices = bad["channel_index"].astype(int).to_numpy()
        indices = indices[(indices >= 0) & (indices < locations.shape[0])]
        if indices.size:
            ax.scatter(locations[indices, 0], locations[indices, 1], s=24, c="#dc2626", label="detected bad")
    ax.set_title("Probe geometry")
    ax.set_xlabel("x um")
    ax.set_ylabel("y um")
    ax.legend(loc="best", fontsize=7)


def _bad_channel_rows(channel_qc: pd.DataFrame) -> pd.DataFrame:
    if channel_qc.empty:
        return pd.DataFrame()
    if "detected_bad" in channel_qc:
        return channel_qc[channel_qc["detected_bad"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if "label" in channel_qc:
        return channel_qc[channel_qc["label"].astype(str).str.lower() != "good"]
    if "status" in channel_qc:
        return channel_qc[channel_qc["status"].astype(str).str.lower() != "good"]
    return pd.DataFrame()


def _draw_metric_histograms(ax, metrics: pd.DataFrame) -> None:
    columns = [column for column in HIGH_YIELD_METRICS if column in metrics]
    if metrics.empty or not columns:
        _draw_unavailable(ax, "Quality metric histograms", "quality metrics unavailable")
        return
    ax.set_title("Quality metric distributions")
    colors = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#dc2626", "#0f766e", "#4f46e5", "#0891b2"]
    for index, column in enumerate(columns[:8]):
        values = pd.to_numeric(metrics[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        if column in {"firing_rate"}:
            values = np.log10(values + 0.01)
            label = "log10(firing_rate + .01)"
        else:
            label = column
        hist, edges = np.histogram(values, bins=28)
        if hist.max() > 0:
            hist = hist / hist.max()
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist + index * 1.18, color=colors[index % len(colors)], linewidth=1.2)
        ax.text(0.99, index * 1.18, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("metric value")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)


def _draw_unit_qc_scatter(ax, metrics: pd.DataFrame) -> None:
    if metrics.empty:
        _draw_unavailable(ax, "Unit QC scatter", "quality metrics unavailable")
        return
    x_column, y_column = _scatter_columns(metrics)
    if x_column is None or y_column is None:
        _draw_unavailable(ax, "Unit QC scatter", "not enough metric columns")
        return
    x = pd.to_numeric(metrics[x_column], errors="coerce")
    y = pd.to_numeric(metrics[y_column], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        _draw_unavailable(ax, "Unit QC scatter", "metric values unavailable")
        return
    if "unitrefine_label" in metrics:
        label_values = metrics.loc[mask, "unitrefine_label"].fillna("unlabeled").astype(str)
        palette = _label_palette(label_values.unique())
        for label, group in label_values.groupby(label_values):
            indices = group.index
            ax.scatter(x.loc[indices], y.loc[indices], s=14, alpha=0.65, color=palette[label], label=label)
        ax.legend(loc="best", fontsize=6, frameon=False)
    else:
        ax.scatter(x[mask], y[mask], s=12, alpha=0.55, color="#334155")
    if x_column == "firing_rate":
        ax.set_xscale("symlog", linthresh=0.1)
    ax.set_title("Unit QC scatter")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(color="#e5e7eb", linewidth=0.7)


def _scatter_columns(metrics: pd.DataFrame) -> tuple[str | None, str | None]:
    candidates = [
        ("firing_rate", "snr"),
        ("firing_rate", "amplitude_median"),
        ("firing_rate", "presence_ratio"),
    ]
    return next(((x, y) for x, y in candidates if x in metrics and y in metrics), (None, None))


def _label_palette(labels) -> dict[str, str]:
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0f766e", "#4f46e5", "#64748b"]
    return {str(label): colors[index % len(colors)] for index, label in enumerate(sorted(map(str, labels)))}


def _draw_unit_locations(ax, analyzer) -> None:
    try:
        data = analyzer.get_extension("unit_locations").get_data()
        locations = _locations_to_array(data)
    except Exception as exc:
        _draw_unavailable(ax, "Unit locations", f"unit locations unavailable: {exc}")
        return
    if locations.size == 0 or locations.shape[1] < 2:
        _draw_unavailable(ax, "Unit locations", "unit locations unavailable")
        return
    ax.scatter(locations[:, 0], locations[:, 1], s=12, alpha=0.65, color="#7c3aed")
    ax.set_title("Unit locations")
    ax.set_xlabel("x um")
    ax.set_ylabel("depth um")
    ax.grid(color="#e5e7eb", linewidth=0.7)


def _locations_to_array(data) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        columns = [column for column in ["x", "y", "z"] if column in data]
        return data[columns].to_numpy(dtype=float) if columns else data.to_numpy(dtype=float)
    if isinstance(data, dict):
        values = list(data.values())
        return np.asarray(values, dtype=float)
    return np.asarray(data, dtype=float)


def _draw_example_templates(ax, analyzer, metrics: pd.DataFrame, max_units: int) -> None:
    try:
        unit_ids = list(analyzer.sorting.get_unit_ids())
        templates = analyzer.get_extension("templates").get_data()
        template_array = np.asarray(templates)
    except Exception as exc:
        _draw_unavailable(ax, "Example templates", f"templates unavailable: {exc}")
        return
    if not unit_ids or template_array.size == 0:
        _draw_unavailable(ax, "Example templates", "templates unavailable")
        return
    selected = _select_example_unit_indices(unit_ids, metrics, max_units)
    if not selected:
        _draw_unavailable(ax, "Example templates", "no units selected")
        return
    ax.set_title("Example templates")
    offsets = np.arange(len(selected)) * 3.0
    for row, unit_index in enumerate(selected):
        try:
            waveform = _template_trace(template_array, unit_index)
        except Exception:
            continue
        if waveform.size == 0:
            continue
        centered = waveform - np.nanmedian(waveform)
        scale = np.nanpercentile(np.abs(centered), 95)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        time = np.arange(centered.size)
        ax.plot(time, centered / scale + offsets[row], linewidth=1.0, label=str(unit_ids[unit_index]))
    ax.set_yticks(offsets)
    ax.set_yticklabels([str(unit_ids[index]) for index in selected], fontsize=7)
    ax.set_xlabel("sample")
    ax.set_ylabel("unit id")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)


def _select_example_unit_indices(unit_ids: list, metrics: pd.DataFrame, max_units: int) -> list[int]:
    if max_units <= 0:
        return []
    selected: list[int] = []
    if not metrics.empty:
        for column, ascending in [("snr", False), ("firing_rate", False), ("presence_ratio", True)]:
            if column not in metrics:
                continue
            ordered = pd.to_numeric(metrics[column], errors="coerce").sort_values(ascending=ascending).dropna()
            for unit_id in ordered.index:
                unit_id = _coerce_unit_id(unit_id, unit_ids)
                if unit_id in unit_ids:
                    index = unit_ids.index(unit_id)
                    if index not in selected:
                        selected.append(index)
                if len(selected) >= max_units:
                    return selected
    for index in np.linspace(0, len(unit_ids) - 1, min(max_units, len(unit_ids)), dtype=int):
        if int(index) not in selected:
            selected.append(int(index))
    return selected[:max_units]


def _coerce_unit_id(value, unit_ids: list):
    if value in unit_ids:
        return value
    for unit_id in unit_ids:
        if str(unit_id) == str(value):
            return unit_id
    try:
        as_int = int(value)
        if as_int in unit_ids:
            return as_int
    except Exception:
        pass
    return value


def _template_trace(template_array: np.ndarray, unit_index: int) -> np.ndarray:
    template = template_array[unit_index]
    if template.ndim == 1:
        return template
    if template.ndim == 2:
        peak_channel = int(np.nanargmax(np.nanmax(np.abs(template), axis=0)))
        return template[:, peak_channel]
    flattened = np.reshape(template, (template.shape[0], -1))
    peak_channel = int(np.nanargmax(np.nanmax(np.abs(flattened), axis=0)))
    return flattened[:, peak_channel]


def _draw_policy_box(
    ax,
    summary: dict[str, Any],
    channel_qc: pd.DataFrame,
    config: dict[str, Any],
    unitrefine_labels: pd.DataFrame | None = None,
) -> None:
    ax.axis("off")
    label_counts = _bad_label_counts(channel_qc)
    unitrefine_counts = _unitrefine_label_counts(unitrefine_labels if unitrefine_labels is not None else pd.DataFrame())
    unitrefine_probability = _unitrefine_probability_summary(unitrefine_labels if unitrefine_labels is not None else pd.DataFrame())
    preprocessing = config.get("preprocessing", {})
    sorting_params = config.get("sorting", {}).get("params", {})
    car_config = preprocessing.get("car", {})
    lines = [
        "QC notes",
        f"bad channel labels: {label_counts or 'none'}",
        f"phase_shift: {preprocessing.get('phase_shift', {}).get('enabled', True)}",
        f"SI CAR: {car_config.get('enabled', True)} ({car_config.get('operator', 'median')})",
        f"highpass: {preprocessing.get('highpass_filter', {}).get('freq_min', 300)} Hz",
        f"exclude bad from sorting: {preprocessing.get('bad_channels', {}).get('exclude_from_sorting', False)}",
        f"KS do_CAR: {sorting_params.get('do_CAR', False)}",
        f"KS drift correction: {sorting_params.get('do_correction', True)}",
        f"UnitRefine: {summary.get('automated_labels_status', 'n/a')}",
        f"UnitRefine labels: {unitrefine_counts or 'none'}",
        f"UnitRefine p: {unitrefine_probability or 'n/a'}",
        f"summary json: {Path(str(summary.get('quality_metrics_csv', ''))).parent / 'summary.json'}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        transform=ax.transAxes,
    )


def _bad_label_counts(channel_qc: pd.DataFrame) -> str:
    bad = _bad_channel_rows(channel_qc)
    if bad.empty:
        return ""
    column = "label" if "label" in bad else "status" if "status" in bad else ""
    if not column:
        return str(len(bad))
    counts = bad[column].astype(str).value_counts().to_dict()
    return ", ".join(f"{label}:{count}" for label, count in counts.items())


def _unitrefine_label_counts(labels: pd.DataFrame) -> str:
    if labels.empty or "unitrefine_label" not in labels:
        return ""
    counts = labels["unitrefine_label"].astype(str).value_counts().to_dict()
    return ", ".join(f"{label}:{int(count)}" for label, count in counts.items())


def _unitrefine_probability_summary(labels: pd.DataFrame) -> str:
    if labels.empty or "unitrefine_probability" not in labels:
        return ""
    probabilities = pd.to_numeric(labels["unitrefine_probability"], errors="coerce").dropna()
    if probabilities.empty:
        return ""
    return f"median={probabilities.median():.2f}, q25={probabilities.quantile(0.25):.2f}, q75={probabilities.quantile(0.75):.2f}"


def _safe_recording(analyzer):
    try:
        return analyzer.recording
    except Exception:
        return None


def _expanded_qc_html(metrics: pd.DataFrame, labels: pd.DataFrame) -> str:
    parts: list[str] = []
    if not labels.empty:
        counts = _unitrefine_label_counts(labels)
        probability = _unitrefine_probability_summary(labels)
        parts.append("<section><h2>UnitRefine labels</h2>")
        parts.append(f"<p>{html.escape(counts or 'No UnitRefine label counts available.')}</p>")
        if probability:
            parts.append(f"<p>{html.escape(probability)}</p>")
        parts.append("</section>")

    metric_summary = _metric_summary_table(metrics)
    if not metric_summary.empty:
        parts.append("<section><h2>Metric summary</h2>")
        parts.append(metric_summary.to_html(index=True, escape=True, float_format=lambda value: f"{value:.4g}"))
        parts.append("</section>")

    for title, table in _worst_units_tables(metrics).items():
        if table.empty:
            continue
        parts.append(f"<section><h2>{html.escape(title)}</h2>")
        parts.append(table.to_html(index=False, escape=True, float_format=lambda value: f"{value:.4g}"))
        parts.append("</section>")
    return "\n".join(parts)


def _metric_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in HIGH_YIELD_METRICS:
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        rows.append(
            {
                "metric": column,
                "n": int(finite.size),
                "median": _safe_metric_float(finite.median()),
                "q25": _safe_metric_float(finite.quantile(0.25)),
                "q75": _safe_metric_float(finite.quantile(0.75)),
                "min": _safe_metric_float(finite.min()),
                "max": _safe_metric_float(finite.max()),
                "nan_count": int(values.isna().sum()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("metric")


def _worst_units_tables(metrics: pd.DataFrame, limit: int = 8) -> dict[str, pd.DataFrame]:
    if metrics.empty:
        return {}
    rules = [
        ("High ISI violations", "isi_violations_ratio", False),
        ("High RP contamination", "rp_contamination", False),
        ("High sliding RP violation", "sliding_rp_violation", False),
        ("Low SNR", "snr", True),
        ("Low presence ratio", "presence_ratio", True),
        ("High amplitude cutoff", "amplitude_cutoff", False),
    ]
    tables: dict[str, pd.DataFrame] = {}
    for title, column, ascending in rules:
        if column not in metrics:
            continue
        table = metrics.copy()
        table["unit_id"] = table.index.astype(str)
        table[column] = pd.to_numeric(table[column], errors="coerce")
        table = table.dropna(subset=[column]).sort_values(column, ascending=ascending).head(limit)
        columns = ["unit_id", column]
        for optional in ["unitrefine_label", "unitrefine_probability", "firing_rate", "snr", "presence_ratio", "amplitude_cutoff"]:
            if optional in table and optional not in columns:
                columns.append(optional)
        tables[title] = table[columns]
    return tables


def _safe_metric_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _plot_probe_geometry(recording, channel_qc_csv: str | Path, output: Path) -> Path:
    locations = recording.get_channel_locations()
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(locations[:, 0], locations[:, 1], s=10, c="#3b82f6", label="channel")
    if Path(channel_qc_csv).exists():
        qc = pd.read_csv(channel_qc_csv)
        bad = _bad_channel_rows(qc)
        if not bad.empty:
            indices = bad["channel_index"].astype(int).to_numpy()
            ax.scatter(locations[indices, 0], locations[indices, 1], s=24, c="#dc2626", label="bad")
    ax.set_title("Probe geometry and detected bad channels")
    ax.set_xlabel("x um")
    ax.set_ylabel("y um")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_metric_histograms(metrics: pd.DataFrame, output: Path) -> Path:
    columns = [column for column in HIGH_YIELD_METRICS if column in metrics]
    if not columns:
        return output
    fig, axes = plt.subplots(len(columns), 1, figsize=(7, 2.3 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        ax.hist(pd.to_numeric(metrics[column], errors="coerce").dropna(), bins=40, color="#334155")
        ax.set_title(column)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_sorting_summary(analyzer, output: Path) -> Path:
    try:
        si.plot_sorting_summary(analyzer, backend="matplotlib")
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.01, 0.5, f"Sorting summary plot unavailable: {exc}", va="center")
        ax.axis("off")
        fig.savefig(output, dpi=140)
        plt.close(fig)
    return output


def find_reports(config: dict) -> list[dict[str, str]]:
    rows = read_registry(config["project"]["registry_csv"])
    reports: list[dict[str, str]] = []
    for row in rows:
        processed = Path(row.get("processed_folder", ""))
        report = processed / "report" / "summary.html"
        if report.exists():
            reports.append(
                {
                    "session_id": row.get("session_id", ""),
                    "mouse_id": row.get("mouse_id") or row.get("animal_id", ""),
                    "recording_date": row.get("recording_date", ""),
                    "task": row.get("task", ""),
                    "probe_label": row.get("probe_label", ""),
                    "status": row.get("status", ""),
                    "report": str(report),
                }
            )
    return reports


def write_summary_pngs_for_registry(config: dict, *, include_preprocessing_traces: bool = False) -> list[Path]:
    rows = read_registry(config["project"]["registry_csv"])
    written: list[Path] = []
    for row in rows:
        processed = Path(row.get("processed_folder", ""))
        analyzer_folder = Path(row.get("sorting_analyzer_folder", ""))
        if not processed.exists() or not analyzer_folder.exists():
            continue
        try:
            if include_preprocessing_traces:
                _backfill_preprocessing_traces(row, config, processed)
            analyzer = _load_analyzer_for_backfill(row, config, analyzer_folder)
            summary = _load_summary_for_row(row, processed)
            output = write_shareable_summary_png(
                analyzer=analyzer,
                probe_label=row.get("probe_label", ""),
                output_folder=processed,
                quality_metrics_csv=row.get("quality_metrics_csv") or processed / "quality_metrics.csv",
                channel_qc_csv=processed / "channel_qc.csv",
                summary=summary,
                config=config,
            )
            print(f"wrote {output}")
            written.append(output)
        except Exception as exc:
            print(f"skipped {row.get('session_id', processed)}: {exc}")
    return written


class _AnalyzerWithRecording:
    def __init__(self, analyzer, recording):
        self._analyzer = analyzer
        self.recording = recording
        self.sorting = analyzer.sorting

    def get_extension(self, name):
        return self._analyzer.get_extension(name)


def _load_analyzer_for_backfill(row: dict[str, str], config: dict, analyzer_folder: Path):
    try:
        analyzer = si.load_sorting_analyzer(analyzer_folder)
    except Exception:
        analyzer = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)
    try:
        _ = analyzer.recording
        return analyzer
    except Exception:
        pass
    try:
        recording, _probe_json = load_recording_with_probe(row, config)
        return _AnalyzerWithRecording(analyzer, recording)
    except Exception:
        return analyzer


def _load_summary_for_row(row: dict[str, str], processed: Path) -> dict[str, Any]:
    summary_json = processed / "summary.json"
    if summary_json.exists():
        try:
            return json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "session_id": row.get("session_id", ""),
        "probe_label": row.get("probe_label", ""),
        "raw_folder": row.get("raw_folder", ""),
        "stream_name": row.get("stream_name", ""),
        "num_units": "",
        "num_channels": row.get("n_channels", ""),
        "duration_seconds": row.get("duration_seconds", 0),
        "bad_channels_csv": str(processed / "channel_qc.csv"),
        "quality_metrics_csv": row.get("quality_metrics_csv", str(processed / "quality_metrics.csv")),
        "created_at": row.get("last_updated", ""),
    }


def _backfill_preprocessing_traces(row: dict[str, str], config: dict, processed: Path) -> None:
    recording, _probe_json = load_recording_with_probe(row, config)
    phase_config = config.get("preprocessing", {}).get("phase_shift", {})
    if phase_config.get("enabled", True):
        if _recording_has_property(recording, "inter_sample_shift"):
            recording = si.phase_shift(recording)
        else:
            raise RuntimeError("phase_shift enabled but recording has no inter_sample_shift property")
    pre_car_recording = recording
    car_config = config.get("preprocessing", {}).get("car", {})
    if car_config.get("enabled", True):
        recording = si.common_reference(
            recording,
            reference=car_config.get("reference", "global"),
            operator=car_config.get("operator", "median"),
        )
    trace_config = config.get("preprocessing", {}).get("trace_plots", {})
    png_config = config.get("qc", {}).get("shareable_png", {})
    write_preprocessing_trace_plots(
        pre_car_recording=pre_car_recording,
        post_car_recording=recording,
        output_folder=processed,
        start_seconds=float(trace_config.get("start_seconds", 0)),
        seconds=float(trace_config.get("seconds", 2)),
        max_channels=int(png_config.get("max_car_channels", trace_config.get("max_channels", 8))),
    )


def _recording_has_property(recording, property_name: str) -> bool:
    if hasattr(recording, "get_property_keys"):
        return property_name in recording.get_property_keys()
    try:
        return recording.get_property(property_name) is not None
    except Exception:
        return False


def print_reports(reports: list[dict[str, str]]) -> None:
    if not reports:
        print("No summary reports found yet.")
        return
    for index, report in enumerate(reports, start=1):
        print(f"{index}. {report['session_id']} | {report['probe_label']} | {report['status']}")
        print(f"   {report['report']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List or open SpikeInterface summary reports.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--open", action="store_true", dest="open_report", help="Open the newest report in the default browser.")
    parser.add_argument("--index", type=int, default=1, help="1-based report index to open after sorting newest first.")
    parser.add_argument("--write-summary-png", action="store_true", help="Backfill shareable summary PNGs for completed outputs.")
    parser.add_argument(
        "--include-preprocessing-traces",
        action="store_true",
        help="Reload raw recordings and regenerate pre/post CAR trace panels while backfilling summary PNGs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.write_summary_png:
        written = write_summary_pngs_for_registry(config, include_preprocessing_traces=args.include_preprocessing_traces)
        print(f"wrote {len(written)} summary PNG(s)")
        return

    reports = sorted(find_reports(config), key=lambda item: Path(item["report"]).stat().st_mtime, reverse=True)
    print_reports(reports)
    if args.open_report:
        if not reports:
            raise SystemExit("No reports available to open.")
        if args.index < 1 or args.index > len(reports):
            raise SystemExit(f"--index must be between 1 and {len(reports)}")
        os.startfile(reports[args.index - 1]["report"])


if __name__ == "__main__":
    main()
