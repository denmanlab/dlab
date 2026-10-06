from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import spikeinterface.full as si
from spikeinterface.curation import unitrefine_label_units

from .config import load_config
from .registry import find_row, read_registry


DEFAULT_NOISE_NEURAL_CLASSIFIER = "SpikeInterface/UnitRefine_noise_neural_classifier"
DEFAULT_SUA_MUA_CLASSIFIER = "SpikeInterface/UnitRefine_sua_mua_classifier"


def run_unitrefine_for_analyzer(
    analyzer,
    processed_folder: str | Path,
    config: dict[str, Any],
    *,
    label_set: str | None = None,
    noise_neural_classifier: str | None = None,
    sua_mua_classifier: str | None = None,
    overwrite: bool = False,
    fail_on_error: bool | None = None,
) -> dict[str, Any]:
    label_config = unitrefine_config(
        config,
        label_set=label_set,
        noise_neural_classifier=noise_neural_classifier,
        sua_mua_classifier=sua_mua_classifier,
        fail_on_error=fail_on_error,
    )
    if not label_config.get("enabled", False):
        return {"status": "disabled", "label_set": label_config["label_set"], "unit_labels_csv": "", "summary_json": ""}

    output_folder = unitrefine_output_folder(processed_folder, label_config["label_set"])
    labels_csv = output_folder / "unit_labels.csv"
    summary_json = output_folder / "summary.json"
    if output_folder.exists() and not overwrite:
        raise FileExistsError(f"UnitRefine label set already exists: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        labels = unitrefine_label_units(
            sorting_analyzer=analyzer,
            noise_neural_classifier=label_config.get("noise_neural_classifier") or None,
            sua_mua_classifier=label_config.get("sua_mua_classifier") or None,
        )
        labels.to_csv(labels_csv, index_label="unit_id")
        summary = unitrefine_summary(labels, label_config)
        summary.update({"status": "complete", "unit_labels_csv": str(labels_csv), "summary_json": str(summary_json)})
        summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary
    except Exception as exc:
        error_message = _format_exception(exc)
        summary = {
            "status": "failed",
            "label_set": label_config["label_set"],
            "method": "unitrefine",
            "unit_labels_csv": "",
            "summary_json": str(summary_json),
            "error": error_message,
            "hint": _unitrefine_failure_hint(error_message),
            "config": label_config,
            "created_at": _now(),
        }
        summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        if label_config.get("fail_on_error", False):
            raise
        return summary


def unitrefine_config(
    config: dict[str, Any],
    *,
    label_set: str | None = None,
    noise_neural_classifier: str | None = None,
    sua_mua_classifier: str | None = None,
    fail_on_error: bool | None = None,
) -> dict[str, Any]:
    base = dict(config.get("qc", {}).get("automated_labels", {}))
    base.setdefault("enabled", False)
    base.setdefault("method", "unitrefine")
    base["label_set"] = _sanitize_label_set(label_set or base.get("label_set", "unitrefine_full"))
    base["noise_neural_classifier"] = (
        noise_neural_classifier
        if noise_neural_classifier is not None
        else base.get("noise_neural_classifier", DEFAULT_NOISE_NEURAL_CLASSIFIER)
    )
    base["sua_mua_classifier"] = (
        sua_mua_classifier
        if sua_mua_classifier is not None
        else base.get("sua_mua_classifier", DEFAULT_SUA_MUA_CLASSIFIER)
    )
    if fail_on_error is not None:
        base["fail_on_error"] = fail_on_error
    base.setdefault("fail_on_error", False)
    if str(base.get("method", "unitrefine")).lower() != "unitrefine":
        raise ValueError(f"Unsupported automated label method: {base.get('method')}")
    if not base.get("noise_neural_classifier") and not base.get("sua_mua_classifier"):
        raise ValueError("UnitRefine requires noise_neural_classifier and/or sua_mua_classifier")
    return base


def unitrefine_summary(labels: pd.DataFrame, label_config: dict[str, Any]) -> dict[str, Any]:
    label_column = "unitrefine_label" if "unitrefine_label" in labels else "label"
    probability_column = "unitrefine_probability" if "unitrefine_probability" in labels else "probability"
    counts = labels[label_column].astype(str).value_counts().to_dict() if label_column in labels else {}
    probabilities = pd.to_numeric(labels.get(probability_column, pd.Series(dtype=float)), errors="coerce")
    return {
        "label_set": label_config["label_set"],
        "method": "unitrefine",
        "classifiers": {
            "noise_neural_classifier": label_config.get("noise_neural_classifier"),
            "sua_mua_classifier": label_config.get("sua_mua_classifier"),
        },
        "num_units": int(len(labels)),
        "label_counts": {str(label): int(count) for label, count in counts.items()},
        "probability": {
            "median": _safe_float(probabilities.median()),
            "q25": _safe_float(probabilities.quantile(0.25)),
            "q75": _safe_float(probabilities.quantile(0.75)),
            "min": _safe_float(probabilities.min()),
            "max": _safe_float(probabilities.max()),
            "nan_count": int(probabilities.isna().sum()),
        },
        "config": label_config,
        "created_at": _now(),
    }


def run_unitrefine_for_processed_folder(
    processed_folder: str | Path,
    config: dict[str, Any],
    *,
    label_set: str | None = None,
    noise_neural_classifier: str | None = None,
    sua_mua_classifier: str | None = None,
    overwrite: bool = False,
    fail_on_error: bool | None = None,
) -> dict[str, Any]:
    processed = Path(processed_folder)
    analyzer = si.load_sorting_analyzer(processed / "sorting_analyzer")
    return run_unitrefine_for_analyzer(
        analyzer,
        processed,
        config,
        label_set=label_set,
        noise_neural_classifier=noise_neural_classifier,
        sua_mua_classifier=sua_mua_classifier,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
    )


def resolve_rows_for_cli(args, config: dict[str, Any]) -> list[dict[str, str]]:
    if args.processed_folder:
        processed = Path(args.processed_folder)
        return [
            {
                "session_id": processed.name,
                "probe_label": "",
                "processed_folder": str(processed),
                "sorting_analyzer_folder": str(processed / "sorting_analyzer"),
            }
        ]
    if args.session_id:
        return [find_row(config["project"]["registry_csv"], session_id=args.session_id)]
    if args.all_complete:
        return [row for row in read_registry(config["project"]["registry_csv"]) if row.get("status") == "complete"]
    raise ValueError("Choose one of --session-id, --processed-folder, or --all-complete")


def unitrefine_output_folder(processed_folder: str | Path, label_set: str) -> Path:
    return Path(processed_folder) / "unitrefine" / _sanitize_label_set(label_set)


def _sanitize_label_set(label_set: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label_set).strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("label_set must contain at least one letter or number")
    return sanitized


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_exception(exc: Exception) -> str:
    if getattr(exc, "args", None):
        return " | ".join(str(arg) for arg in exc.args)
    return str(exc)


def _unitrefine_failure_hint(error_message: str) -> str:
    if "Missing metrics" in error_message or "required metrics" in error_message:
        return (
            "UnitRefine requires a SortingAnalyzer with template_metrics, quality_metrics, "
            "principal_components, spike_locations-derived drift/PCA metrics, and the expanded "
            "qc.quality_metrics.unitrefine_metric_names feature set. Future full/restart QC runs "
            "compute those before labeling; older analyzers may need quality_metrics recomputed."
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UnitRefine labels on existing SortingAnalyzer outputs.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--session-id")
    target.add_argument("--processed-folder")
    target.add_argument("--all-complete", action="store_true")
    parser.add_argument("--label-set", default=None)
    parser.add_argument("--noise-neural-classifier")
    parser.add_argument("--sua-mua-classifier")
    parser.add_argument("--overwrite-label-set", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = resolve_rows_for_cli(args, config)
    for row in rows:
        processed = Path(row["processed_folder"])
        print(f"{row.get('session_id', processed.name)} | {row.get('probe_label', '')} | {processed}")
        result = run_unitrefine_for_processed_folder(
            processed,
            config,
            label_set=args.label_set,
            noise_neural_classifier=args.noise_neural_classifier,
            sua_mua_classifier=args.sua_mua_classifier,
            overwrite=args.overwrite_label_set,
            fail_on_error=True if args.fail_on_error else None,
        )
        print(f"  {result['status']} | {result.get('unit_labels_csv', '') or result.get('error', '')}")


if __name__ == "__main__":
    main()
