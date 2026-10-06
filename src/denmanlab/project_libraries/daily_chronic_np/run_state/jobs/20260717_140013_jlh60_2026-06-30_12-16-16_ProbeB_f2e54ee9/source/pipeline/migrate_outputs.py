from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .config import load_config
from .output_paths import (
    output_paths_for_processed_folder,
    processed_folder_for_registry_row,
    smoke_processed_folder,
)
from .registry import read_registry, upsert_rows


def plan_migration(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_registry(config["project"]["registry_csv"])
    planned: list[dict[str, Any]] = []
    for row in rows:
        source = Path(row["processed_folder"])
        target_base = processed_folder_for_registry_row(row, config)
        if "_smoke_" in row["session_id"]:
            duration = _smoke_duration_from_session_id(row["session_id"])
            target = smoke_processed_folder(target_base, duration)
        else:
            target = target_base
        planned.append({"row": row, "source": source, "target": target})
    return planned


def migrate_outputs(config: dict[str, Any], *, dry_run: bool = True, force: bool = False) -> list[dict[str, Any]]:
    planned = plan_migration(config)
    migrated_rows: list[dict[str, Any]] = []
    for item in planned:
        row = dict(item["row"])
        source = item["source"]
        target = item["target"]
        print(f"{row['session_id']}:")
        print(f"  source: {source}")
        print(f"  target: {target}")

        if source.resolve() == target.resolve():
            print("  action: already recording-local")
            row.update(_updated_path_fields(row, target))
            migrated_rows.append(row)
            continue
        if not source.exists():
            raise FileNotFoundError(f"Source output folder does not exist: {source}")
        if target.exists():
            if not force:
                raise FileExistsError(f"Destination already exists: {target}")
            print("  action: destination exists; --force will replace it")
        else:
            print("  action: move")

        if dry_run:
            migrated_rows.append({**row, **_updated_path_fields(row, target)})
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and force:
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
        updated = {**row, **_updated_path_fields(row, target)}
        _rewrite_json_paths(target, old=str(source), new=str(target), row=updated)
        migrated_rows.append(updated)

    if not dry_run:
        upsert_rows(config["project"]["registry_csv"], migrated_rows)
    return migrated_rows


def _updated_path_fields(row: dict[str, Any], processed_folder: Path) -> dict[str, str]:
    paths = output_paths_for_processed_folder(processed_folder)
    probe_json = processed_folder / "metadata" / f"{row['probe_label']}_probeinterface.json"
    updated = {key: str(value) for key, value in paths.items()}
    updated["probeinterface_json"] = str(probe_json)
    return updated


def _rewrite_json_paths(output_folder: Path, *, old: str, new: str, row: dict[str, Any]) -> None:
    for name in ("manifest.json", "summary.json"):
        path = output_folder / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data = _replace_strings(data, old, new)
        if name == "manifest.json":
            data["probeinterface_json"] = row.get("probeinterface_json", data.get("probeinterface_json", ""))
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, old, new) for key, item in value.items()}
    return value


def _smoke_duration_from_session_id(session_id: str) -> float:
    suffix = session_id.rsplit("_smoke_", 1)[1]
    return float(suffix.rstrip("s"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Move completed outputs to the recording-local output layout.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--to-recording-local", action="store_true", help="Required explicit migration direction.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.to_recording_local:
        raise SystemExit("Pass --to-recording-local to confirm migration direction.")

    config = load_config(args.config)
    migrate_outputs(config, dry_run=args.dry_run, force=args.force)
    if args.dry_run:
        print("dry run only; no files moved and registry was not updated")
    else:
        print(f"updated {config['project']['registry_csv']}")


if __name__ == "__main__":
    main()
