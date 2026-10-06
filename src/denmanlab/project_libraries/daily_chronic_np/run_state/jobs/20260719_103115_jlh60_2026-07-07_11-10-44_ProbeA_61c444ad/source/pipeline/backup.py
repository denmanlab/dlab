from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .registry import read_registry, update_row


@dataclass(frozen=True)
class BackupResult:
    ok: bool
    source: Path
    target: Path
    checked_files: int
    missing_files: list[str]
    size_mismatches: list[str]
    error: str = ""

    @property
    def status(self) -> str:
        return "verified" if self.ok else "failed"


def verify_raw_backup(raw_folder: str | Path, config: dict[str, Any]) -> BackupResult:
    source = Path(raw_folder)
    target = backup_folder_for_raw(source, config)
    if not source.exists():
        return BackupResult(False, source, target, 0, [], [], f"source raw folder does not exist: {source}")
    if not target.exists():
        return BackupResult(False, source, target, 0, [], [], f"backup folder does not exist: {target}")

    source_files = _file_size_map(source, config)
    target_files = _file_size_map(target, config)
    missing = sorted(str(path) for path in source_files if path not in target_files)
    mismatched = sorted(
        f"{path} source={source_files[path]} backup={target_files[path]}"
        for path in source_files
        if path in target_files and source_files[path] != target_files[path]
    )
    ok = not missing and not mismatched
    error = ""
    if missing:
        error += f"{len(missing)} missing backup file(s)"
    if mismatched:
        error += ("; " if error else "") + f"{len(mismatched)} size mismatch(es)"
    return BackupResult(ok, source, target, len(source_files), missing, mismatched, error)


def backup_folder_for_raw(raw_folder: str | Path, config: dict[str, Any]) -> Path:
    raw_path = Path(raw_folder)
    backup_root = Path(config.get("backup", {}).get("root", ""))
    raw_root = Path(config["project"]["raw_root"])
    try:
        relative = raw_path.resolve().relative_to(raw_root.resolve())
    except ValueError:
        relative = Path(raw_path.name)
    return backup_root / relative


def verify_registry_row_backup(row: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    result = verify_raw_backup(row["raw_folder"], config)
    return {
        "backup_status": result.status,
        "backup_folder": str(result.target),
        "backup_checked_at": datetime.now(timezone.utc).isoformat(),
        "backup_error": result.error,
    }


def require_backup_verified(row: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    backup_config = config.get("backup", {})
    if not backup_config.get("enabled", False) or not backup_config.get("require_before_processing", False):
        return {"backup_status": "not_required", "backup_folder": "", "backup_checked_at": "", "backup_error": ""}
    if backup_config.get("mode", "verify_only") != "verify_only":
        raise ValueError("Only backup.mode='verify_only' is currently implemented")
    fields = verify_registry_row_backup(row, config)
    if fields["backup_status"] != "verified":
        raise RuntimeError(f"Raw backup verification failed: {fields['backup_error']} ({fields['backup_folder']})")
    return fields


def print_backup_result(result: BackupResult, *, verbose: bool = False) -> None:
    print(f"source: {result.source}")
    print(f"backup: {result.target}")
    print(f"status: {result.status}")
    print(f"checked_files: {result.checked_files}")
    if result.error:
        print(f"error: {result.error}")
    if verbose:
        for path in result.missing_files[:100]:
            print(f"missing: {path}")
        for item in result.size_mismatches[:100]:
            print(f"size_mismatch: {item}")


def _file_size_map(folder: Path, config: dict[str, Any]) -> dict[Path, int]:
    sizes: dict[Path, int] = {}
    for path in folder.rglob("*"):
        if _excluded(path, folder, config):
            continue
        if path.is_file():
            sizes[path.relative_to(folder)] = path.stat().st_size
    return sizes


def _excluded(path: Path, root: Path, config: dict[str, Any]) -> bool:
    prefixes = tuple(config.get("backup", {}).get("exclude_dir_prefixes", ["spikeinterface_output"]))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(prefixes) for part in relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify raw recordings are backed up to the configured server path.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--raw-folder")
    parser.add_argument("--all", action="store_true", help="Verify every unique raw folder in the registry.")
    parser.add_argument("--update-registry", action="store_true", help="Write backup status fields to matching registry rows.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.all:
        rows = read_registry(config["project"]["registry_csv"])
        raw_folders = sorted({row["raw_folder"] for row in rows if row.get("raw_folder")})
    elif args.raw_folder:
        rows = []
        raw_folders = [args.raw_folder]
    else:
        raise SystemExit("Pass --raw-folder RAW_FOLDER or --all")

    by_raw = {}
    for raw_folder in raw_folders:
        result = verify_raw_backup(raw_folder, config)
        print_backup_result(result, verbose=args.verbose)
        by_raw[str(Path(raw_folder).resolve())] = result
        if not result.ok:
            raise SystemExit(1)

    if args.update_registry:
        if not args.all:
            raise SystemExit("--update-registry requires --all")
        for row in rows:
            result = by_raw.get(str(Path(row["raw_folder"]).resolve()))
            if result is None:
                continue
            row.update(
                {
                    "backup_status": result.status,
                    "backup_folder": str(result.target),
                    "backup_checked_at": datetime.now(timezone.utc).isoformat(),
                    "backup_error": result.error,
                }
            )
            update_row(config["project"]["registry_csv"], row)


if __name__ == "__main__":
    main()
