from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    _expand_project_paths(config)
    return config


def _expand_project_paths(config: dict[str, Any]) -> None:
    project = config.setdefault("project", {})
    for key in ("raw_root", "processed_root", "registry_csv"):
        if key in project and project[key] is not None and str(project[key]).strip():
            project[key] = str(Path(project[key]).expanduser())
    for section_name, keys in {
        "staging": ("server_raw_root", "local_raw_root"),
        "acquisition": ("local_raw_root", "server_raw_root"),
        "backup": ("root",),
    }.items():
        section = config.get(section_name, {})
        for key in keys:
            if key in section and section[key] is not None and str(section[key]).strip():
                section[key] = str(Path(section[key]).expanduser())
    derived = config.get("backup", {}).get("derived_outputs", {})
    for key in ("local_archive_root",):
        if key in derived and derived[key] is not None and str(derived[key]).strip():
            derived[key] = str(Path(derived[key]).expanduser())
    full_archive = config.get("backup", {}).get("local_recording_archive", {})
    for key in ("root",):
        if key in full_archive and full_archive[key] is not None and str(full_archive[key]).strip():
            full_archive[key] = str(Path(full_archive[key]).expanduser())
    nwb = config.get("mouse_arena_nwb", {})
    for key in ("backup_root",):
        if key in nwb and nwb[key] is not None and str(nwb[key]).strip():
            nwb[key] = str(Path(nwb[key]).expanduser())


