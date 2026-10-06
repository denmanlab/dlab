from __future__ import annotations

from pathlib import Path
from typing import Any


ANALYSIS_ROLE = "analysis"
ACQUISITION_ROLE = "acquisition"


def machine_role(config: dict[str, Any]) -> str:
    return str(config.get("machine", {}).get("role") or config.get("project", {}).get("machine_role") or ANALYSIS_ROLE).strip().lower()


def is_analysis_machine(config: dict[str, Any]) -> bool:
    return machine_role(config) == ANALYSIS_ROLE


def is_acquisition_machine(config: dict[str, Any]) -> bool:
    return machine_role(config) == ACQUISITION_ROLE


def discovery_root(config: dict[str, Any]) -> Path:
    project = config.get("project", {})
    staging = config.get("staging", {})
    if staging.get("enabled") and staging.get("server_raw_root"):
        return Path(staging["server_raw_root"])
    return Path(project["raw_root"])


def local_raw_root(config: dict[str, Any]) -> Path:
    staging = config.get("staging", {})
    if staging.get("enabled") and staging.get("local_raw_root"):
        return Path(staging["local_raw_root"])
    return Path(config["project"]["raw_root"])


def server_raw_root(config: dict[str, Any]) -> Path:
    staging = config.get("staging", {})
    if staging.get("server_raw_root"):
        return Path(staging["server_raw_root"])
    return Path(config.get("backup", {}).get("root", config["project"]["raw_root"]))


def local_path_for_server_session(server_session_folder: str | Path, config: dict[str, Any]) -> Path:
    server_folder = Path(server_session_folder)
    root = server_raw_root(config)
    try:
        relative = server_folder.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(server_folder.name)
    return local_raw_root(config) / relative


def server_path_for_local_session(local_session_folder: str | Path, config: dict[str, Any]) -> Path:
    local_folder = Path(local_session_folder)
    root = local_raw_root(config)
    try:
        relative = local_folder.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(local_folder.name)
    return server_raw_root(config) / relative


def raw_folder_for_discovered_session(session_folder: str | Path, config: dict[str, Any]) -> Path:
    if config.get("staging", {}).get("enabled"):
        return local_path_for_server_session(session_folder, config)
    return Path(session_folder)


def server_folder_for_discovered_session(session_folder: str | Path, config: dict[str, Any]) -> Path:
    if config.get("staging", {}).get("enabled"):
        return Path(session_folder)
    return server_path_for_local_session(session_folder, config)


def display_role(config: dict[str, Any]) -> str:
    role = machine_role(config)
    return role if role in {ANALYSIS_ROLE, ACQUISITION_ROLE} else "custom"
