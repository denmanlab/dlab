from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import load_config


PROCESSING_SECTIONS = (
    "recordings",
    "probes",
    "preprocessing",
    "sorting",
    "qc",
    "jobs",
    "progress",
)
PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def profiles_root(config_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    path = Path(config_path).resolve()
    config = config or load_config(path)
    configured = str(config.get("profiles", {}).get("root") or "").strip()
    if configured:
        root = Path(configured)
        return root if root.is_absolute() else (path.parent.parent / root).resolve()
    return (path.parent / "profiles").resolve()


def default_profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profiles", {}).get("default") or "default").strip() or "default"


def ensure_default_profile(config_path: str | Path) -> Path:
    config = load_config(config_path)
    root = profiles_root(config_path, config)
    root.mkdir(parents=True, exist_ok=True)
    name = default_profile_name(config)
    path = profile_path(config_path, name, config=config)
    if not path.exists():
        payload = {section: copy.deepcopy(config.get(section, {})) for section in PROCESSING_SECTIONS if section in config}
        payload["_profile"] = {
            "name": name,
            "description": "Default processing settings migrated from the analysis config.",
            "created_at": _now(),
            "updated_at": _now(),
            "version": 1,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def profile_path(config_path: str | Path, name: str, *, config: dict[str, Any] | None = None) -> Path:
    validate_profile_name(name)
    return profiles_root(config_path, config) / f"{name}.yaml"


def list_profiles(config_path: str | Path) -> list[dict[str, Any]]:
    ensure_default_profile(config_path)
    config = load_config(config_path)
    default_name = default_profile_name(config)
    rows: list[dict[str, Any]] = []
    for path in sorted(profiles_root(config_path, config).glob("*.yaml")):
        payload = _read_profile(path)
        metadata = dict(payload.get("_profile") or {})
        name = str(metadata.get("name") or path.stem)
        rows.append(
            {
                "name": name,
                "description": str(metadata.get("description") or ""),
                "version": int(metadata.get("version") or 1),
                "updated_at": str(metadata.get("updated_at") or ""),
                "path": str(path),
                "is_default": name == default_name,
                "hash": profile_hash(payload),
            }
        )
    return rows


def load_profile(config_path: str | Path, name: str) -> dict[str, Any]:
    ensure_default_profile(config_path)
    path = profile_path(config_path, name)
    if not path.exists():
        raise FileNotFoundError(f"Processing profile not found: {name}")
    payload = _read_profile(path)
    validate_profile_payload(payload)
    return payload


def resolve_profile(config_path: str | Path, name: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_config(config_path)
    selected = name or default_profile_name(base)
    profile = load_profile(config_path, selected)
    effective = copy.deepcopy(base)
    for section in PROCESSING_SECTIONS:
        if section in profile:
            effective[section] = copy.deepcopy(profile[section])
    metadata = dict(profile.get("_profile") or {})
    metadata.update({"name": selected, "hash": profile_hash(profile)})
    return effective, metadata


def save_profile(
    config_path: str | Path,
    name: str,
    payload: dict[str, Any],
    *,
    description: str | None = None,
) -> dict[str, Any]:
    validate_profile_name(name)
    validate_profile_payload(payload)
    path = profile_path(config_path, name)
    existing = _read_profile(path) if path.exists() else {}
    old_meta = dict(existing.get("_profile") or {})
    clean = {section: copy.deepcopy(payload[section]) for section in PROCESSING_SECTIONS if section in payload}
    clean["_profile"] = {
        "name": name,
        "description": description if description is not None else str(old_meta.get("description") or ""),
        "created_at": str(old_meta.get("created_at") or _now()),
        "updated_at": _now(),
        "version": int(old_meta.get("version") or 0) + 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    return {
        "name": name,
        "path": str(path),
        "version": clean["_profile"]["version"],
        "hash": profile_hash(clean),
        "profile": clean,
    }


def profile_hash(payload: dict[str, Any]) -> str:
    clean = copy.deepcopy(payload)
    metadata = clean.get("_profile")
    if isinstance(metadata, dict):
        metadata.pop("updated_at", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_profile_name(name: str) -> None:
    if not PROFILE_NAME_PATTERN.fullmatch(str(name or "")):
        raise ValueError("Profile name must use letters, numbers, period, underscore, or dash.")


def validate_profile_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Profile must be a YAML mapping.")
    unknown = set(payload) - set(PROCESSING_SECTIONS) - {"_profile"}
    if unknown:
        raise ValueError(f"Unsupported profile section(s): {', '.join(sorted(unknown))}")
    for section in PROCESSING_SECTIONS:
        if section in payload and not isinstance(payload[section], dict):
            raise ValueError(f"Profile section '{section}' must be a mapping.")


def _read_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Profile must be a YAML mapping: {path}")
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
