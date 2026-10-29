import os
from typing import Optional

import yaml

DEFAULT_PATH = "/etc/aries/core_allocation.yaml"


def resolve_path(cli_override: Optional[str]) -> str:
    tried = []
    if cli_override:
        if os.path.exists(cli_override):
            return cli_override
        tried.append(f"cli override: {cli_override}")
    env_path = os.environ.get("ARIES_CORE_ALLOCATION_PATH")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        tried.append(f"ARIES_CORE_ALLOCATION_PATH: {env_path}")
    if os.path.exists(DEFAULT_PATH):
        return DEFAULT_PATH
    tried.append(f"default: {DEFAULT_PATH}")
    # Repo-relative fallback anchored to this file's location so `python src/server.py`
    # from backend_vlm/ works regardless of CWD.
    fallback = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "core_allocation.yaml")
    )
    if os.path.exists(fallback):
        return fallback
    tried.append(f"repo-relative fallback: {fallback}")
    raise FileNotFoundError(
        "core_allocation.yaml not found. Tried: ["
        + ", ".join(tried)
        + f"]. Set ARIES_CORE_ALLOCATION_PATH or place the file at {DEFAULT_PATH}."
    )


def load_core_allocation(path: str, category: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    vlm_section = data.get("vlm")
    if not isinstance(vlm_section, dict):
        raise KeyError(f"core_allocation.yaml at {path} is missing 'vlm' section")

    entry = vlm_section.get(category)
    if entry is None:
        raise KeyError(f"core_allocation.yaml at {path} has no vlm entry for category '{category}'")

    return entry
