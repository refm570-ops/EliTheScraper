from __future__ import annotations

import os

import structlog

log = structlog.get_logger()


def resolve_config_path(path: str) -> str:
    """Resolve a config path, preferring the operator's local file.

    Target configs (which groups/accounts to monitor) are gitignored so real
    IDs never reach the repo. Only the ``.example.yml`` placeholders are
    tracked. Falls back to the example so a fresh clone still runs, but warns
    loudly, since running on placeholders monitors nothing real.
    """
    if os.path.exists(path):
        return path

    example = path.replace(".yml", ".example.yml")
    if os.path.exists(example):
        log.warning("config.using_example", requested=path, fallback=example)
        return example

    raise FileNotFoundError(f"No config found at {path} or {example}")
