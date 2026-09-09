"""Privacy-safe normalization for values written to portable traces."""

from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def safe_model_identifier(model: str) -> str:
    """Keep registry IDs while replacing local model paths with a neutral label."""
    expanded = Path(model).expanduser()
    if expanded.is_absolute() or model.startswith(("./", "../", "~")):
        return "local-model"
    if _WINDOWS_ABSOLUTE.match(model):
        return "local-model"
    return model
