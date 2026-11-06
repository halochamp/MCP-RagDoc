"""Filesystem truth shared by discovery consumers; never resolve away a link."""
import os
from pathlib import Path

from helpers.sensitive_paths import is_sensitive_path


def visible_source_file(path: str | Path, root: Path) -> Path | None:
    try:
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        for base in (Path(os.path.abspath(root.expanduser())), root.resolve()):
            try:
                relative = candidate.relative_to(base)
            except ValueError:
                continue
            if not relative.parts or any(part.startswith(".") for part in relative.parts):
                return None
            current = base
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    return None
            if candidate.is_file() and not is_sensitive_path(candidate):
                return candidate
            return None
    except (OSError, ValueError, RuntimeError):
        pass
    return None
