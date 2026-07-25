"""Small atomic file writers for durable workflow artifacts."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def write_text_atomic(path: Path | str, text: str) -> Path:
    """Replace a text file only after its complete contents reach disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_temporary(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_json_atomic(
    path: Path | str,
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> Path:
    """Serialize JSON and atomically replace the destination."""

    text = json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        separators=separators,
    )
    return write_text_atomic(path, text + "\n")


def write_binary_atomic(
    path: Path | str,
    writer: Callable[[BinaryIO], None],
) -> Path:
    """Atomically replace a binary file produced by ``writer``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_temporary(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _replace_temporary(temporary: Path, target: Path) -> None:
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = 0o644
    temporary.chmod(mode)
    os.replace(temporary, target)
