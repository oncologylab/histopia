"""Lightweight runtime validation shared by workflow configuration."""

from __future__ import annotations

import math
from numbers import Integral, Real


def require_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    """Require one exact string from ``choices``."""

    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {expected}")


def require_bool(name: str, value: object) -> None:
    """Reject integer lookalikes for boolean controls."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def finite_float(name: str, value: object) -> float:
    """Return a finite float without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def positive_float(name: str, value: object) -> float:
    """Return a finite, strictly positive float."""

    normalized = finite_float(name, value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def positive_int(name: str, value: object) -> int:
    """Return a strictly positive integer without coercing floats."""

    normalized = integer(name, value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def nonnegative_int(name: str, value: object) -> int:
    """Return a non-negative integer without coercing floats."""

    normalized = integer(name, value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def integer(name: str, value: object) -> int:
    """Return an integer while rejecting booleans and fractional values."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)
