from __future__ import annotations

import pytest

from histopia._vips_retry import (
    is_transient_vips_error,
    retry_transient_vips_read,
)


@pytest.mark.parametrize(
    "message",
    (
        "unable to write to memory",
        "openslide2vips: Operation not permitted",
        "read failed: stale file handle",
        "I/O error while reading a remote region",
    ),
)
def test_transient_vips_error_detection(message: str) -> None:
    assert is_transient_vips_error(RuntimeError(message))


def test_transient_vips_read_reopens_and_uses_bounded_backoff() -> None:
    calls = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("unable to write to memory: Operation not permitted")
        return "decoded"

    assert (
        retry_transient_vips_read(
            operation,
            attempts=3,
            initial_delay_seconds=0.1,
            sleep=delays.append,
        )
        == "decoded"
    )
    assert calls == 3
    assert delays == [0.1, 0.2]


def test_non_transient_vips_error_is_not_retried() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("slide geometry is invalid")

    with pytest.raises(RuntimeError, match="geometry"):
        retry_transient_vips_read(operation, sleep=lambda _: None)
    assert calls == 1


@pytest.mark.parametrize(
    ("attempts", "delay", "message"),
    ((0, 0.1, "attempts"), (1, -0.1, "initial_delay_seconds")),
)
def test_vips_retry_rejects_invalid_controls(
    attempts: int,
    delay: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        retry_transient_vips_read(
            lambda: None,
            attempts=attempts,
            initial_delay_seconds=delay,
        )
