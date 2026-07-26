from __future__ import annotations

import signal

import pytest

from histopia._signals import graceful_sigterm


def test_graceful_sigterm_translates_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = object()
    installed: list[tuple[signal.Signals, object]] = []

    monkeypatch.setattr(signal, "getsignal", lambda _signal: previous)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda selected, handler: installed.append((selected, handler)),
    )

    with graceful_sigterm():
        handler = installed[-1][1]
        assert callable(handler)
        with pytest.raises(SystemExit) as raised:
            handler(signal.SIGTERM, None)

    assert raised.value.code == 128 + signal.SIGTERM
    assert installed[-1] == (signal.SIGTERM, previous)
