"""Registration-specific exceptions."""


class OptionalDependencyError(ImportError):
    """Raised when an optional workflow dependency is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"{package!r} is required for this workflow. "
            f"Install Histopia with the {extra!r} extra."
        )
