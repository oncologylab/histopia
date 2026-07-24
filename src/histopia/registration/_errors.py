"""Registration-specific exceptions."""

from __future__ import annotations

from pathlib import Path


class OptionalDependencyError(ImportError):
    """Raised when an optional workflow dependency is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"{package!r} is required for this workflow. "
            f"Install Histopia with the {extra!r} extra."
        )


class RegistrationApprovalRequired(RuntimeError):
    """Raised after preparing an artifact that requires human review."""

    def __init__(
        self,
        stage: str,
        review_path: Path | str,
        *,
        pending_slides: tuple[str, ...] = (),
    ) -> None:
        self.stage = stage
        self.review_path = Path(review_path)
        self.pending_slides = pending_slides
        if stage == "masks":
            detail = ", ".join(pending_slides)
            message = f"registration requires approved masks: {detail}"
        elif stage == "order":
            message = (
                "registration requires approval of the current section order: "
                f"{self.review_path}"
            )
        else:
            message = f"registration requires approval of {stage}: {self.review_path}"
        super().__init__(message)

    def to_json_dict(self) -> dict[str, object]:
        """Return a path-explicit status payload for local workflow clients."""

        return {
            "status": "review_required",
            "stage": self.stage,
            "review_path": str(self.review_path),
            "pending_slides": list(self.pending_slides),
        }
