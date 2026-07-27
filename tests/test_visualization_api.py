from __future__ import annotations

from histopia import visualization


def test_visualization_is_canonical_viewer_api() -> None:
    assert callable(visualization.build_section_viewer)
    assert callable(visualization.build_section_order_review)


def test_registration_does_not_reexport_visualization_api() -> None:
    from histopia import registration

    assert not hasattr(registration, "build_section_viewer")
    assert not hasattr(registration, "build_section_order_review")
