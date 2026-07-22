from __future__ import annotations

from histopia import visualization
from histopia.registration import build_section_order_review as legacy_order_review
from histopia.registration import build_section_viewer as legacy_viewer


def test_visualization_is_canonical_viewer_api() -> None:
    assert visualization.build_section_viewer is legacy_viewer
    assert visualization.build_section_order_review is legacy_order_review
    assert visualization.MAX_DISPLAY_LINKS == 500
