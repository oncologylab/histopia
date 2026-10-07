from histopia.visualization._review_theme import (
    REVIEW_THEME_CSS,
    REVIEW_THEME_MARKER,
    themed_review_css,
)


def test_review_theme_matches_reference_design_contract() -> None:
    for declaration in (
        "--hp-bg: #f3f6fa",
        "--hp-surface: #ffffff",
        "--hp-text: #111827",
        "--hp-accent: #2563eb",
        "--hp-accent-hover: #1d4ed8",
        "--hp-focus: #2563eb",
        "--hp-active: #1d4ed8",
        "--hp-radius: 8px",
        '--hp-font: "Source Sans 3", "Source Sans Pro"',
        "font-family: var(--hp-font)",
    ):
        assert declaration in REVIEW_THEME_CSS
    for component in (
        ".rail, #queue-panel, #evidence, .context",
        "#queue button.active, .slide-row.active",
        ".evidence-section, fieldset, .metrics, .card, .metric",
        ".histopia-focus-toolbar, .histopia-focus-status",
        "#registration-feedback",
        '.segments button[aria-pressed="true"]',
        ".app aside > section",
        ".app .metric",
        "#decision",
    ):
        assert component in REVIEW_THEME_CSS


def test_review_theme_application_replaces_an_older_theme() -> None:
    initial = themed_review_css(".viewer { overflow: hidden; }")
    refreshed = themed_review_css(initial)

    assert refreshed == initial
    assert refreshed.count(REVIEW_THEME_MARKER) == 1
    assert refreshed.startswith(".viewer { overflow: hidden; }")
