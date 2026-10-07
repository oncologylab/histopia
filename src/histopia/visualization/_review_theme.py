"""Shared visual language for Histopia's standalone review applications.

The review tools have different scientific layouts, but they deliberately share
one visual system.  Image canvases keep the space their task requires while all
surrounding navigation, controls, cards, text, and review rails use the same
compact light theme.
"""

from __future__ import annotations

REVIEW_THEME_MARKER = "/* Histopia review theme:"

REVIEW_THEME_CSS = r"""
/* Histopia review theme: one compact, readable system for every review page. */
:root {
  color-scheme: light;
  --hp-bg: #f3f6fa;
  --hp-surface: #ffffff;
  --hp-surface-soft: #f9fafb;
  --hp-border: #cfd8e3;
  --hp-border-soft: #dbe3ec;
  --hp-text: #111827;
  --hp-muted: #4b5563;
  --hp-subtle: #64748b;
  --hp-accent: #2563eb;
  --hp-accent-hover: #1d4ed8;
  --hp-accent-soft: #eaf2ff;
  --hp-focus: #2563eb;
  --hp-active: #1d4ed8;
  --hp-active-mark: #72e0b2;
  --hp-success: #176b45;
  --hp-warning: #945b08;
  --hp-danger: #9b2c2c;
  --hp-nav: #0f172a;
  --hp-nav-hover: #1e293b;
  --hp-nav-border: #334155;
  --hp-nav-text: #e5edf6;
  --hp-radius: 8px;
  --hp-control-height: 34px;
  --hp-shadow: 0 8px 24px rgba(15, 23, 42, .07);
  --hp-font: "Source Sans 3", "Source Sans Pro", -apple-system,
    BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-family: var(--hp-font);
  color: var(--hp-text);
  background: var(--hp-bg);
}
html, body {
  font-family: var(--hp-font);
  font-size: 14px;
  line-height: 1.4;
  color: var(--hp-text);
  background-color: var(--hp-bg);
}
*, *::before, *::after { box-sizing: border-box; }
body { accent-color: var(--hp-focus); }
header, .topbar, .section-head {
  color: var(--hp-text);
  background-color: var(--hp-surface);
  border-color: var(--hp-border-soft);
}
header, .topbar {
  min-width: 0;
  padding-inline: 14px;
}
header > strong, .topbar .brand {
  font-weight: 800;
}
aside, .rail, .metrics, .evidence-section, figure, form, fieldset,
.card, .metric, .fp-card, .run-card {
  border-color: var(--hp-border-soft);
}
aside, .metrics, .evidence-section, figure, fieldset, .card, .metric,
.fp-card, .run-card {
  color: var(--hp-text);
  background-color: var(--hp-surface);
}
.rail, #queue-panel, #evidence, .context {
  color: var(--hp-text);
  background-color: var(--hp-surface);
  border-color: var(--hp-border-soft);
}
.rail-head, .queue-toolbar, #progress, .section-head, .context-title,
#neighbor-label, figcaption, dl div, .probe-row, #annotations li,
.evidence-section {
  border-color: var(--hp-border-soft);
}
h1, h2, h3, strong, legend {
  color: var(--hp-text);
  letter-spacing: 0;
}
h1 { font-size: 1.45rem; line-height: 1.2; }
h2 { font-size: 1.15rem; line-height: 1.25; }
h3, legend { font-size: 1rem; line-height: 1.3; }
h1, h2, h3 { margin-top: 0; }
label, dt, figcaption, .status, .counter { font-size: .9rem; }
button, select, input, textarea, a[role="button"] {
  font-family: inherit;
  font-size: .9rem;
  color: var(--hp-text);
  border-color: var(--hp-border);
  border-radius: var(--hp-radius);
  background-color: var(--hp-surface);
  box-shadow: none;
}
button, select, input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
a[role="button"] {
  min-height: var(--hp-control-height);
}
button, a[role="button"] { font-weight: 700; }
button:disabled, a[role="button"][aria-disabled="true"] {
  color: var(--hp-subtle);
  border-color: var(--hp-border-soft);
  background: #e5e7eb;
  cursor: not-allowed;
  opacity: 1;
}
textarea { line-height: 1.4; padding: 8px 10px; }
select, input:not([type="checkbox"]):not([type="radio"]):not([type="range"]) {
  padding-inline: 10px;
}
button:hover, a[role="button"]:hover {
  color: var(--hp-text);
  border-color: #94a3b8;
  background-color: var(--hp-surface-soft);
}
button:focus-visible, select:focus-visible, input:focus-visible,
textarea:focus-visible, a:focus-visible {
  outline: 3px solid rgba(37, 99, 235, .22);
  outline-offset: 1px;
}
button.primary, .primary, #accept, #approve, #connect {
  color: #fff;
  border-color: var(--hp-accent);
  background-color: var(--hp-accent);
  font-weight: 700;
}
button.primary:hover, .primary:hover, #accept:hover, #approve:hover,
#connect:hover {
  color: #fff;
  border-color: var(--hp-accent-hover);
  background-color: var(--hp-accent-hover);
}
.segmented button[aria-pressed="true"], .segmented button.active,
.segments button[aria-pressed="true"], .segments button.active,
.tools button.active,
.queue-toolbar button[aria-pressed="true"], #queue button.active,
nav button[aria-pressed="true"] {
  color: #fff;
  border-color: var(--hp-focus);
  background-color: var(--hp-active);
}
#queue button.active, .slide-row.active {
  color: #fff;
  background: var(--hp-active);
  box-shadow: inset 4px 0 0 var(--hp-active-mark);
}
#queue, .slide-list { padding: 6px; }
#queue button, .slide-row {
  margin: 0 0 3px;
  border-bottom-color: transparent;
  border-radius: var(--hp-radius);
}
#queue button:hover:not(.active), .slide-row:hover, #annotations li:hover {
  background: var(--hp-surface-soft);
}
#queue button.active .order, #queue button.active small,
.slide-row.active .order, .slide-row.active .name {
  color: #e5edf6;
}
nav button[aria-pressed="true"].approved::after {
  color: #e5edf6;
}
fieldset, .card, .metric, .fp-card, .run-card {
  border-color: var(--hp-border-soft);
  border-radius: var(--hp-radius);
}
figure, .metrics {
  border-radius: var(--hp-radius);
  overflow: hidden;
}
fieldset {
  background-color: var(--hp-surface);
  padding: 10px 12px;
}
legend { padding-inline: 4px; font-weight: 800; }
.evidence-section {
  margin: 8px;
  border: 1px solid var(--hp-border-soft);
  border-radius: var(--hp-radius);
}
.evidence-section, fieldset, .metrics, .card, .metric, .fp-card, .run-card {
  box-shadow: var(--hp-shadow);
}
.segmented, .segments { gap: 4px; }
.segmented button, .segmented button:first-child, .segmented button:last-child,
.segments button, .segments button:first-child, .segments button:last-child,
.icon-button, .tools button, .zoom-tools button {
  margin-left: 0;
  border-radius: var(--hp-radius);
}
#mode.segmented button, #mode.segmented button:first-child,
#mode.segmented button:last-child, .histopia-focus-layers button,
.histopia-focus-layers button:first-child,
.histopia-focus-layers button:last-child {
  margin-left: 0;
  border-radius: var(--hp-radius);
}
figcaption, .context-title {
  color: var(--hp-muted);
  background: var(--hp-surface-soft);
  font-weight: 800;
}
#scope, .content, #workspace { background-color: var(--hp-bg); }
#scope, #detail, #status, .status, .counter, dt, label, figcaption,
.section-title span, .context-title, #neighbor-label {
  color: var(--hp-muted);
}
.provisional, #known-issue { border-radius: var(--hp-radius); }
.provisional {
  color: var(--hp-warning);
  border-color: #e1c590;
  background: #fff8e8;
}
#family-summary, #issues label, .probe-header {
  color: var(--hp-muted);
  background: var(--hp-surface-soft);
}
.app aside {
  color: var(--hp-text);
  background: var(--hp-surface);
  border-color: var(--hp-border-soft);
}
.app aside > section {
  color: var(--hp-text);
  background: var(--hp-surface);
  border-color: var(--hp-border-soft);
}
.app .dataset output, .app .review h2, .app .review > label,
.app #transition-summary, .app #status {
  color: var(--hp-muted);
}
.app .metrics {
  padding: 8px;
  border: 0;
  background: var(--hp-bg);
  box-shadow: none;
}
.app .metric {
  color: var(--hp-muted);
  border: 1px solid var(--hp-border-soft);
  background: var(--hp-surface);
  box-shadow: var(--hp-shadow);
}
.app .metric b { color: var(--hp-text); }
.app #issues label {
  color: var(--hp-muted);
  border: 1px solid var(--hp-border-soft);
  border-radius: var(--hp-radius);
  background: var(--hp-surface-soft);
}
.app #accept-passing {
  color: #fff;
  border-color: var(--hp-accent);
  background: var(--hp-accent);
}
#decision {
  align-self: start;
  max-width: 960px;
  margin: 24px;
  border: 1px solid var(--hp-border-soft);
  border-radius: var(--hp-radius);
  background: var(--hp-surface);
  box-shadow: var(--hp-shadow);
}
.status.accepted, .accepted #status, dd.pass { color: var(--hp-success); }
.status.rejected, dd.fail { color: var(--hp-danger); }
dd.warn { color: var(--hp-warning); }
.actions { gap: 8px; }
::placeholder { color: var(--hp-subtle); opacity: 1; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--hp-surface-soft); }
::-webkit-scrollbar-thumb {
  background: #b8c4d1;
  border: 2px solid var(--hp-surface-soft);
  border-radius: 999px;
}
.histopia-focus-host {
  color: var(--hp-text);
  background: var(--hp-surface);
}
.histopia-focus-toolbar, .histopia-focus-status {
  color: var(--hp-text);
  border-color: var(--hp-border-soft);
  background: var(--hp-surface);
}
.histopia-focus-toolbar button {
  color: var(--hp-text);
  border-color: var(--hp-border);
  border-radius: var(--hp-radius);
  background: var(--hp-surface);
}
.histopia-focus-layers button.active {
  color: #fff;
  border-color: var(--hp-accent);
  background: var(--hp-accent);
}
.histopia-focus-status { color: var(--hp-muted); }
#registration-feedback {
  color: var(--hp-text);
  border-color: var(--hp-border-soft);
  background: var(--hp-surface);
  font-family: inherit;
}
#registration-feedback input, #registration-feedback textarea,
#registration-feedback select, #registration-feedback button {
  border-color: var(--hp-border);
  border-radius: var(--hp-radius);
}
#registration-feedback .save {
  color: #fff;
  border-color: var(--hp-accent);
  background: var(--hp-accent);
}
@media (max-width: 700px) {
  #queue button.active, .slide-row.active {
    box-shadow: inset 0 -4px 0 var(--hp-active-mark);
  }
  #decision { margin: 8px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
}
"""


def themed_review_css(css: str) -> str:
    """Append one current shared theme after viewer-specific structural CSS."""

    structural_css = css.split(REVIEW_THEME_MARKER, maxsplit=1)[0].rstrip()
    return f"{structural_css}\n{REVIEW_THEME_CSS.strip()}\n"
