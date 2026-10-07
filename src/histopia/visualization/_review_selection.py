"""Shared specimen URL state for standalone and embedded review views."""

REVIEW_SELECTION_JS = r"""
globalThis.histopiaReviewSelection = {
  requested(cohorts) {
    const query = new URLSearchParams(location.search);
    const requested = query.get("subject") || query.get("mouse") || query.get("cohort");
    const selected = requested
      ? cohorts.find(row => String(row.id) === requested) : cohorts[0];
    if (!selected) throw new Error("No result for the requested specimen.");
    return String(selected.id);
  },
  remember(id) {
    const url = new URL(location.href);
    const previous = url.searchParams.get("subject") || url.searchParams.get("mouse") ||
      url.searchParams.get("cohort");
    if (previous && previous !== String(id)) {
      for (const key of ["section", "slide", "reconstruction", "dataset", "field",
        "stack_section", "stack_mode"]) url.searchParams.delete(key);
    }
    for (const key of ["subject", "mouse", "cohort"]) url.searchParams.set(key, id);
    history.replaceState(null, "", url);
    if (parent !== window) parent.postMessage(
      {type: "histopia-review-selection", subject: String(id)},
      location.origin === "null" ? "*" : location.origin);
  },
};
"""
