from __future__ import annotations

import numpy as np

from histopia.topology._interpolate import (
    interpolate_pair,
    observed_plane,
    smooth_displacement_field,
)
from histopia.topology._model import ObservedSection


def test_flow_interpolation_moves_semantic_support_without_changing_endpoints() -> None:
    source = _section("source", left=2)
    target = _section("target", left=4)
    source_xy = np.array([[2.0, 3.0], [2.0, 4.0], [3.0, 3.0], [3.0, 4.0]])
    target_xy = source_xy + np.array([2.0, 0.0])
    displacement, confidence = smooth_displacement_field(
        (9, 10),
        source_xy=source_xy,
        target_xy=target_xy,
        origin_um_xy=(0.0, 0.0),
        spacing_um=1.0,
        sigma_cells=1.0,
    )

    middle = interpolate_pair(
        source,
        target,
        fraction=0.5,
        z_um=5.0,
        segment=0,
        source_section=0,
        target_section=1,
        displacement_rc=displacement,
        flow_confidence=confidence,
    )
    observed = observed_plane(source, z_um=0, segment=0, section_index=0)

    assert np.array_equal(observed.labels, source.labels)
    assert np.array_equal(observed.support, source.support)
    assert np.count_nonzero(observed.uncertainty) == 0
    middle_cols = np.argwhere(middle.support)[:, 1]
    assert 3 <= np.median(middle_cols) <= 4
    assert np.allclose(middle.membership.sum(axis=0)[middle.support], 1.0)
    assert np.all((middle.uncertainty >= 0) & (middle.uncertainty <= 1))


def test_empty_flow_is_valid_and_returns_zero_confidence() -> None:
    field, confidence = smooth_displacement_field(
        (5, 6),
        source_xy=np.empty((0, 2)),
        target_xy=np.empty((0, 2)),
        origin_um_xy=(0, 0),
        spacing_um=2,
    )

    assert field.shape == (2, 5, 6)
    assert not np.any(field)
    assert not np.any(confidence)


def _section(name: str, *, left: int) -> ObservedSection:
    labels = np.full((9, 10), -1, dtype=np.int16)
    labels[2:7, left : left + 3] = 0
    membership = np.zeros((2, 9, 10), dtype=np.float32)
    membership[0] = labels == 0
    return ObservedSection(
        slide_id=name,
        labels=labels,
        membership=membership,
        support=labels >= 0,
        tissue_fraction=(labels >= 0).astype(np.float32),
        sparse_labels=np.zeros(4, dtype=np.int16),
    )
