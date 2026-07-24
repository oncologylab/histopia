import numpy as np
from scipy import ndimage as ndi

from histopia.registration import (
    BrightfieldMaskConfig,
    create_tissue_mask,
    refine_group_tissue_masks,
)
from histopia.registration._masking import (
    TissueMaskResult,
    _align_peer_mask_translation,
    _augment_with_group_components,
    _axis_binary_dilation,
    _axis_binary_opening,
    _carve_large_blank_regions,
    _clean_mask,
    _fill_small_holes,
    _group_density_union_candidate,
    _has_unrepresented_group_component,
    _mask_score,
    _pale_tissue_candidate,
    _polish_selected_mask,
    _recover_supported_pale_pixels,
    _recover_undercovered_pale_tissue,
    _remove_border_bar_components,
    _remove_hollow_detached_artifacts,
    _remove_image_frame_exterior,
    _remove_scanner_edges,
    clean_external_tissue_mask,
)


def test_linear_axis_morphology_matches_scipy_binary_operations() -> None:
    mask = np.random.default_rng(17).random((73, 91)) > 0.42

    for axis, size in ((0, 8), (0, 9), (1, 12), (1, 13)):
        shape = (size, 1) if axis == 0 else (1, size)
        structure = np.ones(shape, dtype=bool)

        assert np.array_equal(
            _axis_binary_opening(mask, size, axis=axis),
            ndi.binary_opening(mask, structure=structure),
        )
        assert np.array_equal(
            _axis_binary_dilation(mask, size, axis=axis),
            ndi.binary_dilation(mask, structure=structure),
        )


def test_scanner_edge_removal_disconnects_straight_rail_from_tissue() -> None:
    candidate = np.zeros((200, 300), dtype=bool)
    candidate[10:14, 5:180] = True
    candidate[12:100, 90:210] = True
    candidate[35:115, 80:220] |= np.tri(80, 140, dtype=bool)

    cleaned = _remove_scanner_edges(candidate)

    assert not cleaned[11, 20]
    assert cleaned[60:95, 120:180].mean() > 0.8


def test_image_frame_removal_clips_only_scanner_exterior() -> None:
    image = np.ones((240, 320, 3), dtype=np.float32)
    for column, value in enumerate((0.85, 0.70, 0.55, 0.40, 0.25), start=45):
        image[20:220, column] = value
    candidate = np.zeros((240, 320), dtype=bool)
    candidate[70:150, 12:44] = True
    candidate[35:105, 95:185] = True
    candidate[125:220, 105:215] = True

    cleaned = _remove_image_frame_exterior(candidate, image)

    assert not cleaned[:, :52].any()
    assert np.array_equal(cleaned[:, 52:], candidate[:, 52:])


def test_image_frame_removal_ignores_edge_without_exterior_foreground() -> None:
    image = np.ones((240, 320, 3), dtype=np.float32)
    image[20:220, 47:50] = 0.25
    candidate = np.zeros((240, 320), dtype=bool)
    candidate[35:220, 95:215] = True

    cleaned = _remove_image_frame_exterior(candidate, image)

    assert np.array_equal(cleaned, candidate)


def test_scanner_edge_removal_disconnects_thick_rail_from_tissue() -> None:
    candidate = np.zeros((400, 600), dtype=bool)
    yy, xx = np.ogrid[:400, :600]
    candidate |= ((yy - 190) / 125) ** 2 + ((xx - 380) / 155) ** 2 <= 1
    candidate[20:42, :330] = True
    candidate[35:90, 300:350] = True

    cleaned = _remove_scanner_edges(candidate)

    assert not cleaned[20:42, :180].any()
    assert cleaned[120:260, 330:430].mean() > 0.75


def test_scanner_edge_removal_preserves_dense_central_tissue_rows() -> None:
    candidate = np.zeros((300, 400), dtype=bool)
    candidate[90:210, 20:175] = True
    candidate[90:210, 190:385] = True

    cleaned = _remove_scanner_edges(candidate)

    assert np.array_equal(cleaned, candidate)


def test_hollow_detached_artifact_is_removed_but_solid_fragment_remains() -> None:
    mask = np.zeros((220, 300), dtype=bool)
    mask[70:190, 80:220] = True
    yy, xx = np.ogrid[:220, :300]
    ring = ((yy - 35) ** 2 + (xx - 245) ** 2 <= 24**2) & (
        (yy - 35) ** 2 + (xx - 245) ** 2 >= 16**2
    )
    mask |= ring
    mask[35:60, 35:65] = True

    cleaned = _remove_hollow_detached_artifacts(mask)

    assert not cleaned[35, 245]
    assert cleaned[45:55, 45:55].all()
    assert cleaned[100:160, 110:190].all()


def test_border_bar_component_is_removed_without_clipping_edge_tissue() -> None:
    mask = np.zeros((240, 320), dtype=bool)
    mask[70:240, 100:290] = True
    mask[20:180, :10] = True

    cleaned = _remove_border_bar_components(mask)

    assert not cleaned[50, 5]
    assert cleaned[150:230, 150:250].all()
    assert cleaned[-1, 180]


def test_group_augmentation_only_adds_nearby_or_substantial_tissue() -> None:
    shape = (200, 300)
    trusted = np.zeros(shape, dtype=bool)
    trusted[70:150, 70:170] = True
    candidate = trusted.copy()
    candidate[65:155, 65:180] = True
    candidate[20:25, 20:280] = True
    candidate[155:175, 220:245] = True
    peer_support = np.ones((256, 256), dtype=float)

    augmented = _augment_with_group_components(
        candidate, trusted, peer_support, (256, 256)
    )

    assert augmented[68, 68]
    assert not augmented[20:25, 20:280].any()
    assert not augmented[155:175, 220:245].any()


def test_group_augmentation_keeps_large_disconnected_tissue_without_support() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[70:150, 50:150] = True
    candidate = trusted.copy()
    candidate[75:145, 205:245] = True

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        np.zeros((256, 256), dtype=float),
        (256, 256),
    )

    assert augmented[85:135, 215:235].all()


def test_group_augmentation_keeps_smaller_fragment_with_peer_support() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 40:160] = True
    candidate = trusted.copy()
    candidate[85:125, 220:255] = True
    peer_support = np.zeros((256, 256), dtype=float)
    peer_support[105:165, 185:230] = 1.0

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        peer_support,
        (256, 256),
    )

    assert augmented[90:120, 225:250].all()


def test_peer_translation_aligns_shifted_dominant_tissue() -> None:
    target = np.zeros((128, 128), dtype=bool)
    target[40:100, 25:80] = True
    peer = np.zeros_like(target)
    peer[18:78, 55:110] = True

    aligned = _align_peer_mask_translation(peer, target)

    assert np.array_equal(aligned, target)


def test_peer_translation_keeps_raw_mask_when_overlap_does_not_improve() -> None:
    target = np.zeros((128, 128), dtype=bool)
    target[30:100, 25:75] = True
    peer = target.copy()
    peer[15:25, 95:110] = True

    aligned = _align_peer_mask_translation(peer, target)

    assert np.array_equal(aligned, peer)


def test_group_supported_pale_recovery_adds_tissue_not_debris() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[145:225, 45:175] = True
    image = np.ones((256, 256, 3), dtype=np.float32)
    image[145:225, 45:175] = [0.72, 0.45, 0.40]
    image[30:105, 55:170] = [0.91, 0.86, 0.82]
    image[35:100:5, 60:165:5] = [0.80, 0.73, 0.68]
    image[25:70, 205:245] = [0.65, 0.35, 0.30]
    support = np.zeros((256, 256), dtype=float)
    support[25:110, 50:175] = 0.9

    recovered = _recover_supported_pale_pixels(mask, image, support)

    assert recovered[40:95, 65:160].mean() > 0.85
    assert not recovered[25:70, 205:245].any()


def test_group_supported_pale_recovery_fills_supported_stain_dropout_only() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[25:230, 25:230] = True
    mask[55:115, 55:115] = False
    mask[140:200, 140:200] = False
    image = np.ones((256, 256, 3), dtype=np.float32)
    image[25:230, 25:230] = [0.72, 0.45, 0.40]
    image[55:115, 55:115] = [0.91, 0.86, 0.82]
    image[60:110:5, 60:110:5] = [0.80, 0.73, 0.68]
    image[140:200, 140:200] = 1.0
    support = np.ones((256, 256), dtype=float)

    recovered = _recover_supported_pale_pixels(mask, image, support)

    assert recovered[65:105, 65:105].mean() > 0.85
    assert not recovered[150:190, 150:190].any()


def test_group_supported_pale_recovery_does_not_expand_boundary_halo() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[70:190, 70:190] = True
    image = np.ones((256, 256, 3), dtype=np.float32)
    image[60:200, 60:200] = [0.91, 0.86, 0.82]
    image[70:190, 70:190] = [0.72, 0.45, 0.40]
    support = np.ones((256, 256), dtype=float)

    recovered = _recover_supported_pale_pixels(mask, image, support)

    assert np.array_equal(recovered, mask)


def test_group_supported_pale_recovery_limits_detached_growth_to_seed() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[150:225, 80:180] = True
    image = np.ones((256, 256, 3), dtype=np.float32)
    image[150:225, 80:180] = [0.72, 0.45, 0.40]
    image[30:100, 35:105] = [0.91, 0.86, 0.82]
    image[35:95:4, 40:100:4] = [0.80, 0.73, 0.68]
    image[95:145, 70:100] = [0.91, 0.86, 0.82]
    image[100:140:4, 75:95:4] = [0.80, 0.73, 0.68]
    image[30:100, 155:225] = [0.91, 0.86, 0.82]
    image[35:95:4, 160:220:4] = [0.80, 0.73, 0.68]
    support = np.zeros((256, 256), dtype=float)
    support[25:105, 30:110] = 0.7
    support[25:105, 150:230] = 0.7
    seed = np.zeros((256, 256), dtype=bool)
    seed[40:90, 45:95] = True

    recovered = _recover_supported_pale_pixels(
        mask,
        image,
        support,
        detached_seed=seed,
    )

    assert recovered[40:90, 45:95].mean() > 0.85
    assert not recovered[120:140, 75:95].any()
    assert not recovered[40:90, 165:215].any()


def test_group_supported_pale_recovery_rejects_smooth_seeded_halo() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[150:225, 80:180] = True
    image = np.ones((256, 256, 3), dtype=np.float32)
    image[150:225, 80:180] = [0.72, 0.45, 0.40]
    image[30:100, 35:105] = [0.91, 0.90, 0.89]
    support = np.zeros((256, 256), dtype=float)
    support[25:105, 30:110] = 0.8
    seed = support > 0.7

    recovered = _recover_supported_pale_pixels(
        mask,
        image,
        support,
        detached_seed=seed,
    )

    assert np.array_equal(recovered, mask)


def test_group_pale_recovery_requires_severe_area_undercoverage() -> None:
    typical = np.zeros((160, 160), dtype=bool)
    typical[40:120, 30:130] = True
    target = typical.copy()
    target[40:70, 30:130] = False
    image = np.ones((160, 160, 3), dtype=np.float32)
    image[40:120, 30:130] = [0.72, 0.45, 0.40]
    image[40:70, 30:130] = [0.91, 0.86, 0.82]
    image[42:68:4, 32:128:4] = [0.80, 0.73, 0.68]

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    recovered = _recover_undercovered_pale_tissue(
        {
            "target": result(target),
            "peer-1": result(typical),
            "peer-2": result(typical),
        },
        {"target": image},
        physical_pixel_areas=None,
        normalized_shape=(160, 160),
    )

    assert np.array_equal(recovered["target"].mask, target)
    assert recovered["target"].method == "synthetic"


def test_group_component_gate_detects_missing_recurring_object() -> None:
    target = np.zeros((128, 128), dtype=bool)
    target[60:110, 55:105] = True
    support = target.astype(float)
    support[15:50, 10:50] = 0.8

    assert _has_unrepresented_group_component(target, support)

    target[15:50, 10:24] = True
    assert _has_unrepresented_group_component(target, support)

    target[15:50, 24:50] = True
    assert not _has_unrepresented_group_component(target, support)


def test_group_component_gate_extends_partial_medium_object_only() -> None:
    support = np.zeros((200, 200), dtype=float)
    support[20:80, 20:70] = 0.8
    target = np.zeros((200, 200), dtype=bool)
    target[20:80, 20:48] = True

    assert _has_unrepresented_group_component(target, support)

    small_support = np.zeros_like(support)
    small_support[20:50, 20:60] = 0.8
    small_target = np.zeros_like(target)
    small_target[20:50, 20:42] = True
    assert not _has_unrepresented_group_component(small_target, small_support)


def test_group_augmentation_keeps_small_nearby_fragment_with_strong_support() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 50:170] = True
    candidate = trusted.copy()
    candidate[85:105, 180:195] = True
    peer_support = np.zeros((256, 256), dtype=float)
    peer_support[105:140, 150:175] = 1.0

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        peer_support,
        (256, 256),
    )

    assert augmented[88:102, 183:192].all()


def test_group_augmentation_keeps_nearby_fragment_with_moderate_support() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 50:170] = True
    candidate = trusted.copy()
    candidate[85:105, 180:195] = True
    peer_support = np.zeros((256, 256), dtype=float)
    peer_support[105:140, 150:175] = 0.35

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        peer_support,
        (256, 256),
    )

    assert augmented[88:102, 183:192].all()


def test_fragment_mode_uses_low_recurrence_threshold_for_large_groups() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[50:170, 35:175] = True
    candidate = trusted.copy()
    candidate[70:110, 185:220] = True
    peer_support = np.zeros((256, 256), dtype=float)
    peer_support[85:145, 155:190] = 0.04

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        peer_support,
        (256, 256),
        small_fragment_support=0.02,
        small_fragments_only=True,
    )

    assert augmented[75:105, 190:215].all()


def test_group_augmentation_adds_supported_extension_of_trusted_component() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 50:170] = True
    candidate = trusted.copy()
    candidate[90:95, 170:225] = True
    candidate[75:110, 220:255] = True
    peer_support = np.zeros((256, 256), dtype=float)
    peer_support[90:145, 185:225] = 0.80

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        peer_support,
        (256, 256),
    )

    assert augmented[80:105, 225:250].mean() > 0.8


def test_group_augmentation_rejects_small_distant_supported_debris() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 50:170] = True
    candidate = trusted.copy()
    candidate[15:35, 260:275] = True

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        np.ones((256, 256), dtype=float),
        (256, 256),
    )

    assert not augmented[15:35, 260:275].any()


def test_fragment_only_augmentation_does_not_add_large_object() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[60:160, 40:150] = True
    candidate = trusted.copy()
    candidate[70:150, 190:250] = True

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        np.ones((256, 256), dtype=float),
        (256, 256),
        small_fragments_only=True,
    )

    assert np.array_equal(augmented, trusted)


def test_group_augmentation_rejects_detached_low_fill_smear() -> None:
    trusted = np.zeros((200, 300), dtype=bool)
    trusted[70:155, 40:150] = True
    candidate = trusted.copy()
    for row in range(30, 155):
        column = 225 + (row - 30) // 4
        candidate[row, column : column + 10] = True

    augmented = _augment_with_group_components(
        candidate,
        trusted,
        np.ones((256, 256), dtype=float),
        (256, 256),
    )

    assert not augmented[30:155, 220:260].any()


def test_pale_candidate_exposes_disconnected_fragment_for_group_support() -> None:
    image = np.ones((200, 300, 3), dtype=np.float32)
    image[:3] = [0.75, 0.75, 0.75]
    image[:, :3] = [0.75, 0.75, 0.75]
    image[60:150, 35:155] = [0.92, 0.86, 0.82]
    image[85:135, 220:260] = [0.97, 0.94, 0.92]
    evidence = np.zeros(image.shape[:2], dtype=np.uint8)
    optical_density = np.mean(-np.log(np.clip(image, 1 / 255, 1.0)), axis=2)

    candidate = _pale_tissue_candidate(
        image,
        evidence,
        optical_density,
        BrightfieldMaskConfig(),
    )

    assert candidate[90:130, 225:255].mean() > 0.9
    assert not candidate[:3].any()
    assert not candidate[:, :3].any()


def test_external_mask_cleanup_removes_solid_scanner_rail() -> None:
    mask = np.zeros((200, 300), dtype=bool)
    mask[4:12, 3:297] = True
    mask[60:160, 80:240] = True

    cleaned = clean_external_tissue_mask(mask)

    assert not cleaned[:8].any()
    assert cleaned[80:140, 100:220].all()


def test_create_tissue_mask_detects_weak_brightfield_tissue() -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[25:80, 35:95] = np.array([238, 223, 204], dtype=np.uint8)

    result = create_tissue_mask(image)

    assert result.accepted
    assert result.method != "full_fallback"
    assert result.mask[40:70, 50:85].mean() > 0.9
    assert 0.15 < result.metrics["foreground_fraction"] < 0.45


def test_group_pale_candidate_preserves_density_supported_tissue() -> None:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[35:145, 25:115] = [150, 145, 155]
    image[65:130, 175:230] = [242, 225, 215]

    result = create_tissue_mask(image)
    density = result.candidate_masks["group_density_union"]
    pale = result.candidate_masks["group_pale_tissue"]

    assert np.all(pale[density])


def test_object_aware_mask_keeps_multiple_tissues_and_rejects_debris() -> None:
    image = np.full((220, 300, 3), 255, dtype=np.uint8)
    image[55:150, 35:115] = [225, 190, 175]
    image[70:175, 175:270] = [235, 205, 190]
    image[20:25, 245:250] = [160, 90, 80]

    result = create_tissue_mask(image)

    assert result.method != "full_fallback"
    assert result.mask[90:130, 60:90].all()
    assert result.mask[100:145, 200:240].all()
    assert not result.mask[20:25, 245:250].any()


def test_object_aware_mask_rejects_inset_rail() -> None:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    image[10:18, 4:316] = [175, 125, 115]
    image[80:190, 90:230] = [225, 190, 175]

    result = create_tissue_mask(image)

    assert result.mask[110:160, 120:200].all()
    assert not result.mask[10:18].any()


def test_object_aware_mask_rejects_distant_stained_label() -> None:
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    image[160:265, 35:150] = [225, 190, 175]
    image[18:62, 220:282] = [190, 115, 105]

    result = create_tissue_mask(image)

    assert result.mask[190:240, 65:120].all()
    assert not result.mask[25:55, 230:275].any()


def test_group_consensus_rejects_one_slide_artifact() -> None:
    common = np.zeros((100, 120), dtype=bool)
    common[35:80, 30:85] = True
    artifact = common.copy()
    artifact[5:25, 95:115] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {"first": result(artifact), "second": result(common), "third": result(common)}
    )

    assert refined["first"].mask[50:70, 45:70].all()
    assert not refined["first"].mask[5:25, 95:115].any()
    assert refined["first"].method.endswith("+group_consensus")
    assert refined["first"].metrics["group_foreground_fraction_ratio"] > 0


def test_group_consensus_rejects_medium_component_without_direct_recurrence() -> None:
    target = np.zeros((200, 300), dtype=bool)
    target[60:160, 35:155] = True
    target[65:115, 210:260] = True
    shifted_peer = np.zeros_like(target)
    shifted_peer[60:160, 75:195] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "target": result(target),
            "peer-1": result(shifted_peer),
            "peer-2": result(shifted_peer),
            "peer-3": result(shifted_peer),
        }
    )

    assert refined["target"].mask[80:140, 60:140].all()
    assert not refined["target"].mask[75:105, 220:250].any()


def test_group_consensus_rejects_small_repeated_debris() -> None:
    tissue = np.zeros((200, 300), dtype=bool)
    tissue[50:150, 60:180] = True
    debris = tissue.copy()
    debris[70:170, 240:250] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "first": result(debris),
            "second": result(debris),
            "third": result(debris),
        }
    )

    assert refined["first"].mask[80:140, 80:160].all()
    assert not refined["first"].mask[70:170, 240:250].any()


def test_group_consensus_rejects_substantial_repeated_scanner_rail() -> None:
    tissue = np.zeros((200, 300), dtype=bool)
    tissue[50:150, 70:190] = True
    rail = tissue.copy()
    rail[35:175, 240:255] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "first": result(rail),
            "second": result(rail),
            "third": result(rail),
        }
    )

    assert refined["first"].mask[80:140, 90:170].all()
    assert not refined["first"].mask[35:175, 240:255].any()


def test_group_consensus_rejects_repeated_l_shaped_frame_fragment() -> None:
    tissue = np.zeros((240, 320), dtype=bool)
    tissue[70:190, 150:270] = True
    framed = tissue.copy()
    framed[20:35, 20:140] = True
    framed[20:190, 20:35] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "first": result(framed),
            "second": result(framed),
            "third": result(framed),
        }
    )

    assert refined["first"].mask[90:170, 170:250].all()
    assert not refined["first"].mask[20:190, 20:140].any()


def test_group_consensus_rejects_broad_corner_rail_with_false_peer_support() -> None:
    tissue = np.zeros((240, 320), dtype=bool)
    tissue[70:190, 120:270] = True
    artifact = np.zeros_like(tissue)
    artifact[20:45, 20:145] = True
    artifact[20:175, 20:45] = True
    target = tissue | artifact
    shifted_peer = np.zeros_like(tissue)
    shifted_peer[55:190, 55:255] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "target": result(target),
            "peer-1": result(shifted_peer),
            "peer-2": result(shifted_peer),
        }
    )

    assert refined["target"].mask[90:170, 140:250].all()
    assert not refined["target"].mask[20:175, 20:85].any()


def test_group_consensus_preserves_displaced_neighbor_supported_tissue() -> None:
    target = np.zeros((220, 340), dtype=bool)
    target[50:170, 35:155] = True
    target[95:155, 205:275] = True
    peer = np.zeros_like(target)
    peer[50:170, 35:155] = True
    peer[95:155, 250:320] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "target": result(target),
            "peer-1": result(peer),
            "peer-2": result(peer),
        }
    )

    assert refined["target"].mask[105:145, 215:265].all()


def test_group_consensus_preserves_low_fill_curved_tissue() -> None:
    rows, cols = np.indices((240, 320))
    radius = np.hypot(rows - 120, cols - 90)
    curved = (radius > 58) & (radius < 74) & (cols < 115)
    tissue = np.zeros((240, 320), dtype=bool)
    tissue[70:190, 160:285] = True
    mask = tissue | curved

    def result(value: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(value, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {"first": result(mask), "second": result(mask), "third": result(mask)}
    )

    assert refined["first"].mask[curved].mean() == 1.0


def test_group_consensus_preserves_components_without_comparable_support() -> None:
    first = np.zeros((100, 120), dtype=bool)
    first[5:20, 5:20] = True
    first[75:90, 95:110] = True
    second = np.zeros_like(first)
    second[40:60, 45:65] = True
    third = np.zeros_like(first)
    third[40:60, 70:90] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {"first": result(first), "second": result(second), "third": result(third)},
        min_group_support=0.95,
    )

    assert np.array_equal(refined["first"].mask, first)


def test_group_only_proposals_recover_recurring_nearby_fragment() -> None:
    trusted = np.zeros((160, 220), dtype=bool)
    trusted[45:135, 35:145] = True
    proposal = trusted.copy()
    proposal[70:90, 155:175] = True

    def result() -> TissueMaskResult:
        return TissueMaskResult(
            trusted.copy(),
            "object_aware_fusion",
            {},
            True,
            [],
            candidate_warnings={
                "object_aware_fusion": [],
                "group_pale_tissue": [],
            },
            candidate_masks={
                "object_aware_fusion": trusted.copy(),
                "group_pale_tissue": proposal.copy(),
            },
        )

    refined = refine_group_tissue_masks(
        {"first": result(), "second": result(), "third": result()}
    )

    assert refined["first"].mask[72:88, 157:173].mean() > 0.9


def test_group_consensus_reports_calibrated_physical_area_ratio() -> None:
    small_canvas = np.zeros((100, 100), dtype=bool)
    small_canvas[25:75, 25:75] = True
    large_canvas = np.zeros((200, 200), dtype=bool)
    large_canvas[50:150, 50:150] = True

    def result(mask: np.ndarray) -> TissueMaskResult:
        return TissueMaskResult(mask, "synthetic", {}, True, [])

    refined = refine_group_tissue_masks(
        {
            "small": result(small_canvas),
            "large": result(large_canvas),
            "peer": result(small_canvas),
        },
        physical_pixel_areas={"small": 4.0, "large": 1.0, "peer": 4.0},
    )

    assert refined["small"].metrics["group_physical_area_ratio"] == 1.0
    assert refined["large"].metrics["group_physical_area_ratio"] == 1.0


def test_group_ranking_rejects_fused_area_outlier() -> None:
    tissue = np.zeros((100, 120), dtype=bool)
    tissue[35:80, 30:85] = True
    artifact = tissue.copy()
    artifact[5:95, 90:115] = True

    def result(mask: np.ndarray, candidates: bool = False) -> TissueMaskResult:
        undercovered = np.zeros_like(tissue)
        undercovered[45:65, 45:65] = True
        candidate_masks = (
            {
                "object_aware_fusion": artifact,
                "background_corrected": tissue,
                "adaptive_brightness": tissue,
                "group_density_union": undercovered,
            }
            if candidates
            else {}
        )
        return TissueMaskResult(
            mask,
            "object_aware_fusion",
            {},
            True,
            [],
            candidate_warnings={name: [] for name in candidate_masks},
            candidate_masks=candidate_masks,
        )

    refined = refine_group_tissue_masks(
        {
            "target": result(artifact, candidates=True),
            "peer-1": result(tissue),
            "peer-2": result(tissue),
        }
    )

    assert not refined["target"].mask[20:30, 100:110].any()
    assert "group_ranked" in refined["target"].method


def test_group_density_is_not_used_when_physical_baseline_is_sufficient() -> None:
    tissue = np.zeros((100, 120), dtype=bool)
    tissue[25:75, 25:95] = True
    density = tissue.copy()
    density[5:95, 100:115] = True

    def result(mask: np.ndarray, *, candidates: bool = False) -> TissueMaskResult:
        candidate_masks = (
            {
                "object_aware_fusion": tissue,
                "background_corrected": tissue,
                "group_density_union": density,
            }
            if candidates
            else {}
        )
        return TissueMaskResult(
            mask,
            "object_aware_fusion",
            {},
            True,
            [],
            candidate_warnings={name: [] for name in candidate_masks},
            candidate_masks=candidate_masks,
        )

    refined = refine_group_tissue_masks(
        {
            "target": result(tissue, candidates=True),
            "peer-1": result(tissue),
            "peer-2": result(tissue),
        },
        physical_pixel_areas={"target": 2.0, "peer-1": 2.0, "peer-2": 2.0},
    )

    assert not refined["target"].method.startswith("group_density_union")
    assert not refined["target"].mask[5:95, 100:115].any()


def test_group_density_candidate_removes_scanner_axis_lines() -> None:
    density = np.zeros((200, 300), dtype=bool)
    density[50:160, 60:240] = True
    density[190:196, :] = True
    density[:, 4:9] = True

    candidate = _group_density_union_candidate(
        density, density, BrightfieldMaskConfig(close_radius_px=0, open_radius_px=0)
    )

    assert candidate[80:140, 90:210].all()
    assert not candidate[190:196].any()
    assert not candidate[:, 4:9].any()


def test_small_hole_filling_preserves_large_lumen() -> None:
    mask = np.ones((100, 100), dtype=bool)
    mask[20:24, 20:24] = False
    mask[50:80, 50:80] = False

    filled = _fill_small_holes(mask, max_area=100)

    assert filled[20:24, 20:24].all()
    assert not filled[50:80, 50:80].any()


def test_group_pale_polishing_fills_small_holes_and_preserves_large_lumen() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:95, 5:95] = True
    mask[20:24, 20:24] = False
    mask[50:80, 50:80] = False
    result = TissueMaskResult(
        mask=mask,
        method="group_pale_tissue+group_ranked",
        metrics={},
        accepted=True,
        warnings=[],
        candidate_masks={"group_pale_tissue": mask},
    )

    polished = _polish_selected_mask(result)

    assert polished.mask[20:24, 20:24].all()
    assert not polished.mask[50:80, 50:80].any()
    assert polished.method.endswith("+polished")


def test_large_blank_region_is_carved_from_tissue_mask() -> None:
    image = np.ones((200, 260, 3), dtype=np.float32)
    image[35:165, 30:150] = [0.75, 0.45, 0.40]
    image[35:165:4, 30:150:4] = [0.35, 0.15, 0.12]
    mask = np.zeros((200, 260), dtype=bool)
    mask[30:170, 25:225] = True

    carved = _carve_large_blank_regions(image, mask)

    assert carved[60:140, 50:130].mean() > 0.8
    assert carved[60:140, 175:215].mean() < 0.1


def test_small_attached_blank_artifact_is_carved_from_tissue_mask() -> None:
    image = np.ones((200, 300, 3), dtype=np.float32)
    image[40:160, 30:190] = [0.75, 0.45, 0.40]
    image[40:160:3, 30:190:3] = [0.35, 0.15, 0.12]
    mask = np.zeros((200, 300), dtype=bool)
    mask[40:160, 30:190] = True
    mask[90:100, 190:260] = True

    carved = _carve_large_blank_regions(image, mask)

    assert carved[60:140, 50:170].mean() > 0.8
    assert not carved[90:100, 210:250].any()


def test_blank_carving_is_noop_when_overlap_is_small() -> None:
    image = np.ones((200, 260, 3), dtype=np.float32)
    image[40:160, 40:220] = [0.70, 0.45, 0.40]
    image[40:160:3, 40:220:3] = [0.35, 0.15, 0.12]
    mask = np.zeros((200, 260), dtype=bool)
    mask[40:160, 40:220] = True

    carved = _carve_large_blank_regions(image, mask)

    assert np.array_equal(carved, mask)


def test_blank_carving_removes_neutral_scanner_plate_attached_to_tissue() -> None:
    image = np.ones((200, 300, 3), dtype=np.float32)
    image[45:155, 35:165] = [0.76, 0.45, 0.40]
    image[45:155:3, 35:165:3] = [0.32, 0.14, 0.12]
    image[70:140, 165:260] = [0.865, 0.860, 0.870]
    image[10:35, 170:240] = [0.865, 0.860, 0.870]
    image[165:190, 55:125] = [0.865, 0.860, 0.870]
    mask = np.zeros((200, 300), dtype=bool)
    mask[40:160, 30:260] = True

    carved = _carve_large_blank_regions(image, mask)

    assert carved[65:145, 45:150].mean() > 0.8
    assert carved[80:130, 185:245].mean() < 0.1


def test_blank_carving_preserves_pale_chromatic_tissue() -> None:
    image = np.ones((200, 300, 3), dtype=np.float32)
    image[45:155, 40:250] = [0.90, 0.76, 0.72]
    mask = np.zeros((200, 300), dtype=bool)
    mask[45:155, 40:250] = True

    carved = _carve_large_blank_regions(image, mask)

    assert np.array_equal(carved, mask)


def test_create_tissue_mask_rejects_blank_slide_by_default() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(image)

    assert not result.accepted
    assert result.method != "full_fallback"
    assert not result.mask.any()


def test_create_tissue_mask_allows_explicit_legacy_full_fallback() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(
        image,
        BrightfieldMaskConfig(allow_full_fallback=True),
    )

    assert result.accepted
    assert result.method == "full_fallback"
    assert result.mask.all()
    assert "all auto_tissue candidates failed QC" in result.warnings


def test_full_mask_mode_is_explicit() -> None:
    image = np.full((16, 12, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(image, BrightfieldMaskConfig(mode="full"))

    assert result.method == "full"
    assert result.mask.shape == (16, 12)
    assert result.mask.all()


def test_clean_mask_removes_scanner_frame_but_keeps_tissue() -> None:
    raw = np.zeros((200, 240), dtype=bool)
    raw[5:15, 5:235] = True
    raw[5:195, 5:15] = True
    raw[90:150, 150:210] = True

    cleaned = _clean_mask(
        raw,
        BrightfieldMaskConfig(close_radius_px=0, open_radius_px=0),
    )

    assert not cleaned[10, 100]
    assert cleaned[120, 180]


def test_mask_score_prefers_balanced_tissue_over_scattered_artifacts() -> None:
    coherent = np.zeros((200, 240), dtype=bool)
    coherent[60:150, 80:180] = True
    scattered = np.zeros_like(coherent)
    for row in range(10, 190, 30):
        for col in range(10, 230, 30):
            scattered[row : row + 6, col : col + 6] = True

    config = BrightfieldMaskConfig()

    assert _mask_score(coherent, config) > _mask_score(scattered, config)
