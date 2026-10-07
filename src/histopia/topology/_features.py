"""Interpretable spatial feature spectra for semantic tissue labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from histopia._validation import positive_float, positive_int, require_choice

SPATIAL_FEATURE_ALGORITHM_VERSION = 1
_SCOPES = ("2d", "2.5d", "3d")


@dataclass(frozen=True, slots=True)
class SpatialFeatureConfig:
    """Physical scales and bounded graph controls for spatial features."""

    entropy_window_um: tuple[float, ...] = (448.0, 896.0)
    contact_distance_um: float = 224.0
    large_region_area_um2: float = 2_000_000.0
    region_block_width_um: float = 1_120.0
    section_thickness_um: float = 4.0
    centrality_exact_node_limit: int = 512

    def __post_init__(self) -> None:
        if not self.entropy_window_um:
            raise ValueError("entropy_window_um must contain at least one scale")
        if any(
            value <= 0 or not math.isfinite(value) for value in self.entropy_window_um
        ):
            raise ValueError("entropy window scales must be finite and positive")
        positive_float("contact_distance_um", self.contact_distance_um)
        positive_float("large_region_area_um2", self.large_region_area_um2)
        positive_float("region_block_width_um", self.region_block_width_um)
        positive_float("section_thickness_um", self.section_thickness_um)
        positive_int("centrality_exact_node_limit", self.centrality_exact_node_limit)


@dataclass(frozen=True, slots=True)
class SpatialFeature:
    """One named, unit-aware measurement in a spatial feature spectrum."""

    scope: str
    feature: str
    value: float
    unit: str
    class_name: str | None = None
    class_pair: tuple[str, str] | None = None
    section_index: int | None = None

    def __post_init__(self) -> None:
        require_choice("scope", self.scope, _SCOPES)
        if not self.feature:
            raise ValueError("feature name must not be empty")
        if not math.isfinite(self.value):
            raise ValueError("feature values must be finite")
        if self.class_name is not None and self.class_pair is not None:
            raise ValueError("a feature cannot bind both a class and a class pair")


@dataclass(frozen=True, slots=True)
class SpatialFeatureSpectrum:
    """A deterministic collection of spatial measurements."""

    class_names: tuple[str, ...]
    records: tuple[SpatialFeature, ...]
    algorithm_version: int = SPATIAL_FEATURE_ALGORITHM_VERSION

    def rows(self) -> list[dict[str, object]]:
        """Return portable table rows suitable for CSV or Parquet output."""

        return [
            {
                "scope": record.scope,
                "section_index": record.section_index,
                "class_name": record.class_name,
                "class_pair": (
                    "|".join(record.class_pair) if record.class_pair else None
                ),
                "feature": record.feature,
                "value": record.value,
                "unit": record.unit,
                "algorithm_version": self.algorithm_version,
            }
            for record in self.records
        ]


def extract_spatial_feature_spectrum(
    labels: np.ndarray,
    *,
    spacing_um_xy: tuple[float, float],
    z_positions_um: np.ndarray | None = None,
    class_names: tuple[str, ...] | None = None,
    scopes: tuple[str, ...] = _SCOPES,
    config: SpatialFeatureConfig | None = None,
) -> SpatialFeatureSpectrum:
    """Extract controlled 2D, adjacent-section, and volumetric features.

    Labels use ``-1`` for background. A two-dimensional input supports only
    ``2d`` features. Three-dimensional input order is ``z, y, x``.
    """

    config = config or SpatialFeatureConfig()
    array = np.asarray(labels)
    if array.ndim not in {2, 3} or not array.size:
        raise ValueError("labels must be a non-empty 2D or 3D array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("labels must contain integers")
    if np.any(array < -1):
        raise ValueError("labels may use only -1 as the background value")
    spacing_y, spacing_x = (
        positive_float("spacing_um_xy[0]", spacing_um_xy[0]),
        positive_float("spacing_um_xy[1]", spacing_um_xy[1]),
    )
    requested = tuple(dict.fromkeys(scopes))
    for scope in requested:
        require_choice("scope", scope, _SCOPES)
    stack = array[None, ...] if array.ndim == 2 else array
    if array.ndim == 2 and any(scope != "2d" for scope in requested):
        raise ValueError("2.5d and 3d features require a label stack")
    if len(stack) < 2 and any(scope != "2d" for scope in requested):
        raise ValueError("2.5d and 3d features require at least two sections")
    z_positions = _z_positions(stack, z_positions_um, config.section_thickness_um)
    names = _class_names(stack, class_names)
    records: list[SpatialFeature] = []
    if "2d" in requested:
        for index, section in enumerate(stack):
            records.extend(
                _section_features(
                    section,
                    section_index=index,
                    class_names=names,
                    spacing_y=spacing_y,
                    spacing_x=spacing_x,
                    config=config,
                )
            )
    if "2.5d" in requested:
        records.extend(
            _adjacent_features(
                stack,
                z_positions,
                names,
                spacing_y=spacing_y,
                spacing_x=spacing_x,
            )
        )
    if "3d" in requested:
        records.extend(
            _volume_features(
                stack,
                z_positions,
                names,
                spacing_y=spacing_y,
                spacing_x=spacing_x,
                section_thickness_um=config.section_thickness_um,
            )
        )
    return SpatialFeatureSpectrum(class_names=names, records=tuple(records))


def _section_features(
    labels: np.ndarray,
    *,
    section_index: int,
    class_names: tuple[str, ...],
    spacing_y: float,
    spacing_x: float,
    config: SpatialFeatureConfig,
) -> list[SpatialFeature]:
    records: list[SpatialFeature] = []
    support = labels >= 0
    support_count = int(np.count_nonzero(support))
    pixel_area = spacing_y * spacing_x
    distribution = _distribution(labels, len(class_names))
    records.append(
        SpatialFeature(
            scope="2d",
            section_index=section_index,
            feature="semantic_entropy",
            value=_entropy(distribution),
            unit="bits",
        )
    )
    for class_index, class_name in enumerate(class_names):
        selected = labels == class_index
        count = int(np.count_nonzero(selected))
        records.extend(
            [
                SpatialFeature(
                    scope="2d",
                    section_index=section_index,
                    class_name=class_name,
                    feature="area",
                    value=count * pixel_area / 1_000_000.0,
                    unit="mm2",
                ),
                SpatialFeature(
                    scope="2d",
                    section_index=section_index,
                    class_name=class_name,
                    feature="area_fraction",
                    value=count / max(support_count, 1),
                    unit="fraction",
                ),
            ]
        )
        nodes = _region_nodes_2d(
            selected,
            spacing_y=spacing_y,
            spacing_x=spacing_x,
            large_region_area_um2=config.large_region_area_um2,
            block_width_um=config.region_block_width_um,
        )
        graph = _distance_graph(nodes, config.contact_distance_um)
        records.extend(
            _single_graph_features(
                graph,
                nodes,
                scope="2d",
                class_name=class_name,
                section_index=section_index,
                exact_limit=config.centrality_exact_node_limit,
            )
        )
    records.extend(
        _local_entropy_features(
            labels,
            support,
            class_names,
            section_index=section_index,
            spacing_y=spacing_y,
            spacing_x=spacing_x,
            windows_um=config.entropy_window_um,
        )
    )
    class_nodes = [
        _region_nodes_2d(
            labels == class_index,
            spacing_y=spacing_y,
            spacing_x=spacing_x,
            large_region_area_um2=config.large_region_area_um2,
            block_width_um=config.region_block_width_um,
        )
        for class_index in range(len(class_names))
    ]
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            records.extend(
                _pair_graph_features(
                    class_nodes[left],
                    class_nodes[right],
                    distance_um=config.contact_distance_um,
                    class_pair=(class_names[left], class_names[right]),
                    section_index=section_index,
                )
            )
    return records


def _local_entropy_features(
    labels: np.ndarray,
    support: np.ndarray,
    class_names: tuple[str, ...],
    *,
    section_index: int,
    spacing_y: float,
    spacing_x: float,
    windows_um: tuple[float, ...],
) -> list[SpatialFeature]:
    from scipy.ndimage import uniform_filter

    records: list[SpatialFeature] = []
    for window_um in windows_um:
        if not class_names:
            records.append(
                SpatialFeature(
                    scope="2d",
                    section_index=section_index,
                    feature=f"local_entropy_{window_um:g}um",
                    value=0.0,
                    unit="bits",
                )
            )
            continue
        size = (
            max(1, int(round(window_um / spacing_y))),
            max(1, int(round(window_um / spacing_x))),
        )
        probabilities = np.stack(
            [
                uniform_filter(
                    (labels == class_index).astype(np.float32),
                    size=size,
                    mode="constant",
                )
                for class_index in range(len(class_names))
            ]
        )
        total = probabilities.sum(axis=0)
        probabilities = np.divide(
            probabilities,
            np.maximum(total, np.finfo(np.float32).eps),
        )
        entropy = -np.sum(
            np.where(
                probabilities > 0,
                probabilities
                * np.log2(np.maximum(probabilities, np.finfo(np.float32).eps)),
                0,
            ),
            axis=0,
        )
        records.append(
            SpatialFeature(
                scope="2d",
                section_index=section_index,
                feature=f"local_entropy_{window_um:g}um",
                value=float(np.mean(entropy[support])) if np.any(support) else 0.0,
                unit="bits",
            )
        )
    return records


def _adjacent_features(
    stack: np.ndarray,
    z_positions: np.ndarray,
    class_names: tuple[str, ...],
    *,
    spacing_y: float,
    spacing_x: float,
) -> list[SpatialFeature]:
    records: list[SpatialFeature] = []
    pair_dice: list[float] = []
    pair_agreement: list[float] = []
    transition_rates: list[float] = []
    gradients: list[list[float]] = [[] for _ in class_names]
    areas = spacing_y * spacing_x
    for index in range(len(stack) - 1):
        left = stack[index]
        right = stack[index + 1]
        left_support = left >= 0
        right_support = right >= 0
        intersection = left_support & right_support
        denominator = np.count_nonzero(left_support) + np.count_nonzero(right_support)
        pair_dice.append(
            2.0 * np.count_nonzero(intersection) / denominator if denominator else 1.0
        )
        agreement = (
            float(np.mean(left[intersection] == right[intersection]))
            if np.any(intersection)
            else 0.0
        )
        pair_agreement.append(agreement)
        transition_rates.append(1.0 - agreement)
        dz = float(z_positions[index + 1] - z_positions[index])
        for class_index in range(len(class_names)):
            left_area = np.count_nonzero(left == class_index) * areas
            right_area = np.count_nonzero(right == class_index) * areas
            gradients[class_index].append((right_area - left_area) / dz / 1_000.0)
    for feature, values in (
        ("mean_support_dice", pair_dice),
        ("mean_label_agreement", pair_agreement),
        ("mean_transition_rate", transition_rates),
    ):
        records.append(
            SpatialFeature(
                scope="2.5d",
                feature=feature,
                value=float(np.mean(values)) if values else 0.0,
                unit="fraction",
            )
        )
    for class_index, (class_name, class_gradients) in enumerate(
        zip(class_names, gradients, strict=True)
    ):
        records.extend(
            [
                SpatialFeature(
                    scope="2.5d",
                    class_name=class_name,
                    feature="mean_absolute_area_gradient",
                    value=(
                        float(np.mean(np.abs(class_gradients)))
                        if class_gradients
                        else 0.0
                    ),
                    unit="mm2_per_mm_z",
                ),
                SpatialFeature(
                    scope="2.5d",
                    class_name=class_name,
                    feature="persistence_fraction",
                    value=float(np.mean(np.any(stack == class_index, axis=(1, 2)))),
                    unit="fraction",
                ),
            ]
        )
    return records


def _volume_features(
    stack: np.ndarray,
    z_positions: np.ndarray,
    class_names: tuple[str, ...],
    *,
    spacing_y: float,
    spacing_x: float,
    section_thickness_um: float,
) -> list[SpatialFeature]:
    from scipy.ndimage import label as component_labels

    records: list[SpatialFeature] = []
    z_weights = _z_weights(z_positions, section_thickness_um)
    voxel_area = spacing_y * spacing_x
    support_volume = float(
        np.sum(np.count_nonzero(stack >= 0, axis=(1, 2)) * voxel_area * z_weights)
    )
    class_volumes: list[float] = []
    for class_index in range(len(class_names)):
        counts = np.count_nonzero(stack == class_index, axis=(1, 2))
        class_volumes.append(float(np.sum(counts * voxel_area * z_weights)))
    records.append(
        SpatialFeature(
            scope="3d",
            feature="semantic_entropy",
            value=_entropy(np.asarray(class_volumes, dtype=float)),
            unit="bits",
        )
    )
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    for class_index, class_name in enumerate(class_names):
        selected = stack == class_index
        component_map, component_count = component_labels(selected, structure=structure)
        component_sizes = np.bincount(component_map.ravel())[1:]
        occupied = np.flatnonzero(np.any(selected, axis=(1, 2)))
        per_z_volume = np.count_nonzero(selected, axis=(1, 2)) * voxel_area * z_weights
        records.extend(
            [
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="volume",
                    value=class_volumes[class_index] / 1_000_000_000.0,
                    unit="mm3",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="volume_fraction",
                    value=class_volumes[class_index] / max(support_volume, 1e-12),
                    unit="fraction",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="connected_components",
                    value=float(component_count),
                    unit="count",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="largest_component_fraction",
                    value=(
                        float(np.max(component_sizes) / max(np.sum(component_sizes), 1))
                        if len(component_sizes)
                        else 0.0
                    ),
                    unit="fraction",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="z_entropy",
                    value=_entropy(per_z_volume),
                    unit="bits",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="z_span",
                    value=(
                        float(z_positions[occupied[-1]] - z_positions[occupied[0]])
                        if len(occupied) > 1
                        else 0.0
                    ),
                    unit="um",
                ),
                SpatialFeature(
                    scope="3d",
                    class_name=class_name,
                    feature="surface_area",
                    value=_class_surface_area(
                        stack,
                        class_index,
                        z_weights=z_weights,
                        spacing_y=spacing_y,
                        spacing_x=spacing_x,
                    )
                    / 1_000_000.0,
                    unit="mm2",
                ),
            ]
        )
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            records.append(
                SpatialFeature(
                    scope="3d",
                    class_pair=(class_names[left], class_names[right]),
                    feature="interface_area",
                    value=_pair_interface_area(
                        stack,
                        left,
                        right,
                        z_weights=z_weights,
                        spacing_y=spacing_y,
                        spacing_x=spacing_x,
                    )
                    / 1_000_000.0,
                    unit="mm2",
                )
            )
    return records


def _region_nodes_2d(
    mask: np.ndarray,
    *,
    spacing_y: float,
    spacing_x: float,
    large_region_area_um2: float,
    block_width_um: float,
) -> np.ndarray:
    from scipy.ndimage import label as component_labels

    components, count = component_labels(mask)
    nodes: list[tuple[float, float, float]] = []
    pixel_area = spacing_y * spacing_x
    for component in range(1, count + 1):
        coordinates = np.argwhere(components == component)
        area = len(coordinates) * pixel_area
        if area <= large_region_area_um2:
            center = np.mean(coordinates, axis=0)
            nodes.append((center[1] * spacing_x, center[0] * spacing_y, area))
            continue
        block_rows = np.floor(coordinates[:, 0] * spacing_y / block_width_um)
        block_cols = np.floor(coordinates[:, 1] * spacing_x / block_width_um)
        block_ids = np.column_stack([block_rows, block_cols]).astype(np.int64)
        for block in np.unique(block_ids, axis=0):
            selected = np.all(block_ids == block, axis=1)
            block_coordinates = coordinates[selected]
            center = np.mean(block_coordinates, axis=0)
            nodes.append(
                (
                    center[1] * spacing_x,
                    center[0] * spacing_y,
                    len(block_coordinates) * pixel_area,
                )
            )
    return np.asarray(nodes, dtype=np.float64).reshape(-1, 3)


def _distance_graph(nodes: np.ndarray, distance_um: float):
    nx = _networkx()
    graph = nx.Graph()
    for index, (x, y, area) in enumerate(nodes):
        graph.add_node(index, position=(float(x), float(y)), area=float(area))
    if len(nodes) > 1:
        from scipy.spatial import cKDTree

        for left, right in sorted(cKDTree(nodes[:, :2]).query_pairs(distance_um)):
            distance = float(np.linalg.norm(nodes[left, :2] - nodes[right, :2]))
            graph.add_edge(left, right, distance=distance, weight=distance)
    return graph


def _single_graph_features(
    graph,
    nodes: np.ndarray,
    *,
    scope: str,
    class_name: str,
    section_index: int,
    exact_limit: int,
) -> list[SpatialFeature]:
    nx = _networkx()
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    distances = np.asarray(
        [data["distance"] for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    components = list(nx.connected_components(graph)) if node_count else []
    largest = max(components, key=len) if components else set()
    clustering = (
        float(np.mean(list(nx.clustering(graph).values()))) if node_count else 0.0
    )
    path_length = 0.0
    diameter = 0.0
    closeness = 0.0
    betweenness = 0.0
    if len(largest) > 1:
        subgraph = graph.subgraph(largest)
        if len(largest) <= exact_limit:
            path_length = float(nx.average_shortest_path_length(subgraph))
            diameter = float(nx.diameter(subgraph))
            closeness = float(np.mean(list(nx.closeness_centrality(subgraph).values())))
            betweenness = float(
                np.mean(list(nx.betweenness_centrality(subgraph).values()))
            )
        else:
            betweenness = float(
                np.mean(
                    list(
                        nx.betweenness_centrality(
                            subgraph,
                            k=min(64, len(largest)),
                            seed=0,
                        ).values()
                    )
                )
            )
    modularity = 0.0
    if edge_count:
        communities = tuple(nx.community.greedy_modularity_communities(graph))
        modularity = float(nx.community.modularity(graph, communities))
    assortativity = 0.0
    if edge_count and np.any(degrees != degrees[0]):
        candidate = float(nx.degree_assortativity_coefficient(graph))
        assortativity = candidate if math.isfinite(candidate) else 0.0
    areas = nodes[:, 2] if len(nodes) else np.empty(0)
    positions = nodes[:, :2] if len(nodes) else np.empty((0, 2))
    values = {
        "region_nodes": (float(node_count), "count"),
        "region_edges": (float(edge_count), "count"),
        "edge_density": (
            float(nx.density(graph)) if node_count > 1 else 0.0,
            "fraction",
        ),
        "mean_edge_distance": (
            float(np.mean(distances)) if len(distances) else 0.0,
            "um",
        ),
        "std_edge_distance": (
            float(np.std(distances)) if len(distances) else 0.0,
            "um",
        ),
        "mean_degree": (float(np.mean(degrees)) if len(degrees) else 0.0, "count"),
        "degree_cv": (_coefficient_of_variation(degrees), "ratio"),
        "degree_entropy": (_entropy(degrees), "bits"),
        "connected_components": (float(len(components)), "count"),
        "largest_component_fraction": (len(largest) / max(node_count, 1), "fraction"),
        "average_clustering": (clustering, "fraction"),
        "average_path_length": (path_length, "edges"),
        "diameter": (diameter, "edges"),
        "mean_region_area": (
            float(np.mean(areas)) / 1_000_000.0 if len(areas) else 0.0,
            "mm2",
        ),
        "region_area_cv": (_coefficient_of_variation(areas), "ratio"),
        "spatial_dispersion": (
            float(np.mean(np.std(positions, axis=0))) if len(positions) else 0.0,
            "um",
        ),
        "mean_closeness": (closeness, "ratio"),
        "mean_betweenness": (betweenness, "ratio"),
        "modularity": (modularity, "ratio"),
        "degree_assortativity": (assortativity, "ratio"),
    }
    return [
        SpatialFeature(
            scope=scope,
            section_index=section_index,
            class_name=class_name,
            feature=feature,
            value=float(value),
            unit=unit,
        )
        for feature, (value, unit) in values.items()
    ]


def _pair_graph_features(
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance_um: float,
    class_pair: tuple[str, str],
    section_index: int,
) -> list[SpatialFeature]:
    left_count = len(left)
    right_count = len(right)
    pairs: list[tuple[int, int, float]] = []
    if left_count and right_count:
        from scipy.spatial import cKDTree

        neighbors = cKDTree(left[:, :2]).query_ball_tree(
            cKDTree(right[:, :2]),
            distance_um,
        )
        pairs = [
            (
                left_index,
                right_index,
                float(np.linalg.norm(left[left_index, :2] - right[right_index, :2])),
            )
            for left_index, indices in enumerate(neighbors)
            for right_index in indices
        ]
    left_degrees = np.bincount(
        [item[0] for item in pairs], minlength=left_count
    ).astype(float)
    right_degrees = np.bincount(
        [item[1] for item in pairs], minlength=right_count
    ).astype(float)
    distances = np.asarray([item[2] for item in pairs], dtype=float)
    values = {
        "interaction_edges": (float(len(pairs)), "count"),
        "interaction_density": (
            len(pairs) / max(left_count * right_count, 1),
            "fraction",
        ),
        "mean_interaction_distance": (
            float(np.mean(distances)) if len(distances) else 0.0,
            "um",
        ),
        "mean_right_neighbors_per_left": (
            float(np.mean(left_degrees)) if left_count else 0.0,
            "count",
        ),
        "mean_left_neighbors_per_right": (
            float(np.mean(right_degrees)) if right_count else 0.0,
            "count",
        ),
        "isolated_left_fraction": (
            float(np.mean(left_degrees == 0)) if left_count else 0.0,
            "fraction",
        ),
        "isolated_right_fraction": (
            float(np.mean(right_degrees == 0)) if right_count else 0.0,
            "fraction",
        ),
        "left_degree_entropy": (_entropy(left_degrees), "bits"),
        "right_degree_entropy": (_entropy(right_degrees), "bits"),
    }
    return [
        SpatialFeature(
            scope="2d",
            section_index=section_index,
            class_pair=class_pair,
            feature=feature,
            value=float(value),
            unit=unit,
        )
        for feature, (value, unit) in values.items()
    ]


def _class_surface_area(
    stack: np.ndarray,
    class_index: int,
    *,
    z_weights: np.ndarray,
    spacing_y: float,
    spacing_x: float,
) -> float:
    selected = stack == class_index
    area = 0.0
    for index, plane in enumerate(selected):
        padded_x = np.pad(plane, ((0, 0), (1, 1)))
        padded_y = np.pad(plane, ((1, 1), (0, 0)))
        area += (
            np.count_nonzero(padded_x[:, 1:] != padded_x[:, :-1])
            * spacing_y
            * z_weights[index]
        )
        area += (
            np.count_nonzero(padded_y[1:, :] != padded_y[:-1, :])
            * spacing_x
            * z_weights[index]
        )
    area += np.count_nonzero(selected[0]) * spacing_x * spacing_y
    area += np.count_nonzero(selected[-1]) * spacing_x * spacing_y
    if len(selected) > 1:
        area += np.count_nonzero(selected[1:] != selected[:-1]) * spacing_x * spacing_y
    return float(area)


def _pair_interface_area(
    stack: np.ndarray,
    left: int,
    right: int,
    *,
    z_weights: np.ndarray,
    spacing_y: float,
    spacing_x: float,
) -> float:
    area = 0.0
    for index, plane in enumerate(stack):
        horizontal = plane[:, 1:]
        horizontal_left = plane[:, :-1]
        area += (
            np.count_nonzero(
                ((horizontal == left) & (horizontal_left == right))
                | ((horizontal == right) & (horizontal_left == left))
            )
            * spacing_y
            * z_weights[index]
        )
        vertical = plane[1:, :]
        vertical_top = plane[:-1, :]
        area += (
            np.count_nonzero(
                ((vertical == left) & (vertical_top == right))
                | ((vertical == right) & (vertical_top == left))
            )
            * spacing_x
            * z_weights[index]
        )
    if len(stack) > 1:
        upper = stack[1:]
        lower = stack[:-1]
        area += (
            np.count_nonzero(
                ((upper == left) & (lower == right))
                | ((upper == right) & (lower == left))
            )
            * spacing_x
            * spacing_y
        )
    return float(area)


def _z_positions(
    stack: np.ndarray,
    z_positions_um: np.ndarray | None,
    section_thickness_um: float,
) -> np.ndarray:
    if z_positions_um is None:
        return np.arange(len(stack), dtype=np.float64) * section_thickness_um
    values = np.asarray(z_positions_um, dtype=np.float64)
    if values.shape != (len(stack),) or not np.all(np.isfinite(values)):
        raise ValueError("z_positions_um must contain one finite value per section")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("z_positions_um must be strictly increasing")
    return values


def _z_weights(z_positions: np.ndarray, section_thickness_um: float) -> np.ndarray:
    if len(z_positions) == 1:
        return np.asarray([section_thickness_um], dtype=np.float64)
    gaps = np.diff(z_positions)
    weights = np.empty(len(z_positions), dtype=np.float64)
    weights[0] = max(section_thickness_um, gaps[0] / 2.0)
    weights[-1] = max(section_thickness_um, gaps[-1] / 2.0)
    if len(weights) > 2:
        weights[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return weights


def _class_names(
    labels: np.ndarray,
    class_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    maximum = int(np.max(labels)) if np.any(labels >= 0) else -1
    count = maximum + 1
    if class_names is None:
        return tuple(f"class_{index + 1}" for index in range(count))
    if len(class_names) != count or len(set(class_names)) != count:
        raise ValueError("class_names must uniquely name every non-background label")
    return class_names


def _distribution(labels: np.ndarray, class_count: int) -> np.ndarray:
    selected = labels[labels >= 0]
    return np.bincount(selected, minlength=class_count).astype(np.float64)


def _entropy(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if not len(array):
        return 0.0
    probabilities = array / np.sum(array)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _coefficient_of_variation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if len(array) else 0.0
    return float(np.std(array) / mean) if mean > 0 else 0.0


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "spatial graph features require networkx from the 'topology' extra"
        ) from exc
    return nx
