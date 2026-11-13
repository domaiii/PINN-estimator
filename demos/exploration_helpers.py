"""Small reusable helpers for the exploration example notebooks."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from source_estimation_pinn.environment import CellType, OccupancyGrid


class AnyAngleGridPlanner:
    """Theta*-style point-robot planner over an occupancy grid.

    The grid supplies the search neighbourhood. Line-of-sight shortcuts produce
    collision-checked path segments at arbitrary angles.
    """

    def __init__(self, occupancy: OccupancyGrid):
        self.occupancy = occupancy
        self.points = occupancy.free_points
        self._graph = self._build_graph()
        self._line_of_sight_cache: dict[tuple[int, int], bool] = {}

    def nearest_node(self, point: np.ndarray) -> int:
        point = np.asarray(point, dtype=float)
        if point.shape != (2,):
            raise ValueError("point must have shape (2,)")
        return int(np.argmin(np.linalg.norm(self.points - point, axis=1)))

    def nearest_nodes(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        squared_distances = np.sum(
            (points[:, None, :] - self.points[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared_distances, axis=1)

    def geodesic_distance_to(self, nodes: Iterable[int] | int) -> np.ndarray:
        nodes = np.atleast_1d(np.asarray(nodes, dtype=int))
        if nodes.size == 0:
            return np.full(len(self.points), np.inf)
        return np.asarray(
            dijkstra(
                self._graph,
                directed=False,
                indices=nodes,
                min_only=True,
            )
        )

    def shortest_paths(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        """Return Theta*-style distances and parents from one node."""
        self._validate_node(start)
        distances = np.full(len(self.points), np.inf)
        parents = np.full(len(self.points), -1, dtype=int)
        closed = np.zeros(len(self.points), dtype=bool)
        distances[start] = 0.0
        parents[start] = start
        queue = [(0.0, start)]

        while queue:
            distance, current = heapq.heappop(queue)
            if closed[current] or distance > distances[current] + 1e-12:
                continue
            closed[current] = True

            for offset in range(
                self._graph.indptr[current], self._graph.indptr[current + 1]
            ):
                neighbour = int(self._graph.indices[offset])
                if closed[neighbour]:
                    continue
                ancestor = parents[current]
                if self._line_of_sight(ancestor, neighbour):
                    predecessor = ancestor
                    candidate_distance = distances[ancestor] + np.linalg.norm(
                        self.points[ancestor] - self.points[neighbour]
                    )
                else:
                    predecessor = current
                    candidate_distance = distance + np.linalg.norm(
                        self.points[current] - self.points[neighbour]
                    )
                if candidate_distance + 1e-12 < distances[neighbour]:
                    distances[neighbour] = candidate_distance
                    parents[neighbour] = predecessor
                    heapq.heappush(queue, (candidate_distance, neighbour))

        return distances, parents

    def reconstruct_path(
        self,
        parents: np.ndarray,
        start: int,
        target: int,
    ) -> np.ndarray | None:
        self._validate_node(start)
        self._validate_node(target)
        if parents.shape != (len(self.points),):
            raise ValueError("parents has incompatible shape")
        if parents[target] < 0:
            return None

        nodes = [target]
        while nodes[-1] != start:
            nodes.append(int(parents[nodes[-1]]))
        nodes.reverse()
        if not all(
            self._line_of_sight(first, second)
            for first, second in itertools.pairwise(nodes)
        ):
            raise RuntimeError("Reconstructed path contains a blocked segment")
        return self.points[nodes]

    def _build_graph(self):
        rows, cols = np.nonzero(self.occupancy.free_mask)
        if len(rows) != len(self.points):
            raise RuntimeError("Free-cell centres and occupancy mask disagree")
        node_ids = np.full(self.occupancy.free_mask.shape, -1, dtype=int)
        node_ids[rows, cols] = np.arange(len(self.points))

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for node, (row, col) in enumerate(zip(rows, cols)):
                next_row, next_col = row + dr, col + dc
                if not (
                    0 <= next_row < node_ids.shape[0]
                    and 0 <= next_col < node_ids.shape[1]
                ):
                    continue
                neighbour = node_ids[next_row, next_col]
                if neighbour < 0:
                    continue
                if dr and dc and not (
                    self.occupancy.free_mask[row, next_col]
                    and self.occupancy.free_mask[next_row, col]
                ):
                    continue
                cost = self.occupancy.resolution * np.hypot(dr, dc)
                edges.extend(((node, neighbour), (neighbour, node)))
                weights.extend((cost, cost))

        edge_array = np.asarray(edges, dtype=int).reshape(-1, 2)
        return coo_matrix(
            (weights, (edge_array[:, 0], edge_array[:, 1])),
            shape=(len(self.points), len(self.points)),
        ).tocsr()

    def _line_of_sight(self, first: int, second: int) -> bool:
        key = (min(first, second), max(first, second))
        if key in self._line_of_sight_cache:
            return self._line_of_sight_cache[key]

        start, end = self.points[first], self.points[second]
        length = np.linalg.norm(end - start)
        count = max(
            2,
            int(np.ceil(length / (self.occupancy.resolution / 4))) + 1,
        )
        points = np.linspace(start, end, count)
        epsilon = self.occupancy.resolution * 1e-6
        offsets = epsilon * np.array([
            [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
            [-1, -1], [-1, 1], [1, -1], [1, 1],
        ])
        probes = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        visible = bool(
            np.all(self.occupancy.cell_type_at(probes) == CellType.FREE)
        )
        self._line_of_sight_cache[key] = visible
        return visible

    def _validate_node(self, node: int) -> None:
        if not 0 <= int(node) < len(self.points):
            raise ValueError("node index outside the planner graph")


def source_laplace_variances(
    estimator,
    covariance: torch.Tensor,
    evaluation_points: np.ndarray,
) -> np.ndarray:
    """Linearized posterior variance of q at every evaluation point."""
    _, covariance_qq, _ = estimator.laplace.split_covariance(covariance)
    with torch.no_grad():
        source_gradient = estimator.laplace.source_gradient_wrt_parameters(
            estimator._tensor(evaluation_points)
        ).to(covariance_qq.dtype)
        variances = torch.sum(
            (source_gradient @ covariance_qq) * source_gradient,
            dim=1,
        )
    if not torch.isfinite(variances).all() or torch.any(variances < -1e-10):
        raise RuntimeError("Invalid source-field variances")
    return variances.clamp_min(0).detach().cpu().numpy()


def mean_laplace_source_variance(
    estimator,
    covariance: torch.Tensor,
    evaluation_points: np.ndarray,
) -> float:
    """Mean linearized posterior variance of q over evaluation points."""
    return float(source_laplace_variances(estimator, covariance, evaluation_points).mean())


def point_source_emd(
    estimator,
    evaluation_points: np.ndarray,
    true_source: np.ndarray,
) -> float:
    """Wasserstein-1 distance from normalized predicted q to a point source."""
    evaluation_points = np.asarray(evaluation_points, dtype=float)
    true_source = np.asarray(true_source, dtype=float)
    source_values = np.asarray(
        estimator.predict_source(evaluation_points),
        dtype=float,
    ).reshape(-1)
    if not np.all(np.isfinite(source_values)) or np.any(source_values < 0):
        raise RuntimeError("Invalid predicted source field")
    total_source = float(source_values.sum())
    if total_source <= 0:
        raise RuntimeError("Predicted source field has no positive mass")
    normalized_source = source_values / total_source
    distances = np.linalg.norm(evaluation_points - true_source, axis=1)
    return float(normalized_source @ distances)
