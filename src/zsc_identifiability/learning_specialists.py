"""Auditable strategy clustering and online belief primitives for style baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zsc_identifiability.models import FiniteConventionGame


@dataclass(frozen=True)
class StrategyClusters:
    mode_to_cluster: dict[str, int]
    cross_play_returns: np.ndarray
    feature_eigenvalues: np.ndarray


def cluster_training_strategies(
    game: FiniteConventionGame, base_team_return: float = 100.0
) -> StrategyClusters:
    """Cluster exact cross-play return rows without reading hidden response labels."""
    returns = np.asarray(
        [
            [
                base_team_return - float(game.loss_exact(mode, decision))
                for decision in game.decisions
            ]
            for mode in game.mode_ids
        ],
        dtype=np.float64,
    )
    gram = returns @ returns.T
    eigenvalues = np.linalg.eigvalsh(gram)
    unique_rows: list[np.ndarray] = []
    assignments: dict[str, int] = {}
    for mode, row in zip(game.mode_ids, returns, strict=True):
        cluster = next(
            (
                index
                for index, known in enumerate(unique_rows)
                if np.allclose(row, known, atol=1e-12)
            ),
            None,
        )
        if cluster is None:
            cluster = len(unique_rows)
            unique_rows.append(row)
        assignments[mode] = cluster
    return StrategyClusters(assignments, returns, eigenvalues)


class FixedShareBelief:
    """TALENTS-style likelihood update with probability mass sharing."""

    def __init__(self, cluster_count: int, share: float = 0.05) -> None:
        if cluster_count < 1:
            raise ValueError("cluster_count must be positive")
        if share < 0 or share >= 1:
            raise ValueError("share must lie in [0, 1)")
        self.share = share
        self.weights = np.full(cluster_count, 1 / cluster_count, dtype=np.float64)

    def update(self, likelihoods: np.ndarray) -> np.ndarray:
        values = np.asarray(likelihoods, dtype=np.float64)
        if values.shape != self.weights.shape or np.any(values < 0):
            raise ValueError("likelihood vector is invalid")
        posterior = self.weights * np.maximum(values, 1e-12)
        posterior /= posterior.sum()
        uniform = np.full_like(posterior, 1 / len(posterior))
        self.weights = (1 - self.share) * posterior + self.share * uniform
        return self.weights.copy()


def tom_kl_selector(
    global_prediction: np.ndarray, cluster_predictions: np.ndarray
) -> tuple[int, np.ndarray]:
    """Select the cluster whose predictor is most distinct from the global model."""
    global_values = _normalize(global_prediction)
    cluster_values = np.asarray(cluster_predictions, dtype=np.float64)
    if cluster_values.ndim != 2 or cluster_values.shape[1] != len(global_values):
        raise ValueError("cluster prediction matrix has the wrong shape")
    divergences = np.asarray(
        [
            np.sum(values * np.log(np.maximum(values, 1e-12) / global_values))
            for values in (_normalize(row) for row in cluster_values)
        ]
    )
    return int(np.argmax(divergences)), divergences


def balanced_offline_policy_weights() -> dict[str, float]:
    """Declared collection mixture shared by TALENTS- and ToM-style preprocessors."""
    return {"passive_oracle": 1 / 3, "uniform_valid": 1 / 3, "task_active_oracle": 1 / 3}


def deterministic_two_means(
    values: np.ndarray, iterations: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster unlabeled trajectory latents with deterministic farthest-point starts."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError("two-means requires at least two latent vectors")
    distances = np.square(data[:, None, :] - data[None, :, :]).sum(axis=-1)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    centers = np.stack((data[first], data[second])).copy()
    assignments = np.zeros(len(data), dtype=np.int64)
    for _ in range(iterations):
        updated = np.square(data[:, None, :] - centers[None, :, :]).sum(axis=-1).argmin(axis=1)
        if np.array_equal(updated, assignments) and _ > 0:
            break
        assignments = updated
        for cluster in range(2):
            members = data[assignments == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return assignments, centers


def _normalize(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return np.asarray(result / result.sum(), dtype=np.float64)
