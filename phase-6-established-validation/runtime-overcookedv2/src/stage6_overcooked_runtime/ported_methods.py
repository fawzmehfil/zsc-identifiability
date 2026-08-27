"""Paper-faithful building blocks for the Stage 6 method ports.

The functions in this module are deliberately independent of the training
orchestrator.  That makes the peer reward, TBS selector, and CSP protocol
auditable without launching an Overcooked training run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from overcooked_v2_experiments.ppo.models.common import CNN
from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN

PACE_AUXILIARY_WEIGHT = 1.0
PACE_INITIAL_BONUS = 0.2
PACE_WARMUP_FRACTION = 1.0 / 30.0
PACE_BONUS_DECAY_FRACTION = 5.0 / 6.0
CSP_INTRINSIC_WEIGHT = 0.1
TBS_PROBABILITY_EPSILON = 1e-6


def pace_bonus_weight(completed_transitions: int, target_transitions: int) -> float:
    """Return the preregistered linearly decayed PACE coefficient."""

    if target_transitions <= 0:
        raise ValueError("target transitions must be positive")
    if completed_transitions < 0:
        raise ValueError("completed transitions cannot be negative")
    end = PACE_BONUS_DECAY_FRACTION * target_transitions
    progress = min(float(completed_transitions) / max(end, 1.0), 1.0)
    return PACE_INITIAL_BONUS * (1.0 - progress)


def pace_is_auxiliary_warmup(completed_transitions: int, target_transitions: int) -> bool:
    if target_transitions <= 0:
        raise ValueError("target transitions must be positive")
    return completed_transitions < PACE_WARMUP_FRACTION * target_transitions


def pace_identity_reward(identity_probabilities, partner_indices):
    """Probability assigned to the actual training partner, detached by callers."""

    probabilities = jnp.asarray(identity_probabilities)
    indices = jnp.asarray(partner_indices, dtype=jnp.int32)
    return jnp.take_along_axis(probabilities, indices[..., None], axis=-1)[..., 0]


class PaceActorCritic(nn.Module):
    """Official convolutional encoder with encounter-local context and ID head."""

    action_dim: int
    partner_count: int
    hidden_size: int = 128
    head_size: int = 128

    def setup(self):
        self.encoder = CNN(output_size=self.hidden_size, activation=nn.relu)
        self.context_rnn = ScannedRNN()
        self.context_projection = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )
        self.actor_hidden = nn.Dense(
            self.head_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )
        self.actor_output = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        self.critic_hidden = nn.Dense(
            self.head_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )
        self.critic_output = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )
        self.identity_output = nn.Dense(
            self.partner_count,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )

    def __call__(self, hidden, x, train=False):
        observations, resets = x
        previous_actions = jnp.zeros(resets.shape, dtype=jnp.int32)
        hidden, policy, value, _ = self.with_identity(
            hidden, observations, previous_actions, resets
        )
        return hidden, policy, value

    def with_identity(self, hidden, observations, previous_actions, resets):
        observations = jnp.asarray(observations)
        previous_actions = jnp.asarray(previous_actions, dtype=jnp.int32)
        resets = jnp.asarray(resets, dtype=jnp.bool_)
        time_size, batch_size = observations.shape[:2]
        flat = observations.reshape((-1, *observations.shape[2:]))
        embedding = self.encoder(flat)
        embedding = embedding.reshape((time_size, batch_size, self.hidden_size))
        action_one_hot = jax.nn.one_hot(previous_actions, self.action_dim)
        context_input = jnp.concatenate((embedding, action_one_hot), axis=-1)
        context_input = nn.relu(self.context_projection(context_input))
        hidden, context = self.context_rnn(hidden, (context_input, resets))
        features = jnp.concatenate((embedding, context), axis=-1)
        actor = nn.relu(self.actor_hidden(features))
        logits = self.actor_output(actor)
        critic = nn.relu(self.critic_hidden(features))
        value = self.critic_output(critic)[..., 0]
        identity_logits = self.identity_output(context)
        return hidden, distrax.Categorical(logits=logits), value, identity_logits


class VisibleHistoryPredictor(nn.Module):
    """CNN-GRU predictor used by TBS concept models and CSP encoders."""

    output_dim: int
    hidden_size: int = 128
    categorical: bool = False

    @nn.compact
    def __call__(self, hidden, observations, resets):
        observations = jnp.asarray(observations)
        resets = jnp.asarray(resets, dtype=jnp.bool_)
        time_size, batch_size = observations.shape[:2]
        flat = observations.reshape((-1, *observations.shape[2:]))
        embedding = CNN(output_size=self.hidden_size, activation=nn.relu)(flat)
        embedding = embedding.reshape((time_size, batch_size, self.hidden_size))
        hidden, context = ScannedRNN()(hidden, (embedding, resets))
        logits = nn.Dense(self.output_dim)(context)
        return hidden, logits if self.categorical else jax.nn.sigmoid(logits)


class CSPTrajectoryModel(nn.Module):
    """Local-history encoder and visible-partner-action decoder."""

    action_dim: int = 7
    hidden_size: int = 128
    latent_size: int = 32

    @nn.compact
    def __call__(self, hidden, observations, ego_actions, rewards, resets):
        observations = jnp.asarray(observations)
        time_size, batch_size = observations.shape[:2]
        flat = observations.reshape((-1, *observations.shape[2:]))
        embedding = CNN(output_size=self.hidden_size, activation=nn.relu)(flat)
        embedding = embedding.reshape((time_size, batch_size, self.hidden_size))
        inputs = jnp.concatenate(
            (
                embedding,
                jax.nn.one_hot(ego_actions, 6),
                jnp.asarray(rewards)[..., None],
            ),
            axis=-1,
        )
        inputs = nn.relu(nn.Dense(self.hidden_size)(inputs))
        hidden, sequence = ScannedRNN()(hidden, (inputs, resets))
        latent = nn.Dense(self.latent_size)(sequence)
        logits = nn.Dense(self.action_dim)(jnp.concatenate((embedding, latent), axis=-1))
        return hidden, latent, logits


def tbs_similarity_matrix(cross_play_values, epsilon: float = 1e-10) -> np.ndarray:
    """Exact ToMZSC similarity calculation with explicit competence checks."""

    values = np.asarray(cross_play_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("TBS cross-play values must form a square matrix")
    diagonal = np.diag(values)
    denominator = diagonal[:, None] + diagonal[None, :]
    if np.any(denominator <= 0):
        raise ValueError("TBS requires positive competent self-play denominators")
    similarity = (values + values.T) / denominator
    similarity = np.clip(similarity, 0.0, 1.0)
    return similarity * (1.0 - epsilon) + epsilon / 2.0


def pinned_tbs_clusters(
    cross_play_values,
    tomzsc_repository: str | Path,
    *,
    maximum_clusters: int = 6,
) -> tuple[int, ...]:
    """Run the pinned self-tuning implementation without copying its source."""

    repository = Path(tomzsc_repository).resolve()
    clustering = repository / "clustering"
    if not clustering.is_dir():
        raise ValueError(f"missing pinned ToMZSC clustering package: {clustering}")
    sys.path.insert(0, str(clustering))
    try:
        module = importlib.import_module("get_clusters")
        similarity = tbs_similarity_matrix(cross_play_values)
        clusters = module.clusters_from_similarity(
            similarity, max_n_cluster=min(maximum_clusters, similarity.shape[0])
        )
        labels = np.empty(similarity.shape[0], dtype=np.int32)
        for cluster_id, members in enumerate(clusters):
            labels[np.asarray(members, dtype=np.int32)] = cluster_id
    finally:
        sys.path.remove(str(clustering))
        for name in tuple(sys.modules):
            if name == "get_clusters" or name.startswith("stsc"):
                sys.modules.pop(name, None)
    return tuple(int(value) for value in labels)


def bernoulli_kl(left, right):
    left = jnp.clip(jnp.asarray(left), TBS_PROBABILITY_EPSILON, 1 - TBS_PROBABILITY_EPSILON)
    right = jnp.clip(jnp.asarray(right), TBS_PROBABILITY_EPSILON, 1 - TBS_PROBABILITY_EPSILON)
    return left * jnp.log(left / right) + (1 - left) * jnp.log((1 - left) / (1 - right))


def select_tbs_cluster(global_predictions, cluster_predictions, valid_mask=None) -> int:
    """Hard minimum cumulative KL selector used at deployment."""

    global_values = jnp.asarray(global_predictions)
    cluster_values = jnp.asarray(cluster_predictions)
    if cluster_values.ndim != global_values.ndim + 1:
        raise ValueError("cluster predictions require a leading cluster dimension")
    divergences = bernoulli_kl(global_values[None, ...], cluster_values)
    if valid_mask is not None:
        mask = jnp.asarray(valid_mask)
        divergences = divergences * mask[None, ..., None]
    totals = divergences.sum(axis=tuple(range(1, divergences.ndim)))
    return int(jnp.argmin(totals))


def csp_probe_reward(task_reward, prediction_loss):
    return jnp.asarray(task_reward) + CSP_INTRINSIC_WEIGHT * jnp.clip(
        jnp.asarray(prediction_loss), 0.0, 5.0
    )


def select_csp_cluster(embedding, centroids) -> int:
    vector = np.asarray(embedding, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if centers.ndim != 2 or vector.shape != centers.shape[1:]:
        raise ValueError("CSP embedding and centroid dimensions do not match")
    return int(np.argmin(np.square(centers - vector[None, :]).sum(axis=-1)))


def deterministic_kmeans(
    embeddings,
    *,
    validation_embeddings=None,
    minimum_clusters: int = 2,
    maximum_clusters: int = 6,
    seed: int,
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], int, float]:
    """Select k by silhouette with the registered small-score tie rule."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    values = np.asarray(embeddings, dtype=np.float64)
    validation_values = (
        values
        if validation_embeddings is None
        else np.asarray(validation_embeddings, dtype=np.float64)
    )
    if values.ndim != 2 or values.shape[0] < minimum_clusters + 1:
        raise ValueError("CSP clustering requires a non-trivial embedding matrix")
    maximum = min(maximum_clusters, values.shape[0] - 1)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for cluster_count in range(minimum_clusters, maximum + 1):
        estimator = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed)
        labels = estimator.fit_predict(values)
        if validation_embeddings is None:
            score = float(silhouette_score(values, labels))
        else:
            validation_labels = estimator.predict(validation_values)
            if len(set(int(item) for item in validation_labels)) < 2:
                score = -1.0
            else:
                score = float(silhouette_score(validation_values, validation_labels))
        candidates.append((score, cluster_count, labels, estimator.cluster_centers_))
    best_score = max(item[0] for item in candidates)
    eligible = [item for item in candidates if best_score - item[0] <= 1e-3]
    score, count, labels, centers = min(eligible, key=lambda item: item[1])
    return (
        tuple(int(value) for value in labels),
        tuple(tuple(float(value) for value in row) for row in centers),
        count,
        score,
    )
