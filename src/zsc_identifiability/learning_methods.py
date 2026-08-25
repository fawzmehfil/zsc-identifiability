"""Compact neural baselines used by the Stage 4 audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised without the learning extra
    raise RuntimeError("Stage 4 learning requires `uv sync --extra learning`") from exc

from zsc_identifiability.learning_models import LearningMethodSpec


@dataclass(frozen=True)
class NetworkOutput:
    logits: Tensor
    value: Tensor
    hidden: Tensor
    identity_logits: Tensor | None = None
    response_logits: Tensor | None = None
    latent_mean: Tensor | None = None
    latent_log_variance: Tensor | None = None
    cluster_logits: Tensor | None = None
    cluster_response_logits: Tensor | None = None


class LearnedPolicy(nn.Module):
    """Common policy interface for feed-forward and context-aware baselines."""

    method_id: str
    hidden_size: int
    recurrent: bool

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        raise NotImplementedError

    def sequence_auxiliary(
        self,
        observations: Tensor,
        valid: Tensor,
        response_targets: Tensor,
        outputs: list[NetworkOutput],
    ) -> Tensor | None:
        """Optional training-only full-trajectory objective."""
        del observations, valid, response_targets, outputs
        return None

    @staticmethod
    def mask_logits(logits: Tensor, masks: Tensor) -> Tensor:
        if masks.dtype is not torch.bool:
            masks = masks.bool()
        return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


class MLPActorCritic(LearnedPolicy):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.method_id = "mlp_ppo"
        self.hidden_size = hidden_size
        self.recurrent = False
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        del hidden
        features = self.encoder(observations)
        logits = self.mask_logits(self.policy_head(features), action_masks)
        empty_hidden = torch.zeros(
            observations.shape[0], self.hidden_size, device=observations.device
        )
        return NetworkOutput(logits, self.value_head(features).squeeze(-1), empty_hidden)


class UniformMaskedPolicy(LearnedPolicy):
    """Untrained uniform-valid policy used only as a random-intervention control."""

    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.method_id = "random_intervention"
        self.hidden_size = 1
        self.recurrent = False
        self.action_size = action_size
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        del hidden
        logits = (
            torch.zeros(observations.shape[0], self.action_size, device=observations.device)
            + self.anchor
        )
        logits = self.mask_logits(logits, action_masks)
        value = torch.zeros(observations.shape[0], device=observations.device)
        next_hidden = torch.zeros(observations.shape[0], 1, device=observations.device)
        return NetworkOutput(logits, value, next_hidden)


class GRUActorCritic(LearnedPolicy):
    def __init__(
        self,
        method_id: str,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        partner_identity_count: int,
        response_count: int,
        latent_dimension: int,
    ) -> None:
        super().__init__()
        self.method_id = method_id
        self.hidden_size = hidden_size
        self.recurrent = True
        self.input_encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.identity_head = (
            nn.Linear(hidden_size, partner_identity_count)
            if method_id in {"pace_aux", "pace_style", "csp_style_reconnaissance"}
            else None
        )
        self.response_head = (
            nn.Linear(hidden_size, response_count)
            if method_id == "csp_style_reconnaissance"
            else None
        )
        self.latent_mean_head = None
        self.latent_log_variance_head = None

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        encoded = self.input_encoder(observations)
        next_hidden = self.gru(encoded, hidden)
        latent_mean = (
            self.latent_mean_head(next_hidden) if self.latent_mean_head is not None else None
        )
        latent_log_variance = (
            self.latent_log_variance_head(next_hidden)
            if self.latent_log_variance_head is not None
            else None
        )
        policy_features = (
            torch.cat((next_hidden, latent_mean), dim=-1)
            if latent_mean is not None
            else next_hidden
        )
        raw_logits = self.policy_head(policy_features)
        logits = self.mask_logits(raw_logits, action_masks)
        return NetworkOutput(
            logits=logits,
            value=self.value_head(next_hidden).squeeze(-1),
            hidden=next_hidden,
            identity_logits=(
                self.identity_head(next_hidden) if self.identity_head is not None else None
            ),
            response_logits=(
                self.response_head(next_hidden) if self.response_head is not None else None
            ),
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
        )


class ODITSStylePolicy(LearnedPolicy):
    """Online proxy latent aligned to a training-only full-trajectory posterior."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        response_count: int,
        latent_dimension: int,
    ) -> None:
        super().__init__()
        self.method_id = "odits_style"
        self.hidden_size = hidden_size
        self.recurrent = True
        self.proxy_encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.proxy_gru = nn.GRUCell(hidden_size, hidden_size)
        self.proxy_mean = nn.Linear(hidden_size, latent_dimension)
        self.proxy_log_variance = nn.Linear(hidden_size, latent_dimension)
        self.full_encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.full_gru = nn.GRUCell(hidden_size, hidden_size)
        self.full_mean = nn.Linear(hidden_size, latent_dimension)
        self.full_log_variance = nn.Linear(hidden_size, latent_dimension)
        self.full_response_head = nn.Linear(hidden_size, response_count)
        self.policy_head = nn.Linear(hidden_size + latent_dimension, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.response_head = nn.Linear(hidden_size, response_count)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        next_hidden = self.proxy_gru(self.proxy_encoder(observations), hidden)
        mean = self.proxy_mean(next_hidden)
        log_variance = self.proxy_log_variance(next_hidden).clamp(-10, 10)
        policy_features = torch.cat((next_hidden, mean), dim=-1)
        return NetworkOutput(
            logits=self.mask_logits(self.policy_head(policy_features), action_masks),
            value=self.value_head(next_hidden).squeeze(-1),
            hidden=next_hidden,
            response_logits=self.response_head(next_hidden),
            latent_mean=mean,
            latent_log_variance=log_variance,
        )

    def sequence_auxiliary(
        self,
        observations: Tensor,
        valid: Tensor,
        response_targets: Tensor,
        outputs: list[NetworkOutput],
    ) -> Tensor | None:
        batch_size = observations.shape[0]
        teacher_hidden = torch.zeros(batch_size, self.hidden_size, device=observations.device)
        for time in range(observations.shape[1]):
            candidate = self.full_gru(self.full_encoder(observations[:, time]), teacher_hidden)
            teacher_hidden = torch.where(valid[:, time, None], candidate, teacher_hidden)
        teacher_mean = self.full_mean(teacher_hidden)
        teacher_log_variance = self.full_log_variance(teacher_hidden).clamp(-10, 10)
        proxy_mean = torch.stack(
            [output.latent_mean for output in outputs if output.latent_mean is not None], dim=1
        )
        proxy_log_variance = torch.stack(
            [
                output.latent_log_variance
                for output in outputs
                if output.latent_log_variance is not None
            ],
            dim=1,
        )
        teacher_mean = teacher_mean[:, None, :].expand_as(proxy_mean).detach()
        teacher_log_variance = (
            teacher_log_variance[:, None, :].expand_as(proxy_log_variance).detach()
        )
        proxy_to_teacher = 0.5 * (
            teacher_log_variance
            - proxy_log_variance
            + (proxy_log_variance.exp() + (proxy_mean - teacher_mean).square())
            / teacher_log_variance.exp()
            - 1
        )
        alignment = proxy_to_teacher[valid].mean()
        teacher_prior = (
            -0.5
            * (
                1
                + self.full_log_variance(teacher_hidden).clamp(-10, 10)
                - self.full_mean(teacher_hidden).square()
                - self.full_log_variance(teacher_hidden).clamp(-10, 10).exp()
            ).mean()
        )
        last_indices = valid.long().sum(dim=1) - 1
        last_targets = response_targets[
            torch.arange(batch_size, device=observations.device), last_indices
        ]
        target_valid = last_targets >= 0
        reconstruction = (
            torch.nn.functional.cross_entropy(
                self.full_response_head(teacher_hidden[target_valid]),
                last_targets[target_valid],
            )
            if target_valid.any()
            else torch.zeros((), device=observations.device)
        )
        result: Tensor = alignment + teacher_prior + reconstruction
        return result


class TalentsStylePolicy(LearnedPolicy):
    """Latent strategy encoder, fixed-share belief, and specialist mixture."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        core_hidden_size: int,
        response_count: int,
        latent_dimension: int,
        share: float = 0.05,
    ) -> None:
        super().__init__()
        self.method_id = "talents_style"
        self.core_hidden_size = core_hidden_size
        self.hidden_size = core_hidden_size + 2
        self.recurrent = True
        self.share = share
        self.pretraining_metadata: dict[str, Any] = {}
        self.input_encoder = nn.Sequential(nn.Linear(observation_size, core_hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(core_hidden_size, core_hidden_size)
        self.latent_mean_head = nn.Linear(core_hidden_size, latent_dimension)
        self.latent_log_variance_head = nn.Linear(core_hidden_size, latent_dimension)
        self.prototypes = nn.Parameter(torch.randn(2, latent_dimension) * 0.1)
        self.response_head = nn.Linear(core_hidden_size, response_count)
        self.specialist_heads = nn.ModuleList(
            [nn.Linear(core_hidden_size, action_size) for _ in range(2)]
        )
        self.value_head = nn.Linear(core_hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        core = torch.zeros(batch_size, self.core_hidden_size, device=device)
        belief = torch.full((batch_size, 2), 0.5, device=device)
        return torch.cat((core, belief), dim=-1)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        core, prior = hidden[:, : self.core_hidden_size], hidden[:, self.core_hidden_size :]
        next_core = self.gru(self.input_encoder(observations), core)
        latent_mean = self.latent_mean_head(next_core)
        latent_log_variance = self.latent_log_variance_head(next_core)
        squared_distance = (latent_mean[:, None, :] - self.prototypes[None, :, :]).square().sum(-1)
        likelihood = torch.softmax(-squared_distance, dim=-1)
        posterior = prior * likelihood
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        posterior = (1 - self.share) * posterior + self.share / 2
        specialist_logits = torch.stack([head(next_core) for head in self.specialist_heads], dim=1)
        raw_logits = (posterior.unsqueeze(-1) * specialist_logits).sum(dim=1)
        return NetworkOutput(
            logits=self.mask_logits(raw_logits, action_masks),
            value=self.value_head(next_core).squeeze(-1),
            hidden=torch.cat((next_core, posterior), dim=-1),
            response_logits=self.response_head(next_core),
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
            cluster_logits=torch.log(posterior.clamp_min(1e-12)),
        )


class TomSelectorStylePolicy(LearnedPolicy):
    """Global/cluster response predictors with KL-based specialist selection."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        response_count: int,
    ) -> None:
        super().__init__()
        self.method_id = "tom_selector_style"
        self.hidden_size = hidden_size
        self.recurrent = True
        self.training_clusters: Tensor | None = None
        self.input_encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.global_predictor = nn.Linear(hidden_size, response_count)
        self.cluster_predictors = nn.ModuleList(
            [nn.Linear(hidden_size, response_count) for _ in range(2)]
        )
        self.specialist_heads = nn.ModuleList(
            [nn.Linear(hidden_size, action_size) for _ in range(2)]
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward_step(
        self, observations: Tensor, hidden: Tensor, action_masks: Tensor
    ) -> NetworkOutput:
        next_hidden = self.gru(self.input_encoder(observations), hidden)
        global_logits = self.global_predictor(next_hidden)
        global_log_probabilities = torch.log_softmax(global_logits, dim=-1)
        cluster_prediction_logits = torch.stack(
            [head(next_hidden) for head in self.cluster_predictors], dim=1
        )
        cluster_probabilities = torch.softmax(cluster_prediction_logits, dim=-1)
        cluster_log_probabilities = torch.log_softmax(cluster_prediction_logits, dim=-1)
        divergences = (
            cluster_probabilities
            * (cluster_log_probabilities - global_log_probabilities[:, None, :])
        ).sum(-1)
        weights = (
            torch.nn.functional.one_hot(self.training_clusters, num_classes=2).to(divergences.dtype)
            if self.training_clusters is not None
            else torch.softmax(divergences, dim=-1)
        )
        specialist_logits = torch.stack(
            [head(next_hidden) for head in self.specialist_heads], dim=1
        )
        raw_logits = (weights.unsqueeze(-1) * specialist_logits).sum(dim=1)
        return NetworkOutput(
            logits=self.mask_logits(raw_logits, action_masks),
            value=self.value_head(next_hidden).squeeze(-1),
            hidden=next_hidden,
            response_logits=global_logits,
            cluster_logits=divergences,
            cluster_response_logits=cluster_prediction_logits,
        )


def build_method(
    method: LearningMethodSpec,
    observation_size: int,
    action_size: int,
    partner_identity_count: int,
    response_count: int,
) -> LearnedPolicy:
    if method.method_id == "mlp_ppo":
        return MLPActorCritic(observation_size, action_size, method.config.hidden_size)
    if method.method_id == "odits_style":
        return ODITSStylePolicy(
            observation_size,
            action_size,
            method.config.hidden_size,
            response_count,
            method.config.latent_dimension,
        )
    if method.method_id == "talents_style":
        return TalentsStylePolicy(
            observation_size,
            action_size,
            method.config.hidden_size,
            response_count,
            method.config.latent_dimension,
        )
    if method.method_id == "tom_selector_style":
        return TomSelectorStylePolicy(
            observation_size,
            action_size,
            method.config.hidden_size,
            response_count,
        )
    return GRUActorCritic(
        method.method_id,
        observation_size,
        action_size,
        method.config.hidden_size,
        partner_identity_count,
        response_count,
        method.config.latent_dimension,
    )


def action_distribution(output: NetworkOutput) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=output.logits)


def model_metadata(model: LearnedPolicy) -> dict[str, Any]:
    return {
        "class": model.__class__.__name__,
        "method_id": model.method_id,
        "hidden_size": model.hidden_size,
        "recurrent": model.recurrent,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
