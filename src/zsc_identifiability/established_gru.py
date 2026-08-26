"""Training-only GRU posterior used by the cross-fitted Stage 6 DRI audit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from zsc_identifiability.established_dri import PosteriorRiskSummary, summarize_posteriors


@dataclass(frozen=True)
class GRUPosteriorResult:
    posteriors: np.ndarray
    temperature: float
    summary: PosteriorRiskSummary
    validation_cross_entropy: float
    epochs_completed: int


def fit_cross_fitted_gru_posterior(
    calibration_sequences: Sequence[np.ndarray],
    calibration_labels: Sequence[int],
    validation_sequences: Sequence[np.ndarray],
    validation_labels: Sequence[int],
    confirmatory_sequences: Sequence[np.ndarray],
    confirmatory_labels: Sequence[int],
    prior: Sequence[float],
    loss_matrix: Sequence[Sequence[float]],
    *,
    response_signatures: Sequence[str] | None = None,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 12,
    seed: int = 6173,
) -> GRUPosteriorResult:
    """Fit on calibration, calibrate on validation, evaluate on confirmatory data.

    Confirmatory labels are used only after inference to score the coordination
    response selected from each posterior. They never enter optimization,
    calibration, or the deployed policy.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional established extra
        raise RuntimeError("GRU DRI requires: uv sync --extra established") from exc
    if not calibration_sequences or not validation_sequences or not confirmatory_sequences:
        raise ValueError("calibration, validation, and confirmatory sequence sets are required")
    feature_sizes = {
        int(sequence.shape[1])
        for sequence in (*calibration_sequences, *validation_sequences, *confirmatory_sequences)
        if sequence.ndim == 2
    }
    if len(feature_sizes) != 1:
        raise ValueError("all GRU sequences must be rank-two with a common feature width")
    mode_count = len(prior)
    if any(label < 0 or label >= mode_count for label in (*calibration_labels, *validation_labels)):
        raise ValueError("GRU partner label is outside the finite evaluation-mode set")
    if len(calibration_sequences) != len(calibration_labels):
        raise ValueError("calibration sequence/label counts differ")
    if len(validation_sequences) != len(validation_labels):
        raise ValueError("validation sequence/label counts differ")
    if len(confirmatory_sequences) != len(confirmatory_labels):
        raise ValueError("confirmatory sequence/label counts differ")
    if any(label < 0 or label >= mode_count for label in confirmatory_labels):
        raise ValueError("GRU confirmatory label is outside the finite mode set")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    class PosteriorGRU(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.gru = nn.GRU(width, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, mode_count)

        def forward(self, batch: Any, lengths: Any) -> Any:
            packed = nn.utils.rnn.pack_padded_sequence(
                batch, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            return self.head(hidden[-1])

    model = PosteriorGRU(next(iter(feature_sizes)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_batch, train_lengths = _pad(calibration_sequences, torch)
    train_targets = torch.tensor(calibration_labels, dtype=torch.long)
    validation_batch, validation_lengths = _pad(validation_sequences, torch)
    validation_targets = torch.tensor(validation_labels, dtype=torch.long)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale = 0
    completed = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_batch, train_lengths), train_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(model(validation_batch, validation_lengths), validation_targets).item()
            )
        completed = epoch + 1
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("GRU posterior training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_batch, validation_lengths)
    temperature = _select_temperature(validation_logits, validation_targets, torch)
    confirm_batch, confirm_lengths = _pad(confirmatory_sequences, torch)
    with torch.no_grad():
        logits = model(confirm_batch, confirm_lengths) / temperature
        posteriors = torch.softmax(logits, dim=-1).cpu().numpy()
    summary = summarize_posteriors(
        prior,
        loss_matrix,
        posteriors.tolist(),
        response_signatures=response_signatures,
        true_modes=confirmatory_labels,
    )
    return GRUPosteriorResult(
        posteriors=posteriors,
        temperature=temperature,
        summary=summary,
        validation_cross_entropy=best_loss,
        epochs_completed=completed,
    )


def _pad(sequences: Sequence[np.ndarray], torch: Any) -> tuple[Any, Any]:
    # `torch` is passed in to keep this module importable without the optional extra.
    lengths = [len(sequence) for sequence in sequences]
    if any(length <= 0 for length in lengths):
        raise ValueError("GRU sequences cannot be empty; use an explicit zero-step token")
    width = sequences[0].shape[1]
    batch = np.zeros((len(sequences), max(lengths), width), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        batch[index, : len(sequence)] = sequence
    return (
        torch.tensor(batch, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
    )


def _select_temperature(logits: Any, labels: Any, torch: Any) -> float:
    candidates = np.geomspace(0.25, 4.0, 81)
    best = (float("inf"), 1.0)
    for candidate in candidates:
        value = float(
            torch.nn.functional.cross_entropy(logits / float(candidate), labels).item()
        )
        if value < best[0] - 1e-12:
            best = (value, float(candidate))
    return best[1]
