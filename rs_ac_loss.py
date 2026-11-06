"""Risk-seeking actor-critic loss for symbolic rollout episodes.

This module implements the tree-level RS-AC objective:

    1. Score each complete theta set by total return.
    2. Keep risk-seeking elite episodes for critic updates.
    3. Update the actor only from successful episodes with DeltaD > 0.
    4. If no successful episode exists in the batch, skip actor updates and
       keep only critic/entropy updates.
    5. Apply entropy regularization over all sampled episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class EpisodeACRecord:
    """Score and validity diagnostics for one rollout episode."""

    index: int
    score: float
    valid: bool
    reason: str
    D0: Optional[int]
    D_best: Optional[int]
    delta_D: Optional[int]
    reward_sum: float
    step_count: int
    tree_length_sum: float


@dataclass(frozen=True)
class RSACLossStats:
    """Diagnostics returned together with the RS-AC loss."""

    total_episodes: int
    valid_episodes: int
    success_episodes: int
    elite_episodes: int
    loss_term_count: int
    actor_term_count: int
    critic_term_count: int
    entropy_term_count: int
    elite_fraction: float
    elite_threshold: float
    mean_valid_score: float
    mean_elite_score: float
    mean_total_return: float
    max_total_return: float
    selected_indices: Tuple[int, ...]
    actor_selected_indices: Tuple[int, ...]
    critic_selected_indices: Tuple[int, ...]
    records: Tuple[EpisodeACRecord, ...]
    total_loss: float
    actor_loss: float
    critic_loss: float
    entropy_loss: float
    critic_coef: float
    entropy_beta: float
    mean_return: float
    mean_value: float
    mean_advantage: float
    mean_entropy: float


def _episode_index(episode: Any, fallback: int) -> int:
    return int(getattr(episode, "index", fallback))


def _episode_result(episode: Any) -> Any:
    return episode.final.result


def _episode_rewards(episode: Any) -> Tuple[float, ...]:
    return tuple(float(x) for x in episode.rewards)


def _mc_return_to_go(rewards: Sequence[float]) -> Tuple[float, ...]:
    """Monte Carlo return-to-go for tree-level rewards.

    For rewards ``r_0, ..., r_{K-1}``, return
    ``G_k = sum_{j=k}^{K-1} r_j`` with no discount.
    """
    returns: List[float] = []
    running = 0.0
    for reward in reversed(tuple(float(x) for x in rewards)):
        running += reward
        returns.append(running)
    returns.reverse()
    return tuple(returns)


def _episode_returns_to_go(episode: Any) -> Tuple[float, ...]:
    if hasattr(episode, "returns_to_go"):
        return tuple(float(x) for x in episode.returns_to_go)
    return _mc_return_to_go(_episode_rewards(episode))


def _tree_length_sum(episode: Any) -> float:
    total = 0.0
    for step in getattr(episode, "steps", ()):
        check = getattr(step.env_step, "check", None)
        if check is not None and hasattr(check, "tree_length"):
            total += float(check.tree_length)
        else:
            total += float(getattr(step.action, "length", 1))
    return total


def _has_finite_tensors(tensors: Sequence[torch.Tensor]) -> bool:
    for tensor in tensors:
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            return False
    return True


def _step_log_probs(episode: Any) -> Tuple[torch.Tensor, ...]:
    return tuple(step.log_prob for step in getattr(episode, "steps", ()))


def _step_values(episode: Any) -> Tuple[torch.Tensor, ...]:
    return tuple(step.state_value for step in getattr(episode, "steps", ()))


def _step_entropies(episode: Any, *, entropy_mode: str) -> Tuple[torch.Tensor, ...]:
    if entropy_mode == "sum":
        return tuple(step.tree_entropy for step in getattr(episode, "steps", ()))
    if entropy_mode == "mean":
        return tuple(step.mean_tree_entropy for step in getattr(episode, "steps", ()))
    raise ValueError("entropy_mode must be 'sum' or 'mean'.")


def episode_score(episode: Any) -> float:
    """Score used for risk-seeking elite selection: total episode reward."""

    return float(sum(_episode_rewards(episode)))


def _first_tensor_or_none(episodes: Sequence[Any]) -> Optional[torch.Tensor]:
    for episode in episodes:
        for step in getattr(episode, "steps", ()):
            for name in ("log_prob", "state_value", "tree_entropy", "mean_tree_entropy"):
                if hasattr(step, name):
                    value = getattr(step, name)
                    if torch.is_tensor(value):
                        return value
    return None


def _zero_loss(episodes: Sequence[Any]) -> torch.Tensor:
    first = _first_tensor_or_none(episodes)
    if first is not None:
        return first.reshape(()).sum() * 0.0
    return torch.tensor(0.0, requires_grad=True)


def _validate_and_score_episode(
    episode: Any,
    *,
    fallback_index: int,
    expected_D0: Optional[int],
    max_reasonable_D: Optional[int],
    max_delta_D: Optional[int],
    min_steps: int,
    entropy_mode: str,
) -> EpisodeACRecord:
    result = _episode_result(episode)
    rewards = _episode_rewards(episode)
    returns = _episode_returns_to_go(episode)
    steps = tuple(getattr(episode, "steps", ()))
    step_count = len(steps)
    reward_sum = float(sum(rewards))
    tree_len = _tree_length_sum(episode)
    index = _episode_index(episode, fallback_index)

    D0 = int(result.D0) if hasattr(result, "D0") else None
    D_best = int(result.D_best) if hasattr(result, "D_best") else None
    delta_D = int(result.delta_D) if hasattr(result, "delta_D") else None

    reason = "valid"
    valid = True
    try:
        log_probs = _step_log_probs(episode)
        values = _step_values(episode)
        entropies = _step_entropies(episode, entropy_mode=entropy_mode)
    except AttributeError as exc:
        log_probs, values, entropies = (), (), ()
        valid = False
        reason = f"missing_tensor:{exc}"

    if valid and not bool(getattr(result, "valid", True)):
        valid = False
        reason = str(getattr(result, "invalid_reason", "marked_invalid"))
    elif valid and step_count < int(min_steps):
        valid = False
        reason = "too_few_steps"
    elif valid and step_count != len(rewards):
        valid = False
        reason = "step_reward_length_mismatch"
    elif valid and step_count != len(returns):
        valid = False
        reason = "step_return_length_mismatch"
    elif valid and step_count != len(log_probs):
        valid = False
        reason = "step_logprob_length_mismatch"
    elif valid and step_count != len(values):
        valid = False
        reason = "step_value_length_mismatch"
    elif valid and step_count != len(entropies):
        valid = False
        reason = "step_entropy_length_mismatch"
    elif valid and not all(isfinite(x) for x in rewards + returns):
        valid = False
        reason = "non_finite_reward_or_return"
    elif valid and not _has_finite_tensors(log_probs + values + entropies):
        valid = False
        reason = "non_finite_policy_tensor"
    elif valid and expected_D0 is not None and D0 != int(expected_D0):
        valid = False
        reason = f"unexpected_D0:{D0}"
    elif valid and D0 is not None and D_best is not None and D_best < D0:
        valid = False
        reason = f"D_best_lt_D0:{D_best}<{D0}"
    elif valid and max_reasonable_D is not None and D_best is not None and D_best > int(max_reasonable_D):
        valid = False
        reason = f"unreasonable_D_best:{D_best}"
    elif valid and max_delta_D is not None and delta_D is not None and delta_D > int(max_delta_D):
        valid = False
        reason = f"unreasonable_delta_D:{delta_D}"

    score = reward_sum if valid else float("-inf")
    return EpisodeACRecord(
        index=index,
        score=score,
        valid=valid,
        reason=reason,
        D0=D0,
        D_best=D_best,
        delta_D=delta_D,
        reward_sum=reward_sum,
        step_count=step_count,
        tree_length_sum=tree_len,
    )


def _elite_records(
    records: Sequence[EpisodeACRecord],
    *,
    elite_fraction: float,
    min_elite_episodes: int,
) -> Tuple[EpisodeACRecord, ...]:
    valid_records = [record for record in records if record.valid]
    if not valid_records:
        return ()
    valid_records.sort(key=lambda record: record.score, reverse=True)
    elite_count = max(int(min_elite_episodes), int(ceil(float(elite_fraction) * len(valid_records))))
    elite_count = min(elite_count, len(valid_records))
    return tuple(valid_records[:elite_count])


def _success_records(records: Sequence[EpisodeACRecord]) -> Tuple[EpisodeACRecord, ...]:
    success = [
        record
        for record in records
        if record.valid and record.delta_D is not None and int(record.delta_D) > 0
    ]
    success.sort(key=lambda record: (int(record.delta_D or 0), record.score), reverse=True)
    return tuple(success)


def _episode_entropy_sum(episode: Any, *, entropy_mode: str) -> Optional[torch.Tensor]:
    entropies = _step_entropies(episode, entropy_mode=entropy_mode)
    finite = [entropy.reshape(()) for entropy in entropies if bool(torch.isfinite(entropy.detach()).all().item())]
    if not finite:
        return None
    return torch.stack(finite).sum()


def risk_seeking_actor_critic_loss(
    episodes: Sequence[Any],
    *,
    elite_fraction: float = 0.1,
    min_elite_episodes: int = 1,
    critic_coef: float = 1.0,
    entropy_beta: float = 0.01,
    entropy_mode: str = "sum",
    normalize_advantages: bool = True,
    expected_D0: Optional[int] = None,
    max_reasonable_D: Optional[int] = None,
    max_delta_D: Optional[int] = None,
    min_steps: int = 1,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, RSACLossStats]:
    """Compute tree-level risk-seeking actor-critic loss.

    Critic terms are computed on risk-seeking elite complete episodes. Actor
    terms are stricter: if the batch contains any valid episode with
    ``DeltaD > 0``, actor gradients are computed only from those success
    episodes. If the batch has no success episode, actor gradients are skipped
    and the update contains only critic and entropy terms.
    """

    if not 0.0 < float(elite_fraction) <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1].")
    if int(min_elite_episodes) <= 0:
        raise ValueError("min_elite_episodes must be positive.")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")
    if entropy_mode not in {"sum", "mean"}:
        raise ValueError("entropy_mode must be 'sum' or 'mean'.")

    episodes = tuple(episodes)
    zero = _zero_loss(episodes)
    records = tuple(
        _validate_and_score_episode(
            episode,
            fallback_index=idx,
            expected_D0=expected_D0,
            max_reasonable_D=max_reasonable_D,
            max_delta_D=max_delta_D,
            min_steps=min_steps,
            entropy_mode=entropy_mode,
        )
        for idx, episode in enumerate(episodes)
    )
    elites = _elite_records(
        records,
        elite_fraction=elite_fraction,
        min_elite_episodes=min_elite_episodes,
    )
    successes = _success_records(records)
    critic_indices = tuple(record.index for record in elites)
    actor_indices = tuple(record.index for record in successes)
    critic_index_set = set(critic_indices)
    actor_index_set = set(actor_indices)

    entropy_terms: List[torch.Tensor] = []
    for episode in episodes:
        entropy_sum = _episode_entropy_sum(episode, entropy_mode=entropy_mode)
        if entropy_sum is not None:
            entropy_terms.append(entropy_sum.reshape(()))
    if entropy_terms:
        entropy_t = torch.stack(entropy_terms)
        entropy_loss = -float(entropy_beta) * entropy_t.mean()
        mean_entropy = float(entropy_t.detach().mean().cpu())
    else:
        entropy_loss = zero
        entropy_t = None
        mean_entropy = float("nan")

    actor_log_prob_terms: List[torch.Tensor] = []
    actor_value_terms: List[torch.Tensor] = []
    actor_target_terms: List[torch.Tensor] = []
    critic_value_terms: List[torch.Tensor] = []
    critic_target_terms: List[torch.Tensor] = []
    for fallback_idx, episode in enumerate(episodes):
        idx = _episode_index(episode, fallback_idx)
        returns = _episode_returns_to_go(episode)
        for step, return_to_go in zip(getattr(episode, "steps", ()), returns):
            value = step.state_value.reshape(())
            target = torch.as_tensor(
                float(return_to_go),
                dtype=value.dtype,
                device=value.device,
            )
            if idx in actor_index_set:
                actor_log_prob_terms.append(step.log_prob.reshape(()))
                actor_value_terms.append(value)
                actor_target_terms.append(target)
            if idx in critic_index_set:
                critic_value_terms.append(value)
                critic_target_terms.append(target)

    if actor_log_prob_terms:
        actor_log_prob_t = torch.stack(actor_log_prob_terms)
        actor_value_t = torch.stack(actor_value_terms)
        actor_target_t = torch.stack(actor_target_terms)
        advantage_t = (actor_target_t - actor_value_t).detach()
        if normalize_advantages and advantage_t.numel() > 1:
            std = advantage_t.std(unbiased=False).clamp(min=1e-6)
            advantage_t = (advantage_t - advantage_t.mean()) / std

        actor_objective = actor_log_prob_t * advantage_t
        if reduction == "mean":
            actor_loss = -actor_objective.mean()
        else:
            actor_loss = -actor_objective.sum()

        raw_advantage = (actor_target_t - actor_value_t).detach()
        mean_return = float(actor_target_t.detach().mean().cpu())
        mean_value = float(actor_value_t.detach().mean().cpu())
        mean_advantage = float(raw_advantage.mean().cpu())
    else:
        actor_loss = zero
        mean_return = float("nan")
        mean_value = float("nan")
        mean_advantage = float("nan")

    if critic_value_terms:
        critic_value_t = torch.stack(critic_value_terms)
        critic_target_t = torch.stack(critic_target_terms)
        critic_terms = 0.5 * (critic_value_t - critic_target_t).pow(2)
        if reduction == "mean":
            critic_loss = critic_terms.mean()
        else:
            critic_loss = critic_terms.sum()
        if not actor_log_prob_terms:
            mean_return = float(critic_target_t.detach().mean().cpu())
            mean_value = float(critic_value_t.detach().mean().cpu())
            mean_advantage = float((critic_target_t - critic_value_t).detach().mean().cpu())
    else:
        critic_loss = zero

    loss = actor_loss + float(critic_coef) * critic_loss + entropy_loss

    valid_scores = [record.score for record in records if record.valid]
    elite_scores = [record.score for record in elites]
    elite_threshold = float(min(elite_scores)) if elite_scores else float("nan")
    mean_total_return = (
        float(sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan")
    )
    max_total_return = float(max(valid_scores)) if valid_scores else float("nan")
    stats = RSACLossStats(
        total_episodes=len(episodes),
        valid_episodes=len(valid_scores),
        success_episodes=len(successes),
        elite_episodes=len(elites),
        loss_term_count=len(actor_log_prob_terms) + len(critic_value_terms) + len(entropy_terms),
        actor_term_count=len(actor_log_prob_terms),
        critic_term_count=len(critic_value_terms),
        entropy_term_count=len(entropy_terms),
        elite_fraction=float(elite_fraction),
        elite_threshold=elite_threshold,
        mean_valid_score=float(sum(valid_scores) / max(len(valid_scores), 1)) if valid_scores else float("nan"),
        mean_elite_score=float(sum(elite_scores) / max(len(elite_scores), 1)) if elite_scores else float("nan"),
        mean_total_return=mean_total_return,
        max_total_return=max_total_return,
        selected_indices=actor_indices,
        actor_selected_indices=actor_indices,
        critic_selected_indices=critic_indices,
        records=records,
        total_loss=float(loss.detach().cpu()),
        actor_loss=float(actor_loss.detach().cpu()),
        critic_loss=float(critic_loss.detach().cpu()),
        entropy_loss=float(entropy_loss.detach().cpu()),
        critic_coef=float(critic_coef),
        entropy_beta=float(entropy_beta),
        mean_return=mean_return,
        mean_value=mean_value,
        mean_advantage=mean_advantage,
        mean_entropy=mean_entropy,
    )
    return loss, stats


__all__ = [
    "EpisodeACRecord",
    "RSACLossStats",
    "episode_score",
    "risk_seeking_actor_critic_loss",
]
