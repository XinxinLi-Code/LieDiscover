"""Rollout collection for encoder-decoder symbolic policies.

The public entry point is:

    episodes = collect_rollouts(policy, env, batch_size)

Each episode performs:

    env.reset(theta0)
    while not done:
        action = policy.sample_action(theta_acc, theta_rej, h=h_k, k=k, K=K)
        env.step(action)
    env.finalize()

The resulting ``RolloutEpisode`` objects keep the decoder log-probabilities
state values, and tree entropies with their computation graph, while rewards
come from the SVD-filtered environment steps in ``reward.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch

from policy_model import PolicyAction, SymbolicPolicy
from rl_env import EnvFinal, EnvObservation, EnvStep, LieThetaEnv
from rs_ac_loss import _mc_return_to_go as mc_return_to_go
from sampling_utils import build_seeded_permutation_loader
from theta_utils import ThetaTermError, theta_term_to_prefix


DEFAULT_THETA0: Tuple[str, ...] = ("1", "t", "x", "u")


@dataclass(frozen=True)
class RolloutStep:
    """One policy action and the environment response it caused."""

    index: int
    observation: EnvObservation
    action: PolicyAction
    env_step: EnvStep
    term_for_env: object

    @property
    def log_prob(self) -> torch.Tensor:
        return self.action.log_prob

    @property
    def state_value(self) -> torch.Tensor:
        return self.action.state_value

    @property
    def tree_entropy(self) -> torch.Tensor:
        return self.action.tree_entropy

    @property
    def mean_tree_entropy(self) -> torch.Tensor:
        return self.action.mean_tree_entropy

    @property
    def accepted(self) -> bool:
        return bool(self.env_step.check.accepted)

    @property
    def canonical_term(self) -> str:
        return self.env_step.check.term


@dataclass(frozen=True)
class RolloutEpisode:
    """One completed episode with final SVD-filtered step rewards."""

    index: int
    theta0: Tuple[str, ...]
    steps: Tuple[RolloutStep, ...]
    final: EnvFinal
    end_reason: str = "finalized"

    @property
    def log_probs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(step.log_prob for step in self.steps)

    @property
    def state_values(self) -> Tuple[torch.Tensor, ...]:
        return tuple(step.state_value for step in self.steps)

    @property
    def tree_entropies(self) -> Tuple[torch.Tensor, ...]:
        return tuple(step.tree_entropy for step in self.steps)

    @property
    def mean_tree_entropies(self) -> Tuple[torch.Tensor, ...]:
        return tuple(step.mean_tree_entropy for step in self.steps)

    @property
    def rewards(self) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.final.result.rewards)

    @property
    def returns_to_go(self) -> Tuple[float, ...]:
        return mc_return_to_go(self.rewards)

    @property
    def theta_acc(self) -> Tuple[str, ...]:
        return tuple(self.final.result.theta_terms)

    @property
    def delta_D(self) -> int:
        return int(self.final.result.delta_D)

    @property
    def done_reason(self) -> str:
        if self.end_reason:
            return self.end_reason
        if self.steps and self.steps[-1].env_step.done:
            return "search_horizon"
        return "finalized"


def _as_theta0(theta0: Iterable[str]) -> Tuple[str, ...]:
    values = tuple(str(term) for term in theta0)
    if not values:
        raise ValueError("theta0 must contain at least one term.")
    return values


def _action_term_for_env(action: PolicyAction) -> object:
    """Choose the richest representation that ``rl_env.step`` can normalize."""

    if action.tree is not None:
        return action.tree
    if action.valid_action_ids:
        return action.valid_action_ids
    if action.expression is not None:
        return action.expression
    return "0"


def _forbidden_terms_for_observation(
    policy: SymbolicPolicy,
    observation: EnvObservation,
    extra_forbidden_terms: Sequence[Tuple[int, ...]],
) -> Tuple[object, ...]:
    prefixes = []
    for term in observation.theta_acc + observation.theta_rej:
        try:
            prefixes.append(
                theta_term_to_prefix(
                    term,
                    library=policy.decoder.library,
                    variable_names=policy.variable_names,
                )
            )
        except ThetaTermError:
            # Some canonical forms simplify to tokens this decoder cannot emit
            # (for example negative terms when the current library has no neg/sub).
            # They cannot be generated in the current rollout, so they do not
            # need to participate in decoder-level forbidden masking.
            continue
    return tuple(prefixes) + tuple(extra_forbidden_terms)


def collect_rollouts(
    policy: SymbolicPolicy,
    env: LieThetaEnv,
    batch_size: int,
    *,
    theta0: Sequence[str] = DEFAULT_THETA0,
    max_steps: Optional[int] = None,
    max_len: Optional[int] = None,
    greedy: bool = False,
    temperature: float = 1.0,
    strict_prefix_ban: bool = False,
    forbid_observed: bool = True,
    pad_to: Optional[int] = None,
    episode_callback: Optional[Callable[[RolloutEpisode], None]] = None,
    **sample_kwargs,
) -> Tuple[RolloutEpisode, ...]:
    """Collect a batch of complete symbolic episodes.

    Episodes are sampled sequentially using the same mutable ``env`` instance.
    This keeps the environment API simple and avoids copying the PDE model or
    data loader. The caller can still treat the returned tuple as a batch for
    policy-gradient optimization.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    base_theta = _as_theta0(theta0)
    episodes: List[RolloutEpisode] = []

    for episode_index in range(int(batch_size)):
        observation = env.reset(base_theta, max_steps=max_steps)
        steps: List[RolloutStep] = []
        extra_forbidden_terms: List[Tuple[int, ...]] = []
        local_step = 0
        end_reason = "finalized"

        while not observation.done:
            action_kwargs = dict(sample_kwargs)
            if forbid_observed and "forbidden_terms" not in action_kwargs:
                action_kwargs["forbidden_terms"] = _forbidden_terms_for_observation(
                    policy,
                    observation,
                    extra_forbidden_terms,
                )
                action_kwargs.setdefault("forbidden_terms_encoding", "dso")

            action = policy.sample_action(
                observation.theta_acc,
                observation.theta_rej,
                h=observation.h,
                k=observation.k,
                K=observation.K,
                max_len=max_len,
                greedy=greedy,
                temperature=temperature,
                strict_prefix_ban=strict_prefix_ban,
                forbid_observed=forbid_observed,
                pad_to=pad_to,
                **action_kwargs,
            )
            term_for_env = _action_term_for_env(action)
            env_step = env.step(term_for_env, library=policy.decoder.library)
            if not env_step.check.accepted and action.valid_action_ids:
                extra_forbidden_terms.append(action.valid_action_ids)

            steps.append(
                RolloutStep(
                    index=local_step,
                    observation=observation,
                    action=action,
                    env_step=env_step,
                    term_for_env=term_for_env,
                )
            )
            observation = env_step.observation
            local_step += 1

        if observation.done:
            end_reason = "search_horizon"

        final = env.finalize()
        episode = RolloutEpisode(
            index=episode_index,
            theta0=base_theta,
            steps=tuple(steps),
            final=final,
            end_reason=end_reason,
        )
        episodes.append(episode)
        if episode_callback is not None:
            episode_callback(episode)

    return tuple(episodes)
