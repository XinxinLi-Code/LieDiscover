"""RL environment wrapper for symbolic Theta expansion.

The environment is intentionally thin: it delegates the numerical work to
``reward.py`` and exposes the RL-facing lifecycle:

    env.reset(theta0)
    env.step(term)
    env.finalize()

Each episode runs for K proposals, including rejected proposals. The MDP state
is (Theta_acc, Theta_rej, h, k), with counters initialized to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from reward import (
    EpisodeGramState,
    StepCheck,
    StepHealthConfig,
    StepRewardResult,
    build_episode_state,
    clone_episode_initial_state,
    finalize_episode_with_svd,
    compute_step_reward_result_fast,
    validate_reward_eta,
    try_add_term_with_svd,
    evaluate_library_subset,
)
from spectral import DEFAULT_SPECTRAL_FLOOR, estimate_spectral_cut
from library_evaluation_cache import LibraryEvaluationCache
from sampling_utils import build_seeded_permutation_loader
from theta_utils import NormalizedThetaTerm, TermLike, split_theta_terms


ThetaInput = Union[str, Sequence[str]]


@dataclass(frozen=True)
class EnvObservation:
    """Current symbolic state exposed to the policy."""

    theta_acc: Tuple[str, ...]
    theta_rej: Tuple[str, ...]
    h: int
    k: int
    K: int
    done: bool
    theta_size: int
    max_theta_size: int


@dataclass(frozen=True)
class EnvStep:
    """Result returned by ``LieThetaEnv.step``."""

    observation: EnvObservation
    check: StepCheck
    done: bool


@dataclass(frozen=True)
class EnvFinal:
    """Final episode result returned by ``LieThetaEnv.finalize``."""

    result: StepRewardResult
    final_S: torch.Tensor
    final_Vh: torch.Tensor


class LieThetaEnv:
    """Episode environment for RL-based construction of ``Theta``.

    Parameters
    ----------
    model, train_loader:
        The trained PDE surrogate and sample loader used by ``reward.py``.
    N:
        Library-size configuration used to infer K when no horizon is supplied.
        It does not terminate an episode.
    k_max:
        Fixed number K of proposals per episode. ``reward_k_max`` is retained
        as a legacy horizon alias and must agree when both are supplied.
    """

    def __init__(
        self,
        *,
        model,
        train_loader,
        task: str,
        device,
        sample: int,
        input_dim: int,
        output_dim: int,
        N: int,
        k_max: Optional[int] = None,
        health_config: StepHealthConfig = StepHealthConfig(),
        reward_alpha: float = 1.0,
        reward_lambda_len: float = 0.02,
        reward_eta: float = 0.98,
        reward_k_max: Optional[int] = None,
        spectral_threshold_min: Optional[float] = None,
        spectral_threshold_max: Optional[float] = None,
        use_threshold_bounds: bool = False,
        scale_eps: float = 1e-12,
        spectral_floor: float = DEFAULT_SPECTRAL_FLOOR,
        initial_D0: Optional[int] = None,
        initial_threshold: Optional[float] = None,
        fix_flag: bool = False,
        fix_seed: int = 0,
        sample_size: int = 100,
        fast_finalize: bool = False,
        reward_verbose: bool = False,
        rank_reference=None,
        rank_config=None,
        numerical_cache_max_bytes=128 * 1024**2,
        numerical_cache_max_entries=256,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.task = task
        self.device = device
        self.sample = int(sample)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be positive.")

        self.health_config = health_config
        self.reward_alpha = float(reward_alpha)
        self.reward_lambda_len = float(reward_lambda_len)
        self.reward_eta = float(reward_eta)
        self.reward_k_max = int(reward_k_max) if reward_k_max is not None else None
        if self.reward_k_max is not None and self.reward_k_max <= 0:
            raise ValueError("reward_k_max must be positive when provided.")
        self.k_max = int(k_max) if k_max is not None else self.reward_k_max
        if self.k_max is not None and self.k_max <= 0:
            raise ValueError("k_max must be positive when provided.")
        if self.reward_k_max is not None and self.k_max != self.reward_k_max:
            raise ValueError("k_max and reward_k_max must specify the same search horizon K.")
        self.use_threshold_bounds = bool(use_threshold_bounds)
        self.spectral_threshold_min = (
            float(spectral_threshold_min) if spectral_threshold_min is not None else None
        )
        self.spectral_threshold_max = (
            float(spectral_threshold_max) if spectral_threshold_max is not None else None
        )
        self.initial_D0 = int(initial_D0) if initial_D0 is not None else None
        self.initial_threshold = float(initial_threshold) if initial_threshold is not None else None
        self.scale_eps = float(scale_eps)
        self.spectral_floor = float(spectral_floor)
        if not 0 < self.scale_eps < float("inf"):
            raise ValueError("scale_eps must be finite and positive.")
        estimate_spectral_cut(
            torch.empty(0), threshold_min=self.spectral_threshold_min,
            threshold_max=self.spectral_threshold_max, spectral_floor=self.spectral_floor,
            fixed_threshold=self.initial_threshold,
            use_threshold_bounds=self.use_threshold_bounds,
        )
        self.fix_flag = bool(fix_flag)
        self.fix_seed = int(fix_seed)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        self.fast_finalize = bool(fast_finalize)
        self.reward_verbose = bool(reward_verbose)
        self.rank_reference = rank_reference
        self.rank_config = rank_config
        if rank_reference is not None and (
            self.initial_threshold is not None or self.use_threshold_bounds
        ):
            raise ValueError(
                "Validated dimension selection does not use a fixed spectral cutoff "
                "or cutoff bounds. Use rank_selection='gap' for legacy cutoff experiments."
            )

        self.state: Optional[EpisodeGramState] = None
        self._initial_state_cache = {}
        self._reference_samples = None
        self._reference_key = None
        self._term_scale_cache = {}
        self._term_ci_cache = {}
        self._evaluation_cache = LibraryEvaluationCache(
            max_bytes=numerical_cache_max_bytes, max_entries=numerical_cache_max_entries,
        )
        self._episode_K: Optional[int] = None
        self.done = False
        self.finalized = False

    def reset(self, theta0: ThetaInput, *, max_steps: Optional[int] = None) -> EnvObservation:
        """Start a new episode from an initial accepted library."""

        theta0_terms = tuple(split_theta_terms(theta0))
        if not theta0_terms:
            raise ValueError("theta0 must contain at least one term.")
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps <= 0:
                raise ValueError("max_steps must be positive when provided.")
            if self.k_max is not None and max_steps != self.k_max:
                raise ValueError("max_steps must agree with the environment search horizon K.")
        self._episode_K = (
            max_steps if max_steps is not None else self.k_max
        )
        self._episode_K = (
            self._episode_K
            if self._episode_K is not None
            else max(int(self.N) - len(theta0_terms), 1)
        )
        validate_reward_eta(self.reward_eta, self._episode_K)

        # Every library evaluation in this environment shares one sample context.
        reference_key = (
            id(self.model), id(self.train_loader), self.task, str(self.device),
            self.sample, self.sample_size, self.fix_flag, self.fix_seed,
            self.scale_eps, self.spectral_floor,
            self.spectral_threshold_min, self.spectral_threshold_max,
            self.initial_threshold, self.use_threshold_bounds,
            id(self.rank_reference), self.rank_config,
        )
        if reference_key != self._reference_key:
            self._initial_state_cache.clear()
            self._reference_samples = None
            self._term_scale_cache = {}
            self._term_ci_cache = {}
            self._evaluation_cache.clear()
            self._reference_key = reference_key
        cache_key = theta0_terms
        if cache_key in self._initial_state_cache:
            self.state = clone_episode_initial_state(self._initial_state_cache[cache_key])
        else:
            built_state = build_episode_state(
                model=self.model,
                train_loader=self.train_loader,
                task=self.task,
                device=self.device,
                sample=self.sample,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                initial_theta_terms=theta0_terms,
                initial_D0=self.initial_D0,
                initial_threshold=self.initial_threshold,
                fix_flag=self.fix_flag,
                fix_seed=self.fix_seed,
                sample_size=self.sample_size,
                verbose=self.reward_verbose,
                threshold_min=self.spectral_threshold_min,
                threshold_max=self.spectral_threshold_max,
                use_threshold_bounds=self.use_threshold_bounds,
                spectral_floor=self.spectral_floor,
                scale_eps=self.scale_eps,
                sample_contexts=self._reference_samples,
                term_scale_cache=self._term_scale_cache,
                term_ci_cache=self._term_ci_cache,
                rank_reference=self.rank_reference,
                rank_config=self.rank_config,
                evaluation_cache=self._evaluation_cache,
            )
            self._reference_samples = built_state.samples
            self._initial_state_cache[cache_key] = built_state
            self.state = clone_episode_initial_state(built_state)
        self.done = False
        self.finalized = False
        return self.observation()

    def observation(self) -> EnvObservation:
        """Return the current symbolic state."""

        state = self._require_state()
        theta_rej = tuple(term.canonical for term in state.registry.rej)
        return EnvObservation(
            theta_acc=tuple(state.theta_terms),
            theta_rej=theta_rej,
            h=state.h,
            k=state.k,
            K=self._episode_K,
            done=self.done,
            theta_size=len(state.theta_terms),
            max_theta_size=self.N,
        )

    def step(self, term: TermLike, *, library=None) -> EnvStep:
        """Try one generated term with candidate SVD filtering and reward."""

        state = self._require_state()
        if self.done:
            raise RuntimeError("Episode is already done. Call finalize() or reset().")
        if self.finalized:
            raise RuntimeError("Episode has already been finalized. Call reset().")

        check = try_add_term_with_svd(
            state,
            term,
            health_config=self.health_config,
            library=library,
            alpha=self.reward_alpha,
            lambda_len=self.reward_lambda_len,
            eta=self.reward_eta,
            k_max=self._episode_K,
            threshold_min=self.spectral_threshold_min,
            threshold_max=self.spectral_threshold_max,
            use_threshold_bounds=self.use_threshold_bounds,
            spectral_floor=self.spectral_floor,
        )
        self.done = state.k >= self._episode_K
        return EnvStep(
            observation=self.observation(),
            check=check,
            done=self.done,
        )

    def finalize(self) -> EnvFinal:
        """Return final SVD tensors and the step rewards accumulated in the episode."""

        state = self._require_state()
        if self.finalized:
            raise RuntimeError("Episode has already been finalized. Call reset().")

        if self.fast_finalize:
            result, final_S, final_Vh = compute_step_reward_result_fast(state=state)
        else:
            result, final_S, final_Vh = finalize_episode_with_svd(
                state,
                threshold_min=self.spectral_threshold_min,
                threshold_max=self.spectral_threshold_max,
                use_threshold_bounds=self.use_threshold_bounds,
                spectral_floor=self.spectral_floor,
            )
        self.done = True
        self.finalized = True
        return EnvFinal(result=result, final_S=final_S, final_Vh=final_Vh)

    def evaluate_library(self, theta_terms, *, feature_scales=None, reference_state=None) -> EnvFinal:
        """Evaluate a library or restrict fixed prolonged-column scales in pruning.

        ``feature_scales`` retains its legacy name; new vectors contain one
        entry per coefficient, with all terms inside each variable block.
        """
        if reference_state is None:
            if feature_scales is not None:
                raise ValueError("Supplied pruning scales require a reference_state.")
            self.reset(theta_terms)
            result, S, Vh = finalize_episode_with_svd(
                self.state, threshold_min=self.spectral_threshold_min,
                threshold_max=self.spectral_threshold_max, spectral_floor=self.spectral_floor,
                use_threshold_bounds=self.use_threshold_bounds,
            )
        else:
            if reference_state.task != self.task or reference_state.model is not self.model:
                raise ValueError("The pruning reference must belong to this task and surrogate.")
            if getattr(reference_state, "rank_reference", None) is not self.rank_reference:
                raise ValueError("Pruning must reuse the same fixed dimension-selection samples.")
            result, S, Vh = evaluate_library_subset(
                reference_state, theta_terms, scales=feature_scales,
                threshold_min=self.spectral_threshold_min,
                threshold_max=self.spectral_threshold_max, spectral_floor=self.spectral_floor,
                use_threshold_bounds=self.use_threshold_bounds,
            )
        return EnvFinal(result=result, final_S=S, final_Vh=Vh)

    @property
    def theta_acc(self) -> Tuple[str, ...]:
        return self.observation().theta_acc

    @property
    def theta_rej(self) -> Tuple[str, ...]:
        return self.observation().theta_rej

    def _require_state(self) -> EpisodeGramState:
        if self.state is None:
            raise RuntimeError("Environment has not been reset. Call reset(theta0) first.")
        return self.state
