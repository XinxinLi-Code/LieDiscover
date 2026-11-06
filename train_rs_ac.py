"""Train the encoder-decoder policy with risk-seeking actor-critic.

This is the first end-to-end training entry point for the RL symbolic search
loop:

    policy -> collect_rollouts -> SVD-filtered step rewards -> RS-AC loss

PDE Jacobians come from a saved WSINDy equation by default. Surrogate Jacobians
remain available as an explicit comparison. Only the policy is optimized.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset import PDEDataset
from decoder import DSORNNDecoder, make_dso_library
from encoder import DualSetEncoder
from policy_model import SymbolicPolicy
from rs_ac_loss import risk_seeking_actor_critic_loss
from sampling_utils import build_seeded_permutation_loader, describe_sample_window
from reward import (
    StepHealthConfig,
    estimate_spectral_cut,
    sparsify_basis_for_visualization,
    vis_dynamic_generators,
)
from feature_normalization import restore_coefficient_basis
from library_identities import prepare_generator_library
from library_evaluation_cache import LibraryEvaluationCache
from spectral import DEFAULT_SPECTRAL_FLOOR, RankSelectionConfig, RankUnresolvedError
from rank_sampling import build_rank_reference
from wsindy_source import DEFAULT_WSINDY_RUN_DIR, DEFAULT_WSINDY_SEED, load_wsindy_model, resolve_wsindy_json
from rl_env import LieThetaEnv
from rollout import collect_rollouts


TASK_DEFAULTS: Dict[str, Dict[str, object]] = {
    "burger": {
        "input_dim": 3,
        "output_dim": 1,
        "variable_names": ("t", "x", "u"),
        "theta0": ("1", "t", "x", "u"),
        "expected_D0": 5,
        "max_reasonable_D": 8,
    },
    "damped_burger": {
        "input_dim": 3,
        "output_dim": 1,
        "variable_names": ("t", "x", "u"),
        "theta0": ("1", "t", "x", "u"),
        "function_set": "1,mul,n2,exp,inv",
        "expected_D0": None,
        "max_reasonable_D": None,
    },
    "forced_transport": {
        "input_dim": 4,
        "output_dim": 1,
        "variable_names": ("t", "x", "u"),
        "theta0": ("1", "t", "x", "u"),
        "function_set": "1,mul,n2,sub,sin,cos",
        "expected_D0": 2,
        "max_reasonable_D": None,
        "special_unary_argument_tokens": "x,t,sub",
        "special_unary_context_only_tokens": "sub",
    },
    "heat": {
        "input_dim": 3,
        "output_dim": 1,
        "variable_names": ("t", "x", "u"),
        "theta0": ("1", "t", "x", "u"),
        "expected_D0": None,
        "max_reasonable_D": None,
    },
    "kdv": {
        "input_dim": 4,
        "output_dim": 1,
        "variable_names": ("t", "x", "u"),
        "theta0": ("1", "t", "x", "u"),
        "expected_D0": None,
        "max_reasonable_D": None,
    },
    "wave": {
        "input_dim": 6,
        "output_dim": 1,
        "variable_names": ("t", "x", "y", "u"),
        "theta0": ("1", "t", "x", "y", "u"),
        "expected_D0": None,
        "max_reasonable_D": None,
    },
    "rd": {
        "input_dim": 12,
        "output_dim": 2,
        "variable_names": ("t", "x", "y", "u", "v"),
        "theta0": ("1", "t", "x", "y", "u", "v"),
        "expected_D0": None,
        "max_reasonable_D": None,
    },
    "schrodinger": {
        "input_dim": 12,
        "output_dim": 2,
        "variable_names": ("t", "x", "y", "u", "v"),
        "theta0": ("1", "t", "x", "y", "u", "v"),
        "expected_D0": None,
        "max_reasonable_D": None,
    },
}


def _comma_tuple(text: Optional[str]) -> Optional[Tuple[str, ...]]:
    if text is None:
        return None
    if isinstance(text, (tuple, list)):
        values = tuple(str(part).strip() for part in text if str(part).strip())
        return values or None
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    return values or None


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return int(text)


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return float(text)


def _resolve_task_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.task not in TASK_DEFAULTS:
        raise ValueError(f"Unsupported task {args.task!r}.")

    defaults = TASK_DEFAULTS[args.task]
    args.input_dim = int(defaults["input_dim"])
    args.output_dim = int(defaults["output_dim"])
    args.variable_names = _comma_tuple(args.variable_names)
    if args.variable_names is None:
        args.variable_names = tuple(defaults["variable_names"])

    args.theta0 = _comma_tuple(args.theta0)
    if args.theta0 is None:
        args.theta0 = tuple(defaults["theta0"])

    if "function_set" in defaults and args.function_set == "1,mul,n2":
        args.function_set = str(defaults["function_set"])
    if args.special_unary_argument_tokens is None:
        args.special_unary_argument_tokens = defaults.get("special_unary_argument_tokens")
    if args.special_unary_context_only_tokens is None:
        args.special_unary_context_only_tokens = defaults.get("special_unary_context_only_tokens")

    if args.expected_D0 is None:
        args.expected_D0 = defaults["expected_D0"]
    if args.max_reasonable_D is None:
        args.max_reasonable_D = defaults["max_reasonable_D"]
    if getattr(args, "jacobian_source", None) is None:
        args.jacobian_source = "wsindy"
    if getattr(args, "rank_selection", None) is None:
        args.rank_selection = "validated"
    if getattr(args, "rank_function_stability_tolerance", None) is None:
        args.rank_function_stability_tolerance = 1e-2
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--task", default="burger", choices=tuple(TASK_DEFAULTS))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--jacobian-source", choices=("wsindy",), default=None,
                        help="Use the saved explicit WSINDy equation.")
    parser.add_argument("--wsindy-json", default=None, help="Saved explicit equation; never includes a residual network.")
    parser.add_argument("--wsindy-run-dir", default=DEFAULT_WSINDY_RUN_DIR)
    parser.add_argument("--wsindy-seed", type=int, default=DEFAULT_WSINDY_SEED,
                        help="WSINDy fitting seed, independent of the search seed.")
    parser.add_argument("--wsindy-sha256", default=None,
                        help="Expected equation hash recorded by a previous search.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="Index among visible CUDA devices, or -1 for CPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--fix-flag", action="store_true")
    parser.add_argument("--fix-seed", type=_optional_int, default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--rank-selection", choices=("validated", "gap"), default=None,
                        help="Default: Top3 candidates checked on scale/validation pools and repeated fits.")
    parser.add_argument("--rank-top-k", type=int, default=3)
    parser.add_argument("--rank-residual-tolerance", type=float, default=1e-3,
                        help="Shared relative residual tolerance for scale-reference and validation pools.")
    parser.add_argument("--rank-stability-tolerance", type=float, default=1e-2)
    parser.add_argument("--rank-function-stability-tolerance", type=float, default=1e-2,
                        help="Generator-function subspace stability tolerance on the shared scale pool.")
    parser.add_argument("--rank-replicates", type=int, choices=(3,), default=3)
    parser.add_argument("--rank-scale-points", type=int, default=1000)
    parser.add_argument("--rank-validation-points", type=int, default=1000)
    parser.add_argument("--numerical-cache-mb", type=_nonnegative_int, default=128,
                        help="Whole-library cache limit in MiB; 0 disables it. Default: 128.")
    parser.add_argument("--numerical-cache-max-entries", type=_nonnegative_int, default=256,
                        help="Whole-library cache entry limit; 0 disables it. Default: 256.")
    parser.add_argument("--cache-log-every", type=_nonnegative_int, default=10,
                        help="Log cache statistics every N iterations, plus first/last; 0 disables logging.")

    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-theta-size", type=int, default=7)
    parser.add_argument("--k-max", type=_optional_int, default=None,
                        help="Fixed search horizon K; every proposal counts, including rejections.")
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-tree-len", type=int, default=5)
    parser.add_argument("--theta0", default=None)
    parser.add_argument("--variable-names", default=None)
    parser.add_argument("--function-set", default="1,mul,n2")
    parser.add_argument("--never-sample-tokens", default="1")
    parser.add_argument("--special-unary-argument-tokens", default=None)
    parser.add_argument("--special-unary-context-only-tokens", default=None)
    parser.add_argument("--protected", action="store_true", default=True)
    parser.add_argument("--no-protected", action="store_false", dest="protected")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--strict-prefix-ban", action="store_true")
    parser.add_argument("--allow-observed", action="store_true")
    parser.add_argument("--pad-to", type=int, default=None)

    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--decoder-embed-size", type=int, default=32)
    parser.add_argument("--encoder-embed-size", type=int, default=16)
    parser.add_argument("--tree-hidden-size", type=int, default=32)
    parser.add_argument("--set-hidden-size", type=int, default=64)

    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--min-elite-episodes", type=int, default=1)
    parser.add_argument("--critic-coef", type=float, default=1.0)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--entropy-mode", choices=("sum", "mean"), default="sum")
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--expected-D0", type=_optional_int, default=None, help="Expected initial nullity for diagnostics only; the normalized spectrum determines D0.")
    parser.add_argument("--max-reasonable-D", type=_optional_int, default=None)
    parser.add_argument("--max-delta-D", type=_optional_int, default=None)

    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--reward-lambda-len", type=float, default=0.02)
    parser.add_argument("--reward-eta", type=float, default=0.98)
    parser.add_argument("--reward-k-max", type=_optional_int, default=None)
    parser.add_argument("--kappa-max", type=float, default=1e6)
    parser.add_argument("--min-cut-confidence", type=_optional_float, default=None)
    parser.add_argument("--spectral-threshold-min", type=_optional_float, default=None, help="Legacy lower bound; only used with --use-threshold-bounds.")
    parser.add_argument("--spectral-threshold-max", type=_optional_float, default=None, help="Legacy upper bound; only used with --use-threshold-bounds.")
    parser.add_argument("--use-threshold-bounds", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into legacy bounded SVD selection; default uses the full relative spectrum.")
    parser.add_argument("--feature-scale-eps", type=float, default=1e-12)
    parser.add_argument("--spectral-floor", type=float, default=DEFAULT_SPECTRAL_FLOOR)
    parser.add_argument("--normalization-version", type=int, choices=(2,), default=2,
                        help="Use fixed RMS scales for prolonged coefficient columns.")
    parser.add_argument("--fast-finalize", action="store_true", dest="fast_finalize", default=True)
    parser.add_argument("--no-fast-finalize", action="store_false", dest="fast_finalize")
    parser.add_argument("--reward-verbose", action="store_true")

    parser.add_argument("--visualize-final", action="store_true", dest="visualize_final", default=True)
    parser.add_argument("--no-visualize-final", action="store_false", dest="visualize_final")
    parser.add_argument("--final-vis-batch-size", type=int, default=20)
    parser.add_argument("--final-vis-greedy", action="store_true")
    parser.add_argument("--final-vis-temperature", type=float, default=1.0)
    parser.add_argument("--final-vis-sample", default=None)
    parser.add_argument("--final-vis-dir", default="./vis")
    parser.add_argument("--sparsify-epsilon1", type=float, default=1e-4)
    parser.add_argument("--sparsify-epsilon2", type=float, default=1e-4)
    parser.add_argument("--sparsify-log-interval", type=int, default=500)
    parser.add_argument("--sparsify-max-iter", type=_optional_int, default=500)
    parser.add_argument("--sparsify-verbose", action="store_true")
    parser.add_argument("--plot-training-curve", action="store_true", dest="plot_training_curve", default=True)
    parser.add_argument("--no-plot-training-curve", action="store_false", dest="plot_training_curve")
    parser.add_argument("--training-curve-dir", default=None)

    parser.add_argument("--checkpoint-dir", default="./checkpoints/rs_ac")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--reward-step-log-every", type=int, default=1)
    parser.add_argument("--max-reward-print-episodes", type=int, default=0)
    parser.add_argument("--print-reward-steps", action="store_true", dest="print_reward_steps")
    parser.add_argument("--no-print-reward-steps", action="store_false", dest="print_reward_steps")
    parser.set_defaults(print_reward_steps=False)
    parser.add_argument("--print-best-episode-details", action="store_true")
    parser.add_argument("--best-detail-log-every", type=int, default=1)
    parser.add_argument("--best-detail-full-svd", action="store_true", default=True)
    parser.add_argument("--no-best-detail-full-svd", action="store_false", dest="best_detail_full_svd")
    parser.add_argument("--best-detail-max-s", type=int, default=120)

    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_args(args: argparse.Namespace) -> torch.device:
    gpu = int(args.gpu)
    if gpu < -1:
        raise ValueError(f"Invalid --gpu {gpu}: use -1 for CPU or a nonnegative visible CUDA index.")
    if gpu == -1 or not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if gpu >= count:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise ValueError(
            f"Requested --gpu {gpu}, but this process sees {count} CUDA device(s) "
            f"(CUDA_VISIBLE_DEVICES={visible!r}). "
            f"Use --gpu 0 for the first visible GPU, an index smaller than {count}, "
            "or --gpu -1 for CPU. CUDA indices refer to the visible devices, "
            "not the physical GPU IDs."
        )
    return torch.device(f"cuda:{gpu}")


def load_search_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load the frozen explicit equation used by search and pruning."""
    source = getattr(args, "jacobian_source", None) or "wsindy"
    if source != "wsindy":
        raise ValueError("Only the WSINDy Jacobian source is supported; legacy surrogate loading was removed.")
    args.jacobian_source = source
    path = resolve_wsindy_json(
        args.task, wsindy_json=getattr(args, "wsindy_json", None),
        wsindy_run_dir=getattr(args, "wsindy_run_dir", None),
        wsindy_seed=getattr(args, "wsindy_seed", DEFAULT_WSINDY_SEED),
    )
    model = load_wsindy_model(path, task=args.task, device=device,
                              expected_sha256=getattr(args, "wsindy_sha256", None),
                              input_dim=args.input_dim, output_dim=args.output_dim)
    args.wsindy_json = model.jacobian_source_metadata["wsindy_json"]
    args.wsindy_sha256 = model.jacobian_source_metadata["wsindy_sha256"]
    print(f"Loading WSINDy equation: {args.wsindy_json} (SHA256 {args.wsindy_sha256}); no residual network", flush=True)
    return model






def build_train_dataset(args: argparse.Namespace):
    return PDEDataset(path=getattr(args, "data_dir", "./data"), pde_name=args.task, mode="train", dtype=torch.float64)


def build_policy(args: argparse.Namespace, device: torch.device) -> SymbolicPolicy:
    function_set = _comma_tuple(args.function_set)
    never_sample = _comma_tuple(args.never_sample_tokens) or ()
    library = make_dso_library(
        input_names=args.variable_names,
        function_set=function_set,
        protected=bool(args.protected),
    )
    decoder = DSORNNDecoder(
        library=library,
        embed_size=args.decoder_embed_size,
        hidden_size=args.rnn_hidden_size,
        max_length=args.max_tree_len,
        never_sample_tokens=never_sample,
        special_unary_argument_tokens=_comma_tuple(args.special_unary_argument_tokens),
        special_unary_context_only_tokens=_comma_tuple(args.special_unary_context_only_tokens),
    )
    encoder = DualSetEncoder(
        vocab_size=decoder.encoder_vocab_size,
        embed_size=args.encoder_embed_size,
        tree_hidden_size=args.tree_hidden_size,
        set_hidden_size=args.set_hidden_size,
        rnn_hidden_size=args.rnn_hidden_size,
    )
    return SymbolicPolicy(
        encoder=encoder,
        decoder=decoder,
        variable_names=args.variable_names,
        device=device,
    )


def build_env(args: argparse.Namespace, model: torch.nn.Module, train_loader: DataLoader, device: torch.device) -> LieThetaEnv:
    mode = getattr(args, "rank_selection", None) or "validated"
    args.rank_selection = mode
    rank_reference = None
    rank_config = None
    if mode == "validated":
        if bool(args.use_threshold_bounds):
            raise ValueError("Validated rank selection cannot use legacy threshold bounds; use --rank-selection gap for that comparison.")
        rank_config = RankSelectionConfig(
            top_k=getattr(args, "rank_top_k", 3),
            residual_tolerance=getattr(args, "rank_residual_tolerance", 1e-3),
            stability_tolerance=getattr(args, "rank_stability_tolerance", 1e-2),
            function_stability_tolerance=getattr(args, "rank_function_stability_tolerance", 1e-2),
            spectral_floor=args.spectral_floor,
        )
        rank_reference = build_rank_reference(args, model, train_loader, device)
        roles = rank_reference.provenance["roles"]
        print(
            f"rank_selection=validated, top_k={rank_config.top_k}, "
            f"residual_tolerance={rank_config.residual_tolerance:g}, "
            f"scale_residual_tolerance={rank_config.residual_tolerance:g}, "
            f"stability_tolerance={rank_config.stability_tolerance:g}, "
            f"function_stability_tolerance={rank_config.function_stability_tolerance:g}, "
            f"fit_seeds={[group['seed'] for group in roles['fitting']]}, "
            f"fit_points_each={args.sample_size}, "
            f"scale_points={len(rank_reference.scale_indices)}, "
            f"selection_validation_points={len(rank_reference.validation_indices)}, test_used=False",
            flush=True,
        )
    else:
        print(f"rank_selection=gap (legacy spectral-gap rule; task={args.task})", flush=True)
    return LieThetaEnv(
        model=model,
        train_loader=train_loader,
        task=args.task,
        device=device,
        sample=args.sample,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        N=args.max_theta_size,
        k_max=(args.k_max if args.k_max is not None
               else max(int(args.max_theta_size) - len(args.theta0), 1)),
        health_config=StepHealthConfig(
            kappa_max=args.kappa_max,
            min_cut_confidence=args.min_cut_confidence,
        ),
        reward_alpha=args.reward_alpha,
        reward_lambda_len=args.reward_lambda_len,
        reward_eta=args.reward_eta,
        reward_k_max=args.reward_k_max,
        spectral_threshold_min=args.spectral_threshold_min,
        spectral_threshold_max=args.spectral_threshold_max,
        use_threshold_bounds=args.use_threshold_bounds,
        initial_D0=args.expected_D0,
        scale_eps=args.feature_scale_eps,
        spectral_floor=args.spectral_floor,
        fix_flag=bool(args.fix_flag),
        fix_seed=args.seed if args.fix_seed is None else args.fix_seed,
        sample_size=args.sample_size,
        rank_reference=rank_reference,
        rank_config=rank_config,
        numerical_cache_max_bytes=getattr(args, "numerical_cache_mb", 128) * 1024**2,
        numerical_cache_max_entries=getattr(args, "numerical_cache_max_entries", 256),
        fast_finalize=bool(args.fast_finalize),
        reward_verbose=bool(args.reward_verbose),
    )


def print_numerical_cache_statistics(env, *, iteration):
    """Report reuse/eviction without GPU reads or changes to the search state."""
    cache = getattr(env, "_evaluation_cache", None)
    if not isinstance(cache, LibraryEvaluationCache):
        return
    stats = cache.statistics()
    print(
        f"library_cache: iter={iteration}, hits={stats['hits']}, misses={stats['misses']}, "
        f"hit_rate={stats['hit_rate']:.2%}, evictions={stats['evictions']}, "
        f"oversize_skips={stats['oversize_skips']}, "
        f"entries={stats['entries']}/{stats['max_entries']}, "
        f"memory_mib={stats['bytes'] / 1024**2:.2f}/{stats['max_bytes'] / 1024**2:.2f}, "
        f"peak_mib={stats['peak_bytes'] / 1024**2:.2f}",
        flush=True,
    )


def save_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    iteration: int,
    policy: SymbolicPolicy,
    optimizer: torch.optim.Optimizer,
    stats,
    search_samples=None,
    historical_best=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": int(iteration),
            "args": vars(args),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": asdict(stats),
            "search_samples": search_samples,
            "best_candidate": historical_best,
        },
        path,
    )


def summarize_invalid_reasons(stats) -> str:
    counter = Counter(record.reason for record in stats.records if not record.valid)
    if not counter:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in counter.most_common())


def best_record_summary(stats) -> str:
    valid = [record for record in stats.records if record.valid]
    if not valid:
        return "none"
    best = max(valid, key=lambda record: record.score)
    return (
        f"idx={best.index},score={best.score:.6g},"
        f"D0={best.D0},D_best={best.D_best},DeltaD={best.delta_D}"
    )


def _best_valid_episode_from_stats(episodes, stats):
    valid_records = [record for record in stats.records if record.valid]
    if valid_records:
        best_record = max(valid_records, key=lambda record: record.score)
        for episode in episodes:
            if int(getattr(episode, "index", -1)) == int(best_record.index):
                return episode, best_record
    if not episodes:
        return None, None
    best_episode = max(episodes, key=_episode_total_return)
    return best_episode, None


def _format_s_values(S: torch.Tensor, max_values: int) -> str:
    values = S.detach().cpu().flatten().tolist()
    if max_values is not None and int(max_values) > 0:
        values = values[: int(max_values)]
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def _replay_episode_with_full_svd(
    *,
    episode,
    env: LieThetaEnv,
    policy: SymbolicPolicy,
):
    old_fast_finalize = bool(getattr(env, "fast_finalize", False))
    env.fast_finalize = False
    try:
        env.reset(episode.theta0)
        for step in episode.steps:
            env.step(step.term_for_env, library=policy.decoder.library)
        return env.finalize()
    finally:
        env.fast_finalize = old_fast_finalize


def _step_rank_summary(check) -> str:
    rank = getattr(check, "rank_selection", {})
    if rank.get("method") == "validated_top_gap":
        summary = (f"validation_R={rank.get('residual')}, stability_S={rank.get('stability')}, "
                   f"function_S={rank.get('function_stability')}, "
                   f"scale_R={rank.get('scale_residual')}, "
                   f"rank_resolved={rank.get('resolved')}, ")
        if not rank.get("resolved"):
            candidates = "; ".join(
                f"d={item['dimension']}: scale_R={item.get('scale_residual')}, "
                f"validation_R={item.get('residual')}, stability_S={item.get('stability')}, "
                f"function_S={item.get('function_stability')}, "
                f"function_rank_resolved={item.get('function_rank_resolved')}, "
                f"function_ranks={item.get('function_ranks')}"
                for item in rank.get("candidates", [])
            )
            summary += f"rank_reason={rank.get('reason')}, rank_candidates=[{candidates}], "
        return summary
    return ""


def print_best_episode_details(
    *,
    iteration: int,
    episodes,
    stats,
    args: argparse.Namespace,
    env: LieThetaEnv,
    policy: SymbolicPolicy,
) -> None:
    """Print the current batch best sequence and optional full-SVD diagnostics."""

    if not bool(args.print_best_episode_details):
        return
    if int(args.best_detail_log_every) <= 0:
        return
    if iteration % int(args.best_detail_log_every) != 0 and iteration != 1:
        return

    best_episode, best_record = _best_valid_episode_from_stats(episodes, stats)
    if best_episode is None:
        return

    result = best_episode.final.result
    score = _episode_total_return(best_episode)
    if best_record is not None:
        score = float(best_record.score)
    print(
        f"=== best episode detail: iter={iteration}, episode={best_episode.index} ===",
        flush=True,
    )
    print(
        f"score={score:.6g}, D0={result.D0}, D_best={result.D_best}, "
        f"DeltaD={result.delta_D}, theta={','.join(result.theta_terms)}",
        flush=True,
    )
    for step, reward in zip(best_episode.steps, best_episode.rewards):
        check = step.env_step.check
        sampled = " -> ".join(str(name) for name in step.action.token_names)
        print(
            f"  step={step.index}, sampled={sampled}, term={step.canonical_term}, "
            f"accepted={step.accepted}, reason={check.reason}, Len={check.tree_length}, "
            f"D={check.D_prev}->{check.D_t}, D_raw={check.D_raw}, "
            f"threshold={check.threshold:.6g}, confidence={check.cut_confidence:.6g}, "
            f"kappa={check.kappa:.6g}, "
            f"sigma_max={check.sigma_max:.6g}, sigma_min_active={check.sigma_min_active:.6g}, "
            + _step_rank_summary(check) + f"reward={float(reward):.6g}",
            flush=True,
        )

    if not bool(args.best_detail_full_svd):
        return

    try:
        cached = best_episode.final
        rank = result.normalization_metadata.get("rank_selection", {})
        if (rank.get("method") == "validated_top_gap"
                and cached.final_S.numel() > 0 and cached.final_Vh.numel() > 0):
            # Validated fast finalization already retains the accepted spectrum
            # and basis. Logging must not replay the search to recover them.
            final = cached
        else:
            final = _replay_episode_with_full_svd(
                episode=best_episode, env=env, policy=policy,
            )
    except RankUnresolvedError as exc:
        print(f"  full_svd: rank_unresolved={exc}", flush=True)
        return
    final_result = final.result
    threshold = float(final_result.threshold_best)
    null_dim = int(final_result.D_final)
    rank = final_result.normalization_metadata.get("rank_selection", {})
    if rank.get("method") == "validated_top_gap":
        print(
            f"  full_svd: D0={final_result.D0}, D_best={final_result.D_best}, "
            f"DeltaD={final_result.delta_D}, selected_dimension={null_dim}, "
            f"rank_resolved={rank.get('resolved')}, validation_R={rank.get('residual')}, "
            f"stability_S={rank.get('stability')}, "
            f"function_S={rank.get('function_stability')}, "
            f"scale_R={rank.get('scale_residual')}, "
            f"display_threshold={threshold:.6g}, threshold_role=display_only",
            flush=True,
        )
        for candidate in rank.get("candidates", []):
            print(f"    rank_candidate: d={candidate['dimension']}, "
                  f"R={candidate.get('residual')}, S={candidate.get('stability')}, "
                  f"function_S={candidate.get('function_stability')}, "
                  f"function_rank_resolved={candidate.get('function_rank_resolved')}, "
                  f"function_ranks={candidate.get('function_ranks')}, "
                  f"scale_R={candidate.get('scale_residual')}, "
                  f"accepted={candidate.get('accepted')}", flush=True)
    else:
        cut = estimate_spectral_cut(
            final.final_S, threshold_min=args.spectral_threshold_min,
            threshold_max=args.spectral_threshold_max,
            use_threshold_bounds=args.use_threshold_bounds, spectral_floor=args.spectral_floor,
        )
        print(
            f"  full_svd: D0={final_result.D0}, D_best={final_result.D_best}, "
            f"DeltaD={final_result.delta_D}, threshold_best={threshold:.6g}, "
            f"null_dim_at_threshold={null_dim}, fresh_cut_threshold={cut.threshold:.6g}, "
            f"fresh_cut_null_dim={cut.null_dim}, rank_cut={cut.rank_cut}, "
            f"gap={cut.gap:.6g}, confidence={cut.confidence:.6g}", flush=True,
        )
    print(
        f"  full_svd_theta={','.join(final_result.theta_terms)}",
        flush=True,
    )
    print(
        f"  full_S={_format_s_values(final.final_S, args.best_detail_max_s)}",
        flush=True,
    )


def print_episode_reward_trace(
    *,
    iteration: int,
    episode,
) -> None:
    """Print one sequence immediately after its step rewards are ready."""

    episode_index = int(getattr(episode, "index", 0))
    result = episode.final.result
    reward_sum = float(sum(episode.rewards))
    print(
        f"=== reward sequence trace: iter={iteration}, episode={episode_index} ===",
        flush=True,
    )
    print(
        f"episode={episode_index}, reward_sum={reward_sum:.6g}, "
        f"D0={result.D0}, D_best={result.D_best}, DeltaD={result.delta_D}, "
        f"done={episode.done_reason}, "
        f"theta={','.join(result.theta_terms)}",
        flush=True,
    )

    action_rewards = tuple(result.action_rewards)
    for step, reward, action_reward in zip(episode.steps, episode.rewards, action_rewards):
        sampled = " -> ".join(str(name) for name in step.action.token_names)
        logp = float(step.log_prob.detach().cpu())
        check = step.env_step.check
        print(
            f"  step={step.index}, sampled={sampled}, term={step.canonical_term}, "
            f"accepted={step.accepted}, reason={check.reason}, "
            f"Len={check.tree_length}, D={check.D_prev}->{check.D_t}, "
            f"kappa={check.kappa:.6g}, threshold={check.threshold:.6g}, "
            + _step_rank_summary(check) + f"reward={float(reward):.6g}, "
            f"status={action_reward.status}, "
            f"breakthrough_bonus={action_reward.breakthrough_bonus:.6g}, "
            f"delay={action_reward.delay_steps}, "
            f"eta_discount={action_reward.eta_discount:.6g}, "
            f"logp={logp:.6g}",
            flush=True,
        )


def _episode_total_return(episode) -> float:
    return float(sum(episode.rewards))


def _episode_is_reasonable_for_visualization(episode, args: argparse.Namespace) -> bool:
    result = episode.final.result
    if not bool(getattr(result, "valid", True)) or not isfinite(_episode_total_return(episode)):
        return False
    rank = getattr(result, "normalization_metadata", {}).get("rank_selection", {})
    if result.D_final is None or (rank.get("method") == "validated_top_gap" and not rank.get("resolved", False)):
        return False
    if args.max_reasonable_D is not None and int(result.D_final) > int(args.max_reasonable_D):
        return False
    if args.max_delta_D is not None and max(0, int(result.D_final) - int(result.D0)) > int(args.max_delta_D):
        return False
    if len(episode.steps) < int(args.min_steps):
        return False
    return True


def _episode_selection_key(episode):
    result = episode.final.result
    return (
        max(0, int(result.D_final) - int(result.D0)),
        int(result.D_final),
        _episode_total_return(episode),
    )


def _plain_proposal(term):
    """Keep the original expression tree without retaining policy tensors."""
    if isinstance(term, str):
        return term
    if hasattr(term, "name") and hasattr(term, "children"):
        tokens = []

        def visit(node):
            tokens.append(str(node.name))
            for child in node.children:
                visit(child)

        visit(term)
        return tokens
    if torch.is_tensor(term):
        return [int(value) for value in term.detach().cpu().reshape(-1).tolist()]
    if hasattr(term, "reshape") and hasattr(term, "tolist"):
        term = term.reshape(-1).tolist()
    if isinstance(term, (list, tuple)):
        return [value if isinstance(value, str) else int(value) for value in term]
    raise TypeError(f"Cannot archive proposal of type {type(term)!r}.")


def update_historical_best(previous, *, episodes, stats, args, iteration):
    """Retain the best valid candidate under the final-output ranking rule."""
    valid_indices = {int(record.index) for record in stats.records if record.valid}
    candidates = [episode for episode in episodes
                  if int(episode.index) in valid_indices
                  and _episode_is_reasonable_for_visualization(episode, args)]
    if not candidates:
        return previous
    best = max(candidates, key=_episode_selection_key)
    key = _episode_selection_key(best)
    if previous is not None and key <= tuple(previous["selection_key"]):
        return previous
    result = best.final.result
    return {
        "version": 1,
        "iteration": int(iteration),
        "episode_index": int(best.index),
        "theta0": list(best.theta0),
        "proposals": [_plain_proposal(step.term_for_env) for step in best.steps],
        "theta_terms": list(result.theta_terms),
        "D0": int(result.D0),
        "D_best": int(result.D_best),
        "D_final": int(result.D_final),
        "total_return": _episode_total_return(best),
        "selection_key": list(key),
        "normalization_metadata": copy.deepcopy(result.normalization_metadata),
    }


def select_final_episode(episodes, *, historical_best, args, env, policy):
    """Recheck the archived trajectory and compare it with full-SVD rollouts."""
    candidates = list(episodes)
    historical_episode = None
    recheck = None
    if historical_best is not None:
        replay = SimpleNamespace(
            theta0=tuple(historical_best["theta0"]),
            steps=tuple(SimpleNamespace(term_for_env=proposal)
                        for proposal in historical_best["proposals"]),
        )
        try:
            final = _replay_episode_with_full_svd(episode=replay, env=env, policy=policy)
        except RankUnresolvedError as exc:
            recheck = {
                "iteration": int(historical_best["iteration"]),
                "episode_index": int(historical_best["episode_index"]),
                "D_before": int(historical_best["D_final"]), "D_after": None,
                "eligible": False, "reason": str(exc),
            }
            print(f"historical_best_recheck: eligible=False, rank_unresolved={exc}", flush=True)
        else:
            historical_episode = SimpleNamespace(
                index=int(historical_best["episode_index"]), theta0=replay.theta0,
                steps=replay.steps, final=final, rewards=tuple(final.result.rewards),
            )
            recheck = {
                "iteration": int(historical_best["iteration"]),
                "episode_index": historical_episode.index,
                "D_before": int(historical_best["D_final"]),
                "D_after": int(final.result.D_final),
                "return_before": float(historical_best["total_return"]),
                "return_after": _episode_total_return(historical_episode),
                "theta_before": list(historical_best["theta_terms"]),
                "theta_after": list(final.result.theta_terms),
                "eligible": _episode_is_reasonable_for_visualization(historical_episode, args),
            }
            print(
                f"historical_best_recheck: iteration={recheck['iteration']}, "
                f"episode={historical_episode.index}, D={recheck['D_before']}->{recheck['D_after']}, "
                f"total_return={recheck['return_before']:.6g}->{recheck['return_after']:.6g}, "
                f"eligible={recheck['eligible']}", flush=True,
            )
            # On exact ties, retain the earlier discovery and its provenance.
            candidates.insert(0, historical_episode)
    best = _select_episode_for_visualization(candidates, args)
    from_history = best is historical_episode
    metadata = {
        "source": "training_history" if from_history else "final_sampling",
        "iteration": int(historical_best["iteration"]) if from_history else None,
        "episode_index": int(best.index),
        "selection_rule": "delta_D, D_final, total_return (descending)",
        "candidate_count": len(candidates),
        "historical_recheck": recheck,
    }
    print(
        f"final_selection: source={metadata['source']}, iteration={metadata['iteration']}, "
        f"episode={metadata['episode_index']}, candidates={len(candidates)}", flush=True,
    )
    return best, metadata


def _final_output_sample(args: argparse.Namespace) -> str:
    return (
        args.final_vis_sample
        if args.final_vis_sample is not None
        else str(args.seed)
    )


def _training_curve_dir(args: argparse.Namespace) -> Path:
    if args.training_curve_dir is not None:
        return Path(args.training_curve_dir)
    return Path(args.final_vis_dir) / f"{args.task}_{_final_output_sample(args)}"


def plot_training_return_curve(
    history: Sequence[Dict[str, float]],
    *,
    args: argparse.Namespace,
) -> None:
    """Plot max total return ``R_tau`` over training iterations."""

    if not bool(args.plot_training_curve):
        return
    if not history:
        return

    import matplotlib.pyplot as plt

    out_dir = _training_curve_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    iterations = np.asarray([entry["iteration"] for entry in history], dtype=float)
    max_returns = np.asarray([entry["max_total_return"] for entry in history], dtype=float)
    mean_returns = np.asarray([entry["mean_total_return"] for entry in history], dtype=float)
    success_counts = np.asarray([entry["success_episodes"] for entry in history], dtype=float)

    running_best = np.maximum.accumulate(
        np.where(np.isfinite(max_returns), max_returns, np.nan)
    )

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(iterations, max_returns, marker="o", linewidth=1.8, label=r"batch max $R_{\tau}$")
    plt.plot(iterations, running_best, linestyle="--", linewidth=1.4, label=r"best so far")
    plt.plot(iterations, mean_returns, alpha=0.55, linewidth=1.2, label="batch mean return")
    plt.xlabel("Iteration")
    plt.ylabel(r"Return $R_{\tau}$")
    plt.title(r"Risk-seeking training curve: max $R_{\tau}$")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    curve_path = out_dir / "training_max_total_return.png"
    plt.savefig(curve_path, bbox_inches="tight", dpi=180)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(iterations, max_returns, marker="o", linewidth=1.8, color="tab:blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"max $R_{\tau}$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(iterations, success_counts, alpha=0.25, color="tab:green")
    ax2.set_ylabel("DeltaD > 0 episodes", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.tight_layout()
    joint_path = out_dir / "training_return_success.png"
    fig.savefig(joint_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    history_path = out_dir / "training_return_history.json"
    history_path.write_text(json.dumps(list(history), indent=2), encoding="utf-8")
    print(
        f"saved training return curve to {curve_path}",
        flush=True,
    )


def _select_episode_for_visualization(episodes, args: argparse.Namespace):
    candidates = [
        episode
        for episode in episodes
        if _episode_is_reasonable_for_visualization(episode, args)
    ]
    if not candidates:
        raise ValueError("No valid episodes are available for final visualization.")
    return max(candidates, key=_episode_selection_key)


def visualize_final_policy_episode(
    *,
    args: argparse.Namespace,
    policy: SymbolicPolicy,
    env: LieThetaEnv,
    device: torch.device,
    k_max: int,
    historical_best=None,
) -> None:
    """Compare history with final rollouts, then sparsify and visualize the best."""

    if not bool(args.visualize_final):
        return
    if args.final_vis_batch_size <= 0:
        raise ValueError("final_vis_batch_size must be positive.")

    policy.eval()
    print("=== final sparse generator visualization ===", flush=True)
    old_fast_finalize = bool(getattr(env, "fast_finalize", False))
    env.fast_finalize = False
    try:
        episodes = collect_rollouts(
            policy,
            env,
            batch_size=args.final_vis_batch_size,
            theta0=args.theta0,
            max_steps=k_max,
            max_len=args.max_tree_len,
            greedy=bool(args.final_vis_greedy),
            temperature=args.final_vis_temperature,
            strict_prefix_ban=bool(args.strict_prefix_ban),
            forbid_observed=not bool(args.allow_observed),
            pad_to=args.pad_to,
        )
    finally:
        env.fast_finalize = old_fast_finalize

    best, selection = select_final_episode(
        episodes, historical_best=historical_best, args=args, env=env, policy=policy,
    )
    result = best.final.result
    total_return = _episode_total_return(best)
    print(
        f"best_final_episode={best.index}, total_return={total_return:.6g}, "
        f"D0={result.D0}, D_best={result.D_best}, DeltaD={result.delta_D}, "
        f"theta={','.join(result.theta_terms)}",
        flush=True,
    )

    prepared = prepare_generator_library(
        env=env, result=result, final_S=best.final.final_S,
        final_Vh=getattr(best.final, "final_Vh", None), mandatory_terms=args.theta0,
    )
    result = prepared.result
    sparse_basis = sparsify_basis_for_visualization(
        result.null_basis,
        device=device,
        epsilon1=args.sparsify_epsilon1,
        epsilon2=args.sparsify_epsilon2,
        log_interval=args.sparsify_log_interval,
        max_iter=args.sparsify_max_iter,
        verbose=bool(args.sparsify_verbose),
    )
    vis_sample = _final_output_sample(args)
    vis_dynamic_generators(
        S=prepared.final_S,
        basis=restore_coefficient_basis(sparse_basis, result.feature_scales),
        feature_scales=result.feature_scales,
        threshold=result.threshold_best,
        normalization_metadata={**result.normalization_metadata, "run_seed": int(args.seed),
                                "candidate_selection": selection},
        task=args.task,
        sample=vis_sample,
        theta_terms=result.theta_terms,
        save_vis_path=args.final_vis_dir,
    )
    print(
        f"saved final sparse generators to {args.final_vis_dir}/{args.task}_{vis_sample}",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = _resolve_task_defaults(parser.parse_args(argv))
    device = device_from_args(args)
    set_seed(args.seed)

    train_dataset = build_train_dataset(args)
    data_seed = args.seed if args.fix_seed is None else args.fix_seed
    train_loader = build_seeded_permutation_loader(
        train_dataset,
        batch_size=1,
        seed=data_seed,
    )

    model = load_search_model(args, device)
    policy = build_policy(args, device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    env = build_env(args, model, train_loader, device)
    env.reset(args.theta0)
    if args.expected_D0 is not None and env.state.initial_D0 != args.expected_D0:
        print(
            f"initial_nullity: expected_D0={args.expected_D0}, "
            f"normalized_D0={env.state.initial_D0}; using normalized_D0",
            flush=True,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("=== risk-seeking actor-critic training ===")
    print(f"task={args.task}, device={device}, theta0={','.join(args.theta0)}")
    k_max = args.k_max
    if k_max is None:
        k_max = max(int(args.max_theta_size) - len(args.theta0), 1)
    print(
        f"batch_size={args.batch_size}, num_iters={args.num_iters}, "
        f"N={args.max_theta_size}, k_max={k_max}, min_steps={args.min_steps}, "
        f"special_unary_argument_tokens={args.special_unary_argument_tokens}, "
        f"special_unary_context_only_tokens={args.special_unary_context_only_tokens}, "
        f"elite_fraction={args.elite_fraction}, expected_D0={args.expected_D0}, "
        f"critic_coef={args.critic_coef}, entropy_beta={args.entropy_beta}, "
        f"entropy_mode={args.entropy_mode}, "
        f"max_reasonable_D={args.max_reasonable_D}, kappa_max={args.kappa_max}, "
        f"min_cut_confidence={args.min_cut_confidence}, "
        f"reward_eta={args.reward_eta}, reward_k_max={args.reward_k_max}, "
        f"spectral_threshold_min={args.spectral_threshold_min}, "
        f"spectral_threshold_max={args.spectral_threshold_max} (relative), "
        f"use_threshold_bounds={args.use_threshold_bounds}, "
        f"normalization=prolonged_column_rms (version {args.normalization_version}), "
        f"feature_scale_eps={args.feature_scale_eps}, spectral_floor={args.spectral_floor}, "
        f"fast_finalize={args.fast_finalize}, reward_verbose={args.reward_verbose}, "
        f"fix_flag={args.fix_flag}, fix_seed={args.seed if args.fix_seed is None else args.fix_seed}, "
        f"sample_size={args.sample_size}"
    )
    print(f"numerical_cache_mb={args.numerical_cache_mb}, "
          f"numerical_cache_max_entries={args.numerical_cache_max_entries}, "
          f"cache_log_every={args.cache_log_every}", flush=True)
    if args.rank_selection == "validated":
        rank = env.state.normalization_metadata.get("rank_selection", {})
        print(f"initial_rank: dimension={env.state.initial_D0}, "
              f"R={rank.get('residual')}, S={rank.get('stability')}, "
              f"function_S={rank.get('function_stability')}, "
              f"scale_R={rank.get('scale_residual')}, "
              f"resolved={rank.get('resolved')}, "
              f"train_used_union={len(env.state.normalization_metadata.get('search_samples', {}).get('indices', []))}, "
              "sampler=balanced_trajectory_rank_v1", flush=True)
    else:
        print(
            "sample_window=" + describe_sample_window(
                len(train_dataset), seed=data_seed, sample=args.sample, sample_size=args.sample_size,
            ), flush=True,
        )

    training_history: List[Dict[str, float]] = []
    historical_best = None
    for iteration in range(1, args.num_iters + 1):
        policy.train()
        should_print_reward_steps = (
            bool(args.print_reward_steps)
            and args.reward_step_log_every > 0
            and (iteration % args.reward_step_log_every == 0 or iteration == 1)
        )

        episode_callback = None
        if should_print_reward_steps:
            def episode_callback(episode):
                if (
                    args.max_reward_print_episodes <= 0
                    or int(getattr(episode, "index", 0)) < args.max_reward_print_episodes
                ):
                    print_episode_reward_trace(iteration=iteration, episode=episode)

        episodes = collect_rollouts(
            policy,
            env,
            batch_size=args.batch_size,
            theta0=args.theta0,
            max_steps=k_max,
            max_len=args.max_tree_len,
            greedy=bool(args.greedy),
            temperature=args.temperature,
            strict_prefix_ban=bool(args.strict_prefix_ban),
            forbid_observed=not bool(args.allow_observed),
            pad_to=args.pad_to,
            episode_callback=episode_callback,
        )

        loss, stats = risk_seeking_actor_critic_loss(
            episodes,
            elite_fraction=args.elite_fraction,
            min_elite_episodes=args.min_elite_episodes,
            critic_coef=args.critic_coef,
            entropy_beta=args.entropy_beta,
            entropy_mode=args.entropy_mode,
            normalize_advantages=not bool(args.no_normalize_advantages),
            expected_D0=None,
            max_reasonable_D=args.max_reasonable_D,
            max_delta_D=args.max_delta_D,
            min_steps=args.min_steps,
        )
        updated_best = update_historical_best(
            historical_best, episodes=episodes, stats=stats, args=args, iteration=iteration,
        )
        if updated_best is not historical_best:
            historical_best = updated_best
            candidate_path = checkpoint_dir / "best_candidate.json"
            temporary_path = candidate_path.with_suffix(".tmp.json")
            temporary_path.write_text(json.dumps(historical_best, indent=2), encoding="utf-8")
            temporary_path.replace(candidate_path)
            print(
                f"historical_best_update: iteration={iteration}, "
                f"episode={historical_best['episode_index']}, D_final={historical_best['D_final']}, "
                f"total_return={historical_best['total_return']:.6g}, saved={candidate_path}",
                flush=True,
            )
        training_history.append(
            {
                "iteration": float(iteration),
                "max_total_return": float(stats.max_total_return),
                "mean_total_return": float(stats.mean_total_return),
                "mean_elite_score": float(stats.mean_elite_score),
                "success_episodes": float(stats.success_episodes),
                "valid_episodes": float(stats.valid_episodes),
                "actor_terms": float(stats.actor_term_count),
                "critic_terms": float(stats.critic_term_count),
            }
        )
        print_best_episode_details(
            iteration=iteration,
            episodes=episodes,
            stats=stats,
            args=args,
            env=env,
            policy=policy,
        )

        optimizer.zero_grad(set_to_none=True)
        if stats.loss_term_count > 0:
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

        if iteration % args.log_every == 0 or iteration == 1:
            print(
                f"iter={iteration}, loss={float(loss.detach().cpu()):.6g}, "
                f"valid={stats.valid_episodes}/{stats.total_episodes}, "
                f"success={stats.success_episodes}, elite={stats.elite_episodes}, "
                f"terms={stats.loss_term_count}, "
                f"actor_terms={stats.actor_term_count}, "
                f"critic_terms={stats.critic_term_count}, "
                f"actor={stats.actor_loss:.6g}, critic={stats.critic_loss:.6g}, "
                f"entropy={stats.entropy_loss:.6g}, "
                f"elite_threshold={stats.elite_threshold:.6g}, "
                f"mean_total_return={stats.mean_total_return:.6g}, "
                f"max_total_return={stats.max_total_return:.6g}, "
                f"mean_valid={stats.mean_valid_score:.6g}, "
                f"mean_elite={stats.mean_elite_score:.6g}, "
                f"mean_value={stats.mean_value:.6g}, "
                f"mean_adv={stats.mean_advantage:.6g}, "
                f"mean_entropy={stats.mean_entropy:.6g}, "
                f"best={best_record_summary(stats)}, "
                f"invalid={summarize_invalid_reasons(stats)}"
            )

        if args.cache_log_every > 0 and (iteration == 1 or iteration == args.num_iters
                                         or iteration % args.cache_log_every == 0):
            print_numerical_cache_statistics(env, iteration=iteration)

        if args.save_every > 0 and iteration % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"policy_iter_{iteration:05d}.pt",
                args=args,
                iteration=iteration,
                policy=policy,
                optimizer=optimizer,
                stats=stats,
                search_samples=env.state.normalization_metadata.get("search_samples"),
                historical_best=historical_best,
            )

    save_checkpoint(
        path=checkpoint_dir / "policy_last.pt",
        args=args,
        iteration=args.num_iters,
        policy=policy,
        optimizer=optimizer,
        stats=stats,
        search_samples=env.state.normalization_metadata.get("search_samples"),
        historical_best=historical_best,
    )
    plot_training_return_curve(training_history, args=args)
    visualize_final_policy_episode(
        args=args,
        policy=policy,
        env=env,
        device=device,
        k_max=k_max,
        historical_best=historical_best,
    )


if __name__ == "__main__":
    main()
