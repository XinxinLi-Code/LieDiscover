"""Explicit RS-AC training entry point for supported symmetry tasks.

This script builds an explicit task configuration and calls
``train_rs_ac.main(argv)``.
PDE searches use the saved WSINDy equation directly by default. The equation
fit is shared across search seeds; no residual or surrogate network is loaded.

Run examples:

    python lie_discover.py burger --seed 0
    python lie_discover.py damped_burger --seed 0
    python lie_discover.py forced_transport --seed 0
    python lie_discover.py heat --seed 0
    python lie_discover.py kdv --seed 0
    python lie_discover.py rd --seed 0
    python lie_discover.py schrodinger --seed 0
    python lie_discover.py wave --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from train_rs_ac import main as train_rs_ac_main
from wsindy_source import DEFAULT_WSINDY_RUN_DIR, DEFAULT_WSINDY_SEED


WSINDY_SEARCH_DEFAULTS = {
    "jacobian_source": "wsindy",
    "wsindy_json": None,
    "wsindy_run_dir": DEFAULT_WSINDY_RUN_DIR,
    "wsindy_seed": DEFAULT_WSINDY_SEED,
    "rank_selection": "validated",
    "rank_top_k": 3,
    "rank_residual_tolerance": 1e-3,
    "rank_stability_tolerance": 1e-2,
    "rank_function_stability_tolerance": 1e-2,
    "rank_replicates": 3,
    "rank_scale_points": 1000,
    "rank_validation_points": 1000,
    "numerical_cache_mb": 128,
    "numerical_cache_max_entries": 256,
    "cache_log_every": 10,
}


BURGERS_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "burger",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 10,
    "k_max": 6,
    "min_steps": 3,
    "max_tree_len": 5,
    "theta0": "1,t,x,u",
    "variable_names": "t,x,u",
    "function_set": "1,mul,n2",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.05,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 5,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.001,
    "reward_lambda_len": 0.02,
    "reward_eta": 0.88,
    "reward_k_max": 6,
    "kappa_max": 1e5,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 20,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-4,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": None,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/burger",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}


DAMPED_BURGERS_RS_AC_CONFIG: Dict[str, Any] = dict(BURGERS_RS_AC_CONFIG)
DAMPED_BURGERS_RS_AC_CONFIG.update(
    {
        "task": "damped_burger",
        "kappa_max": 2e3,
        "theta0": "1,t,x,u",
        "batch_size": 50,
        "sparsify_max_iter": None,
        "function_set": "1,mul,exp,inv",
        "expected_D0": 2,
        "spectral_threshold_max": None,
        "max_reasonable_D": None,
        "final_vis_sample": None,
        "checkpoint_dir": "./checkpoints/rs_ac/damped_burger",
        "print_best_episode_details": True,
        "sparsify_max_iter": 10000,
        # "kappa_max": 2e4,
    }
)


FORCED_TRANSPORT_RS_AC_CONFIG: Dict[str, Any] = dict(BURGERS_RS_AC_CONFIG)
FORCED_TRANSPORT_RS_AC_CONFIG.update(
    {
        "task": "forced_transport",
        "num_iters": 50,
        "sparsify_max_iter": 100000,
        "batch_size": 50,
        "kappa_max": 5e3,
        "theta0": "1,t,x,u",
        "variable_names": "t,x,u",
        # "function_set": "1,mul,sub,sin,cos",
        "function_set": "1,mul,n2,sub,sin,cos",
        "special_unary_argument_tokens": "x,t,sub",
        "special_unary_context_only_tokens": "sub",
        "expected_D0": 2,
        "max_reasonable_D": None,
        # "min_cut_confidence": 0.8,
        "spectral_threshold_max": None,
        # "max_theta_size": 10,
        # "k_max": 6,
        "max_theta_size": 12,
        "k_max": 8,
        "reward_eta": 0.91,
        "min_steps": 5,
        "max_tree_len": 5,
        # "reward_k_max": 6,
        "reward_k_max": 8,
        "final_vis_sample": None,
        "checkpoint_dir": "./checkpoints/rs_ac/forced_transport",
        "print_best_episode_details": True,
        "reward_alpha": 0.2,
        "reward_lambda_len": 0.05,
        "max_tree_len": 6,
    }
)




WAVE_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "wave",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 100,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 20,
    "k_max": 15,
    "min_steps": 15,
    "max_tree_len": 5,
    "theta0": "1,t,x,y,u",
    "variable_names": "t,x,y,u",
    "function_set": "1,mul,n2,n3",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.1,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 11,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.1,
    "reward_lambda_len": 0.2,
    "reward_eta": 0.96,
    "reward_k_max": 15,
    "kappa_max": 1e7,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 200,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-3,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": 10000,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/wave",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}




HEAT_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "heat",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 10,
    "k_max": 6,
    "min_steps": 3,
    "max_tree_len": 5,
    "theta0": "1,t,x,u",
    "variable_names": "t,x,u",
    "function_set": "1,mul,n2",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.05,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 7,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.001,
    "reward_lambda_len": 0.02,
    "reward_eta": 0.88,
    "reward_k_max": 6,
    "kappa_max": 1e4,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 100,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-4,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": None,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/heat",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}



KDV_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "kdv",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 10,
    "k_max": 6,
    "min_steps": 3,
    "max_tree_len": 5,
    "theta0": "1,t,x,u",
    "variable_names": "t,x,u",
    "function_set": "1,mul,n2",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.05,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 4,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.001,
    "reward_lambda_len": 0.02,
    "reward_eta": 0.88,
    "reward_k_max": 6,
    "kappa_max": 1e4,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 20,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-5,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": None,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/kdv",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}



RD_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "rd",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 100,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 21,
    "k_max": 15,
    "min_steps": 15,
    "max_tree_len": 5,
    "theta0": "1,t,x,y,u,v",
    "variable_names": "t,x,y,u,v",
    "function_set": "1,mul,n2",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.1,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 5,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.1,
    "reward_lambda_len": 0.2,
    "reward_eta": 0.96,
    "reward_k_max": 15,
    "kappa_max": 1e3,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 100,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-4,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": 10000,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/rd",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}


SCHRODINGER_RS_AC_CONFIG: Dict[str, Any] = {
    # Equation.
    "task": "schrodinger",
    # Device, data window, and reproducibility.
    "gpu": 0,
    "seed": 0,
    "sample": 0,
    "fix_flag": True,
    "fix_seed": None,
    "sample_size": 100,
    # RS-AC outer training loop.
    "num_iters": 50,
    "batch_size": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    # Theta generation budget.
    "max_theta_size": 21,
    "k_max": 24,
    "min_steps": 15,
    "max_tree_len": 5,
    "theta0": "1,t,x,y,u,v",
    "variable_names": "t,x,y,u,v",
    "function_set": "1,mul,n2",
    "never_sample_tokens": "1",
    "protected": True,
    "temperature": 1.0,
    "greedy": False,
    "strict_prefix_ban": False,
    "allow_observed": False,
    "pad_to": None,
    # Encoder, decoder, and critic sizes.
    "rnn_hidden_size": 128,
    "decoder_embed_size": 32,
    "encoder_embed_size": 16,
    "tree_hidden_size": 32,
    "set_hidden_size": 64,
    # Risk-seeking actor-critic loss.
    "elite_fraction": 0.1,
    "min_elite_episodes": 1,
    "critic_coef": 1.0,
    "entropy_beta": 0.001,
    "entropy_mode": "sum",
    "normalize_advantages": True,
    "expected_D0": 6,
    "max_reasonable_D": None,
    "max_delta_D": None,
    # Step reward and health check.
    "reward_alpha": 0.1,
    "reward_lambda_len": 0.5,
    "reward_eta": 0.98,
    "reward_k_max": 24,
    "kappa_max": 1e4,
    "spectral_threshold_min": None,
    "spectral_threshold_max": None,
    "fast_finalize": True,
    "reward_verbose": False,
    # Final sparse visualization after all training iterations finish.
    "visualize_final": True,
    "final_vis_batch_size": 100,
    "final_vis_greedy": False,
    "final_vis_temperature": 1.0,
    "final_vis_sample": None,
    "final_vis_dir": "./vis",
    "sparsify_epsilon1": 1e-4,
    "sparsify_epsilon2": 1e-3,
    "sparsify_log_interval": 500,
    "sparsify_max_iter": 10000,
    "sparsify_verbose": False,
    # Training diagnostics after all iterations finish.
    "plot_training_curve": True,
    "training_curve_dir": None,
    # Output, logging, and checkpointing.
    "checkpoint_dir": "./checkpoints/rs_ac/schrodinger",
    "save_every": 10,
    "log_every": 1,
    "reward_step_log_every": 0,
    "max_reward_print_episodes": 0,
    "print_reward_steps": False,
    "print_best_episode_details": True,
    "best_detail_log_every": 1,
    "best_detail_full_svd": True,
    "best_detail_max_s": 120,
}




VALUE_FLAGS = (
    ("task", "--task"),
    ("data_dir", "--data-dir"),
    ("jacobian_source", "--jacobian-source"),
    ("wsindy_json", "--wsindy-json"),
    ("wsindy_run_dir", "--wsindy-run-dir"),
    ("wsindy_seed", "--wsindy-seed"),
    ("gpu", "--gpu"),
    ("seed", "--seed"),
    ("sample", "--sample"),
    ("fix_seed", "--fix-seed"),
    ("sample_size", "--sample-size"),
    ("rank_selection", "--rank-selection"),
    ("rank_top_k", "--rank-top-k"),
    ("rank_residual_tolerance", "--rank-residual-tolerance"),
    ("rank_stability_tolerance", "--rank-stability-tolerance"),
    ("rank_function_stability_tolerance", "--rank-function-stability-tolerance"),
    ("rank_replicates", "--rank-replicates"),
    ("rank_scale_points", "--rank-scale-points"),
    ("rank_validation_points", "--rank-validation-points"),
    ("numerical_cache_mb", "--numerical-cache-mb"),
    ("numerical_cache_max_entries", "--numerical-cache-max-entries"),
    ("cache_log_every", "--cache-log-every"),
    ("num_iters", "--num-iters"),
    ("batch_size", "--batch-size"),
    ("lr", "--lr"),
    ("weight_decay", "--weight-decay"),
    ("grad_clip", "--grad-clip"),
    ("max_theta_size", "--max-theta-size"),
    ("k_max", "--k-max"),
    ("min_steps", "--min-steps"),
    ("max_tree_len", "--max-tree-len"),
    ("theta0", "--theta0"),
    ("variable_names", "--variable-names"),
    ("function_set", "--function-set"),
    ("never_sample_tokens", "--never-sample-tokens"),
    ("special_unary_argument_tokens", "--special-unary-argument-tokens"),
    ("special_unary_context_only_tokens", "--special-unary-context-only-tokens"),
    ("temperature", "--temperature"),
    ("pad_to", "--pad-to"),
    ("rnn_hidden_size", "--rnn-hidden-size"),
    ("decoder_embed_size", "--decoder-embed-size"),
    ("encoder_embed_size", "--encoder-embed-size"),
    ("tree_hidden_size", "--tree-hidden-size"),
    ("set_hidden_size", "--set-hidden-size"),
    ("elite_fraction", "--elite-fraction"),
    ("min_elite_episodes", "--min-elite-episodes"),
    ("critic_coef", "--critic-coef"),
    ("entropy_beta", "--entropy-beta"),
    ("entropy_mode", "--entropy-mode"),
    ("expected_D0", "--expected-D0"),
    ("max_reasonable_D", "--max-reasonable-D"),
    ("max_delta_D", "--max-delta-D"),
    ("reward_alpha", "--reward-alpha"),
    ("reward_lambda_len", "--reward-lambda-len"),
    ("reward_eta", "--reward-eta"),
    ("reward_k_max", "--reward-k-max"),
    ("kappa_max", "--kappa-max"),
    ("min_cut_confidence", "--min-cut-confidence"),
    ("spectral_threshold_min", "--spectral-threshold-min"),
    ("spectral_threshold_max", "--spectral-threshold-max"),
    ("feature_scale_eps", "--feature-scale-eps"),
    ("spectral_floor", "--spectral-floor"),
    ("normalization_version", "--normalization-version"),
    ("final_vis_batch_size", "--final-vis-batch-size"),
    ("final_vis_temperature", "--final-vis-temperature"),
    ("final_vis_sample", "--final-vis-sample"),
    ("final_vis_dir", "--final-vis-dir"),
    ("sparsify_epsilon1", "--sparsify-epsilon1"),
    ("sparsify_epsilon2", "--sparsify-epsilon2"),
    ("sparsify_log_interval", "--sparsify-log-interval"),
    ("sparsify_max_iter", "--sparsify-max-iter"),
    ("training_curve_dir", "--training-curve-dir"),
    ("checkpoint_dir", "--checkpoint-dir"),
    ("save_every", "--save-every"),
    ("log_every", "--log-every"),
    ("reward_step_log_every", "--reward-step-log-every"),
    ("max_reward_print_episodes", "--max-reward-print-episodes"),
    ("best_detail_log_every", "--best-detail-log-every"),
    ("best_detail_max_s", "--best-detail-max-s"),
)

OPTIONAL_INT_FLAGS = {
    "fix_seed",
    "k_max",
    "expected_D0",
    "max_reasonable_D",
    "max_delta_D",
    "reward_k_max",
    "sparsify_max_iter",
}

OPTIONAL_NONE_FLAGS = OPTIONAL_INT_FLAGS | {
    "spectral_threshold_min",
    "spectral_threshold_max",
}


def _stringify(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _append_value_flag(argv: List[str], flag: str, key: str, value: Any) -> None:
    if value is None:
        if key in OPTIONAL_NONE_FLAGS:
            argv.extend([flag, "none"])
        return
    argv.extend([flag, _stringify(value)])


def _config_to_argv(config: Mapping[str, Any]) -> List[str]:
    """Convert an explicit task config dictionary into train_rs_ac CLI argv."""

    argv: List[str] = []
    for key, flag in VALUE_FLAGS:
        _append_value_flag(argv, flag, key, config.get(key))
    argv.append("--use-threshold-bounds" if config.get("use_threshold_bounds", False)
                else "--no-use-threshold-bounds")

    if config["fix_flag"]:
        argv.append("--fix-flag")
    if config["protected"]:
        argv.append("--protected")
    else:
        argv.append("--no-protected")
    if config["greedy"]:
        argv.append("--greedy")
    if config["strict_prefix_ban"]:
        argv.append("--strict-prefix-ban")
    if config["allow_observed"]:
        argv.append("--allow-observed")
    if not config["normalize_advantages"]:
        argv.append("--no-normalize-advantages")
    if config["fast_finalize"]:
        argv.append("--fast-finalize")
    else:
        argv.append("--no-fast-finalize")
    if config["reward_verbose"]:
        argv.append("--reward-verbose")
    if config["print_reward_steps"]:
        argv.append("--print-reward-steps")
    else:
        argv.append("--no-print-reward-steps")
    if config["print_best_episode_details"]:
        argv.append("--print-best-episode-details")
    if config["best_detail_full_svd"]:
        argv.append("--best-detail-full-svd")
    else:
        argv.append("--no-best-detail-full-svd")
    if config["visualize_final"]:
        argv.append("--visualize-final")
    else:
        argv.append("--no-visualize-final")
    if config["final_vis_greedy"]:
        argv.append("--final-vis-greedy")
    if config["sparsify_verbose"]:
        argv.append("--sparsify-verbose")
    if config["plot_training_curve"]:
        argv.append("--plot-training-curve")
    else:
        argv.append("--no-plot-training-curve")

    return argv


def _build_task_argv(
    base_config: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    config = {**WSINDY_SEARCH_DEFAULTS, **base_config}
    if overrides:
        retired = {"surrogate_checkpoint", "surrogate_epoch", "surrogate_seed", "residual_run_dir",
                   "hidden_dim", "n_layers", "activation", "classify", "save_dir"}.intersection(overrides)
        if retired:
            raise ValueError(f"Removed surrogate options: {', '.join(sorted(retired))}; use WSINDy source options.")
        config.update(overrides)
    if config["jacobian_source"] != "wsindy":
        raise ValueError("Only the WSINDy Jacobian source is supported; legacy surrogate loading was removed.")
    if not overrides or "checkpoint_dir" not in overrides:
        config["checkpoint_dir"] = f"{base_config['checkpoint_dir']}_{int(config['seed'])}"
    return _config_to_argv(config)


def build_burgers_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(BURGERS_RS_AC_CONFIG, overrides=overrides)


def build_damped_burgers_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(DAMPED_BURGERS_RS_AC_CONFIG, overrides=overrides)


def build_forced_transport_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(FORCED_TRANSPORT_RS_AC_CONFIG, overrides=overrides)


def build_wave_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(WAVE_RS_AC_CONFIG, overrides=overrides)




def build_heat_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(HEAT_RS_AC_CONFIG, overrides=overrides)


def build_kdv_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(KDV_RS_AC_CONFIG, overrides=overrides)


def build_rd_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(RD_RS_AC_CONFIG, overrides=overrides)


def build_schrodinger_argv(overrides: Optional[Mapping[str, Any]] = None) -> List[str]:
    return _build_task_argv(SCHRODINGER_RS_AC_CONFIG, overrides=overrides)


EQUATION_ALIASES = {
    "burger": "burger",
    "burgers": "burger",
    "damped_burger": "damped_burger",
    "damped-burger": "damped_burger",
    "damped_burgers": "damped_burger",
    "damped-burgers": "damped_burger",
    "forced_transport": "forced_transport",
    "forced-transport": "forced_transport",
    "transport": "forced_transport",
    "heat": "heat",
    "kdv": "kdv",
    "rd": "rd",
    "reaction-diffusion": "rd",
    "reaction_diffusion": "rd",
    "schrodinger": "schrodinger",
    "schroedinger": "schrodinger",
    "wave": "wave",
}

EQUATION_BUILDERS = {
    "burger": build_burgers_argv,
    "damped_burger": build_damped_burgers_argv,
    "forced_transport": build_forced_transport_argv,
    "heat": build_heat_argv,
    "kdv": build_kdv_argv,
    "rd": build_rd_argv,
    "schrodinger": build_schrodinger_argv,
    "wave": build_wave_argv,
}


def normalize_equation_name(equation: str) -> str:
    """Map user-facing equation names to config keys."""

    key = str(equation).strip().lower()
    if key not in EQUATION_ALIASES:
        choices = ", ".join(sorted(EQUATION_ALIASES))
        raise ValueError(f"Unknown equation {equation!r}. Choose one of: {choices}.")
    return EQUATION_ALIASES[key]


def build_equation_argv(
    equation: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Build ``train_rs_ac.py`` argv from a terminal equation name."""

    equation_key = normalize_equation_name(equation)
    if equation_key in EQUATION_BUILDERS:
        return EQUATION_BUILDERS[equation_key](overrides=overrides)
    raise AssertionError(f"Unhandled normalized equation: {equation_key}")


def _format_command(argv: Iterable[str]) -> str:
    return " ".join(["python", "train_rs_ac.py", *argv])


def run_equation(equation: str, overrides: Optional[Mapping[str, Any]] = None) -> None:
    equation_key = normalize_equation_name(equation)
    argv = build_equation_argv(equation_key, overrides=overrides)
    print(f"=== {equation_key} RS-AC example ===")
    print(_format_command(argv), flush=True)
    train_rs_ac_main(argv)


def burgers_main() -> None:
    run_equation("burger")


def damped_burgers_main() -> None:
    run_equation("damped_burger")


def forced_transport_main() -> None:
    run_equation("forced_transport")


def wave_main() -> None:
    run_equation("wave")




def heat_main() -> None:
    run_equation("heat")


def kdv_main() -> None:
    run_equation("kdv")


def rd_main() -> None:
    run_equation("rd")


def schrodinger_main() -> None:
    run_equation("schrodinger")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "equation",
        nargs="?",
        default="wave",
        choices=tuple(sorted(EQUATION_ALIASES)),
        help="Equation/task to run. Default keeps the current behavior: wave.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. The RS-AC visualization is saved to vis/<task>_<seed>.",
    )
    parser.add_argument("--checkpoint-dir", help="Override checkpoints/rs_ac/<task>_<seed>.")
    parser.add_argument("--gpu", type=int, help="Index among visible CUDA devices (default 0), or -1 for CPU.")
    parser.add_argument("--data-dir", help="Root of the PDE jet data.")
    parser.add_argument("--numerical-cache-mb", type=int,
                        help="Optional whole-library cache MiB limit; uses the configured default if omitted.")
    parser.add_argument("--numerical-cache-max-entries", type=int,
                        help="Optional whole-library cache entry limit.")
    parser.add_argument("--cache-log-every", type=int,
                        help="Cache statistics interval; 0 disables logging.")
    parser.add_argument("--rank-selection", choices=("validated", "gap"),
                        help="PDE default: validation and resampling rule; gap retains the legacy comparison.")
    parser.add_argument("--rank-function-stability-tolerance", type=float,
                        help="Generator-function subspace tolerance; uses the configured default if omitted.")
    parser.add_argument("--jacobian-source", choices=("wsindy",),
                        help="Use the saved explicit WSINDy equation.")
    equation_source = parser.add_mutually_exclusive_group()
    equation_source.add_argument("--wsindy-json", help="Saved WSINDy equation JSON.")
    equation_source.add_argument("--wsindy-run-dir", help="Equation root containing <task>_<wsindy-seed>/wsindy.json.")
    parser.add_argument("--wsindy-seed", type=int,
                        help="Shared WSINDy fit seed (default 0), independent of search --seed.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.gpu is not None:
        overrides["gpu"] = args.gpu
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.rank_selection is not None:
        overrides["rank_selection"] = args.rank_selection
    for key in ("numerical_cache_mb", "numerical_cache_max_entries", "cache_log_every",
                "rank_function_stability_tolerance"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    for key in ("jacobian_source", "wsindy_json", "wsindy_run_dir", "wsindy_seed"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    run_equation(args.equation, overrides=overrides or None)

if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"elapsed_seconds: {end_time - start_time:.6f}")
