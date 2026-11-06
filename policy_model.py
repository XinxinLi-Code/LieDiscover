"""Policy wrapper for encoder-conditioned symbolic tree sampling.

This module keeps the policy-facing API small:

    action = policy.sample_action(theta_acc, theta_rej, h=h_k, k=k, K=K)

Internally it performs:

    Theta_acc / Theta_rej / h_k / k
        -> encoder padded tensors
        -> encoder h_0
        -> decoder sampled symbolic tree + log_prob + entropy
        -> critic value V(S)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from theta_utils import (
    DEFAULT_VARIABLE_NAMES,
    TermLike,
    term_to_expression,
    theta_sets_to_encoder_inputs,
)


@dataclass
class PolicyAction:
    """One sampled symbolic action and the tensors used to produce it."""

    tree: object
    expression: Optional[str]
    token_names: Sequence[str]
    action_ids: torch.Tensor
    valid_action_ids: Tuple[int, ...]
    length: int
    log_prob: torch.Tensor
    state_value: torch.Tensor
    tree_entropy: torch.Tensor
    mean_tree_entropy: torch.Tensor
    h_0: torch.Tensor
    acc_tensor: torch.Tensor
    acc_mask: torch.Tensor
    rej_tensor: torch.Tensor
    rej_mask: torch.Tensor
    decode_result: object


class SymbolicPolicy(nn.Module):
    """Compose ``DualSetEncoder`` and ``DSORNNDecoder`` into one policy."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.variable_names = tuple(variable_names)
        hidden_size = int(getattr(decoder, "hidden_size"))
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        if device is None:
            device = self._infer_module_device()
        if device is not None:
            self.to(device)

    def _infer_module_device(self) -> Optional[torch.device]:
        """Infer the device from encoder/decoder parameters if they were moved first."""

        for module in (self.encoder, self.decoder):
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
        return None

    def encode_theta_sets(
        self,
        theta_acc: Sequence[Union[TermLike, object]],
        theta_rej: Sequence[Union[TermLike, object]],
        *,
        pad_to: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode theta sets into tensors accepted by ``DualSetEncoder``."""

        if device is None:
            device = self.device
        return theta_sets_to_encoder_inputs(
            theta_acc,
            theta_rej,
            self.decoder.token_map,
            library=self.decoder.library,
            variable_names=self.variable_names,
            pad_to=pad_to,
            device=device,
            add_batch_dim=True,
        )

    def sample_action(
        self,
        theta_acc: Sequence[Union[TermLike, object]],
        theta_rej: Sequence[Union[TermLike, object]] = (),
        *,
        h: int,
        k: int,
        K: int,
        max_len: Optional[int] = None,
        greedy: bool = False,
        temperature: float = 1.0,
        strict_prefix_ban: bool = False,
        forbid_observed: bool = True,
        pad_to: Optional[int] = None,
        **decode_kwargs,
    ) -> PolicyAction:
        """Sample one symbolic tree conditioned on the full search state.

        The returned ``log_prob`` keeps its computation graph and can be used by
        a policy-gradient loss.
        """

        acc_tensor, acc_mask, rej_tensor, rej_mask = self.encode_theta_sets(
            theta_acc,
            theta_rej,
            pad_to=pad_to,
        )
        h_0 = self.encoder(acc_tensor, acc_mask, rej_tensor, rej_mask, h=h, k=k, K=K)

        if forbid_observed and decode_kwargs.get("forbidden_terms") is None:
            forbidden_terms = (
                self.decoder.token_map.forbidden_from_encoder(acc_tensor, acc_mask)
                + self.decoder.token_map.forbidden_from_encoder(rej_tensor, rej_mask)
            )
            decode_kwargs["forbidden_terms"] = forbidden_terms
            decode_kwargs.setdefault("forbidden_terms_encoding", "dso")

        result = self.decoder(
            h_0,
            max_len=max_len,
            greedy=greedy,
            temperature=temperature,
            strict_prefix_ban=strict_prefix_ban,
            **decode_kwargs,
        )
        state_values = self.critic(h_0).squeeze(-1)

        row = 0
        length = int(result.lengths[row].detach().cpu().item())
        action_ids = result.actions[row]
        valid_action_ids = tuple(
            int(token_id)
            for token_id in action_ids[:length].detach().cpu().tolist()
        )
        tree = result.trees[row]
        expression = None
        if tree is not None:
            try:
                expression = term_to_expression(
                    tree,
                    library=self.decoder.library,
                    variable_names=self.variable_names,
                )
            except Exception:
                expression = str(tree)

        return PolicyAction(
            tree=tree,
            expression=expression,
            token_names=result.token_names[row],
            action_ids=action_ids,
            valid_action_ids=valid_action_ids,
            length=length,
            log_prob=result.log_probs[row],
            state_value=state_values[row],
            tree_entropy=result.entropies[row],
            mean_tree_entropy=result.mean_entropies[row],
            h_0=h_0,
            acc_tensor=acc_tensor,
            acc_mask=acc_mask,
            rej_tensor=rej_tensor,
            rej_mask=rej_mask,
            decode_result=result,
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


if __name__ == "__main__":
    from decoder import DSORNNDecoder, make_dso_library
    from encoder import DualSetEncoder

    torch.manual_seed(0)

    variable_names = ("t", "x", "u")
    rnn_hidden = 128

    library = make_dso_library(
        input_names=variable_names,
        function_set=("1", "mul", "n2", "n3"),
        protected=True,
    )
    decoder = DSORNNDecoder(
        library=library,
        embed_size=32,
        hidden_size=rnn_hidden,
        max_length=10,
        never_sample_tokens=("1",),
    )
    encoder = DualSetEncoder(
        vocab_size=decoder.encoder_vocab_size,
        embed_size=16,
        tree_hidden_size=32,
        set_hidden_size=64,
        rnn_hidden_size=rnn_hidden,
    )
    policy = SymbolicPolicy(
        encoder=encoder,
        decoder=decoder,
        variable_names=variable_names,
    )

    theta_acc = ("1", "t", "x", "u")
    theta_rej = ("1",)
    action = policy.sample_action(
        theta_acc,
        theta_rej,
        h=0,
        k=0,
        K=6,
        max_len=5,
        temperature=1.0,
    )

    print("=== SymbolicPolicy actor-critic example ===")
    print(f"library names: {library.names}")
    print(f"theta_acc: {theta_acc}")
    print(f"theta_rej: {theta_rej}")
    print(f"acc_tensor shape: {tuple(action.acc_tensor.shape)}")
    print(f"rej_tensor shape: {tuple(action.rej_tensor.shape)}")
    print(f"h_0 shape: {tuple(action.h_0.shape)}")
    print(f"sampled ids: {action.valid_action_ids}")
    print(f"sampled prefix: {' -> '.join(action.token_names)}")
    print(f"sampled tree: {action.tree}")
    print(f"sampled expression: {action.expression}")
    print(f"log_prob: {float(action.log_prob.detach().cpu()):.4f}")
    print(f"state_value: {float(action.state_value.detach().cpu()):.4f}")
    print(f"tree_entropy_sum: {float(action.tree_entropy.detach().cpu()):.4f}")
    print(f"tree_entropy_mean: {float(action.mean_tree_entropy.detach().cpu()):.4f}")
