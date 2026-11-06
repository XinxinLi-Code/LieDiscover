"""PyTorch decoder that samples DSO-style symbolic trees from encoder h_0.

The original DSO controller is TensorFlow based.
This module keeps the same autoregressive idea in PyTorch so the
``DualSetEncoder`` in ``encoder.py`` can pass its output directly as the
decoder initial hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


TokenSeq = Union[str, Sequence[Union[int, str]], np.ndarray, torch.Tensor]


DEFAULT_FUNCTION_SET = ("add", "sub", "mul", "div")
DEFAULT_INPUT_NAMES = ("t", "x", "u", "u_x")


@dataclass(frozen=True)
class SymbolicTree:
    """Recursive tree node rebuilt from a DSO prefix traversal."""

    token_id: int
    name: str
    children: Tuple["SymbolicTree", ...] = ()

    def prefix(self) -> List[str]:
        out = [self.name]
        for child in self.children:
            out.extend(child.prefix())
        return out

    def __str__(self) -> str:
        if not self.children:
            return self.name
        if len(self.children) == 1:
            child = str(self.children[0])
            if self.name == "neg":
                return f"(-{child})"
            if self.name == "n2":
                return f"({child})**2"
            if self.name == "n3":
                return f"({child})**3"
            if self.name == "n4":
                return f"({child})**4"
            return f"{self.name}({child})"
        if len(self.children) == 2:
            left, right = str(self.children[0]), str(self.children[1])
            infix = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "+": "+",
                "-": "-",
                "*": "*",
                "/": "/",
            }.get(self.name)
            if infix is not None:
                return f"({left} {infix} {right})"
        args = ", ".join(str(child) for child in self.children)
        return f"{self.name}({args})"


@dataclass
class DecodeResult:
    """Container returned by ``DSORNNDecoder.forward``."""

    actions: torch.Tensor
    lengths: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    mean_entropies: torch.Tensor
    token_names: List[List[str]]
    trees: List[Optional[SymbolicTree]]
    priors: Optional[torch.Tensor] = None


@dataclass
class _SubSlotContext:
    """Prefix-generation slot metadata for hard ``sub`` constraints."""

    inside_sub: bool = False
    sub_id: Optional[int] = None
    sub_child_index: Optional[int] = None
    first_terminal: Optional[int] = None


def _load_dso_module(module_name: str):
    """Load the bundled DSO utilities without a TensorFlow trainer."""
    import importlib
    return importlib.import_module(module_name)


def make_dso_library(
    input_names: Sequence[str] = DEFAULT_INPUT_NAMES,
    function_set: Sequence[Union[str, float]] = DEFAULT_FUNCTION_SET,
    protected: bool = False,
):
    """Build a ``dso.library.Library`` for the decoder.

    DSO normally names input variables ``x1``, ``x2``, ... . For PDE discovery it
    is more readable to rename those terminal tokens to values such as ``t``,
    ``x``, ``u`` and ``u_x`` while keeping their DSO input indices unchanged.
    """

    functions = _load_dso_module("dso.functions")
    library_mod = _load_dso_module("dso.library")

    tokens = functions.create_tokens(
        n_input_var=len(input_names),
        function_set=list(function_set),
        protected=protected,
    )
    for token, name in zip(tokens[: len(input_names)], input_names):
        token.name = str(name)
    for token in tokens[len(input_names):]:
        value = getattr(token, "value", None)
        if value is not None and getattr(token, "arity", None) == 0:
            value_float = float(np.atleast_1d(value)[0])
            token.name = str(int(value_float)) if value_float.is_integer() else str(value_float)
    return library_mod.Library(tokens)




def _split_token_string(text: str) -> List[str]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    sep = "," if "," in text else None
    return [part.strip() for part in text.split(sep) if part.strip()]


def _is_float_literal(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _library_name_to_id(library) -> dict:
    name_to_id = {name: idx for idx, name in enumerate(library.names)}
    for idx, name in enumerate(library.names):
        if isinstance(name, str) and _is_float_literal(name):
            value = float(name)
            name_to_id.setdefault(str(int(value)) if value.is_integer() else str(value), idx)
            name_to_id.setdefault(str(value), idx)
    return name_to_id


def _clean_token_sequence(seq: TokenSeq, library, pad_value: int) -> Tuple[int, ...]:
    """Normalize names/ids/tensors into a tuple of DSO action ids."""

    if isinstance(seq, str):
        items: Iterable[Union[int, str]] = _split_token_string(seq)
    elif torch.is_tensor(seq):
        items = seq.detach().cpu().view(-1).tolist()
    elif isinstance(seq, np.ndarray):
        items = seq.reshape(-1).tolist()
    else:
        items = seq

    name_to_id = _library_name_to_id(library)
    out: List[int] = []
    for item in items:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
            if item not in name_to_id:
                raise ValueError(f"Unknown token name {item!r}; library has {library.names}.")
            out.append(name_to_id[item])
        else:
            token_id = int(item)
            if token_id == pad_value:
                continue
            if token_id < 0 or token_id >= library.L:
                raise ValueError(f"Token id {token_id} is outside library size {library.L}.")
            out.append(token_id)
    return tuple(out)


def normalize_forbidden_terms(
    terms: Optional[Union[TokenSeq, Sequence[TokenSeq]]],
    library,
    pad_value: int = -1,
) -> Tuple[Tuple[int, ...], ...]:
    """Normalize accepted/rejected theta terms for duplicate masking."""

    if terms is None:
        return ()
    if isinstance(terms, str):
        return (_clean_token_sequence(terms, library, pad_value),)
    if torch.is_tensor(terms):
        if terms.ndim == 1:
            return (_clean_token_sequence(terms, library, pad_value),)
        return tuple(_clean_token_sequence(row, library, pad_value) for row in terms)
    if isinstance(terms, np.ndarray):
        if terms.ndim == 1:
            return (_clean_token_sequence(terms, library, pad_value),)
        return tuple(_clean_token_sequence(row, library, pad_value) for row in terms)
    if len(terms) == 0:
        return ()

    first = terms[0]  # type: ignore[index]
    if isinstance(first, (int, np.integer, str)):
        return (_clean_token_sequence(terms, library, pad_value),)  # type: ignore[arg-type]
    return tuple(_clean_token_sequence(term, library, pad_value) for term in terms)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EncoderDSOTokenMap:
    """Explicit mapping between encoder ids and DSO/decoder ids.

    Encoder ids:
        0 is PAD, 1..L are DSO token ids shifted by +1.

    DSO/decoder ids:
        0..L-1 are real DSO library token ids. Decoder output padding uses -1.
    """

    library: object
    encoder_pad_id: int = 0
    decoder_pad_id: int = -1

    @property
    def dso_vocab_size(self) -> int:
        return int(self.library.L)

    @property
    def encoder_vocab_size(self) -> int:
        return self.dso_vocab_size + 1

    def dso_id_to_encoder_id(self, token_id: int) -> int:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.dso_vocab_size:
            raise ValueError(f"DSO token id {token_id} is outside library size {self.dso_vocab_size}.")
        return token_id + 1

    def encoder_id_to_dso_id(self, token_id: int) -> int:
        token_id = int(token_id)
        if token_id == self.encoder_pad_id:
            return self.decoder_pad_id
        dso_id = token_id - 1
        if dso_id < 0 or dso_id >= self.dso_vocab_size:
            raise ValueError(
                f"Encoder token id {token_id} is invalid. Expected 0 for PAD "
                f"or 1..{self.dso_vocab_size} for shifted DSO ids."
            )
        return dso_id

    def dso_to_encoder(
        self,
        seq: TokenSeq,
        pad_to: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert one DSO prefix sequence to encoder ids."""

        dso_ids = _clean_token_sequence(seq, self.library, self.decoder_pad_id)
        encoder_ids = [self.dso_id_to_encoder_id(token_id) for token_id in dso_ids]
        if pad_to is not None:
            if len(encoder_ids) > pad_to:
                raise ValueError(f"Sequence length {len(encoder_ids)} exceeds pad_to={pad_to}.")
            encoder_ids = encoder_ids + [self.encoder_pad_id] * (pad_to - len(encoder_ids))
        return torch.as_tensor(encoder_ids, dtype=torch.long, device=device)

    def dso_batch_to_encoder(
        self,
        seqs: Sequence[TokenSeq],
        pad_to: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert a list of DSO prefix sequences to a padded encoder tensor."""

        if pad_to is None:
            pad_to = max(len(_clean_token_sequence(seq, self.library, self.decoder_pad_id)) for seq in seqs)
        rows = [self.dso_to_encoder(seq, pad_to=pad_to) for seq in seqs]
        return torch.stack(rows, dim=0).to(device=device)

    def encoder_to_dso(self, seq: TokenSeq, drop_pad: bool = True) -> Tuple[int, ...]:
        """Convert one encoder-id sequence back to DSO token ids."""

        if isinstance(seq, str):
            items: Iterable[Union[int, str]] = _split_token_string(seq)
        elif torch.is_tensor(seq):
            items = seq.detach().cpu().view(-1).tolist()
        elif isinstance(seq, np.ndarray):
            items = seq.reshape(-1).tolist()
        else:
            items = seq

        out: List[int] = []
        for item in items:
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    continue
                item = int(item)
            dso_id = self.encoder_id_to_dso_id(int(item))
            if dso_id == self.decoder_pad_id and drop_pad:
                continue
            out.append(dso_id)
        return tuple(out)

    def forbidden_from_encoder(
        self,
        terms: Optional[Union[TokenSeq, Sequence[TokenSeq]]],
        mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    ) -> Tuple[Tuple[int, ...], ...]:
        """Convert padded encoder-id theta tensors into DSO forbidden terms."""

        if terms is None:
            return ()

        if torch.is_tensor(terms):
            term_tensor = terms.detach().cpu()
            if term_tensor.ndim == 1:
                rows = term_tensor.view(1, -1)
            else:
                rows = term_tensor.reshape(-1, term_tensor.shape[-1])
        elif isinstance(terms, np.ndarray):
            term_array = np.asarray(terms)
            if term_array.ndim == 1:
                rows = term_array.reshape(1, -1)
            else:
                rows = term_array.reshape(-1, term_array.shape[-1])
        else:
            if len(terms) == 0:  # type: ignore[arg-type]
                return ()
            first = terms[0]  # type: ignore[index]
            if isinstance(first, (int, np.integer, str)):
                return (self.encoder_to_dso(terms),)  # type: ignore[arg-type]
            return tuple(self.encoder_to_dso(term) for term in terms)  # type: ignore[arg-type]

        if mask is None:
            keep = [True] * len(rows)
        elif torch.is_tensor(mask):
            keep = [bool(x) for x in mask.detach().cpu().reshape(-1).tolist()]
        else:
            keep = [bool(x) for x in np.asarray(mask).reshape(-1).tolist()]

        forbidden: List[Tuple[int, ...]] = []
        for row, should_keep in zip(rows, keep):
            if should_keep:
                seq = self.encoder_to_dso(row)
                if seq:
                    forbidden.append(seq)
        return tuple(forbidden)

    def dso_to_names(self, seq: Sequence[int]) -> List[str]:
        return [self.library[int(token_id)].name for token_id in seq]


def tokens_to_tree(tokens: Sequence[int], library) -> SymbolicTree:
    """Recursively rebuild a symbolic tree from DSO prefix token ids."""

    token_list = [int(t) for t in tokens]

    def parse(pos: int) -> Tuple[SymbolicTree, int]:
        if pos >= len(token_list):
            raise ValueError(f"Traversal ended before the tree closed: {token_list}")
        token_id = token_list[pos]
        token = library[token_id]
        children: List[SymbolicTree] = []
        next_pos = pos + 1
        for _ in range(int(token.arity)):
            child, next_pos = parse(next_pos)
            children.append(child)
        return SymbolicTree(token_id=token_id, name=token.name, children=tuple(children)), next_pos

    tree, final_pos = parse(0)
    if final_pos != len(token_list):
        raise ValueError(f"Traversal contains extra tokens after a closed tree: {token_list}")
    return tree


def _parents_siblings_python(actions: np.ndarray, library) -> Tuple[np.ndarray, np.ndarray]:
    """Small fallback matching DSO's parents_siblings for one-step decoding."""

    batch, length = actions.shape
    parents = np.full(batch, library.EMPTY_PARENT, dtype=np.int32)
    siblings = np.full(batch, library.EMPTY_SIBLING, dtype=np.int32)
    for row in range(batch):
        last = actions[row, -1]
        if library.arities[last] > 0:
            parents[row] = library.parent_adjust[last]
            continue
        dangling = 0
        for offset in range(length):
            idx = length - offset - 1
            token_id = actions[row, idx]
            dangling += int(library.arities[token_id]) - 1
            if dangling == 0:
                parents[row] = library.parent_adjust[token_id]
                siblings[row] = actions[row, idx + 1]
                break
    return parents, siblings


def _parents_siblings(actions: np.ndarray, library) -> Tuple[np.ndarray, np.ndarray]:
    try:
        subroutines = _load_dso_module("dso.subroutines")
        return subroutines.parents_siblings(
            actions.astype(np.int32),
            library.arities.astype(np.int32),
            library.parent_adjust.astype(np.int32),
            int(library.EMPTY_PARENT),
            int(library.EMPTY_SIBLING),
        )
    except Exception:
        return _parents_siblings_python(actions.astype(np.int32), library)


class DSORNNDecoder(nn.Module):
    """Autoregressive decoder for DSO prefix traversals.

    Parameters
    ----------
    library:
        A ``dso.library.Library`` instance. Its token ids are the decoder action
        ids.
    hidden_size:
        Must match the encoder's ``h_0`` size unless ``context_size`` is set.
    prior:
        Optional DSO prior object. It can be a ``JointPrior`` from
        a prior object exposing ``initial_prior`` and
        ``__call__(actions, parent, sibling, dangling, finished)``.
    """

    def __init__(
        self,
        library,
        embed_size: int = 32,
        hidden_size: int = 128,
        context_size: Optional[int] = None,
        cell_type: str = "lstm",
        prior=None,
        never_sample_tokens: Optional[Sequence[Union[int, str]]] = None,
        min_length: int = 1,
        max_length: int = 30,
        pad_value: int = -1,
        mask_value: float = -1e9,
        forbid_nested_trig: bool = True,
        nested_trig_tokens: Sequence[str] = ("sin", "cos", "exp"),
        special_unary_argument_tokens: Optional[Sequence[str]] = None,
        special_unary_context_only_tokens: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.library = library
        self.vocab_size = int(library.L)
        self.start_token_id = self.vocab_size
        self.hidden_size = int(hidden_size)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.pad_value = int(pad_value)
        self.mask_value = float(mask_value)
        self.prior = prior
        self.forbid_nested_trig = bool(forbid_nested_trig)
        self.token_map = EncoderDSOTokenMap(library, decoder_pad_id=self.pad_value)
        self.encoder_vocab_size = self.token_map.encoder_vocab_size

        never_sample_ids = (
            _clean_token_sequence(never_sample_tokens, library, self.pad_value)
            if never_sample_tokens is not None
            else ()
        )
        generation_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_id in never_sample_ids:
            generation_mask[int(token_id)] = False

        # Despite the historical name, this mask now covers special unary
        # functions whose arguments should not contain another such function.
        # This prevents structures such as sin(cos(x)), exp(exp(t)), exp(sin(x)).
        nested_trig_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        special_unary_argument_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        special_unary_context_only_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        library_names = tuple(str(name) for name in getattr(library, "names", ()))
        self.sub_token_id = library_names.index("sub") if "sub" in library_names else None
        self.sub_self_cancel_token_ids = tuple(
            token_id for token_id, name in enumerate(library_names) if name in {"t", "x"}
        )
        for token_id, name in enumerate(library_names):
            if name in {str(token) for token in nested_trig_tokens}:
                nested_trig_mask[int(token_id)] = True
        if special_unary_argument_tokens is not None:
            allowed = {str(token) for token in special_unary_argument_tokens}
            special_unary_argument_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for token_id, name in enumerate(library_names):
                if name in allowed:
                    special_unary_argument_mask[int(token_id)] = True
        if special_unary_context_only_tokens is not None:
            context_only = {str(token) for token in special_unary_context_only_tokens}
            for token_id, name in enumerate(library_names):
                if name in context_only:
                    special_unary_context_only_mask[int(token_id)] = True

        self.embedding = nn.Embedding(self.vocab_size + 1, embed_size)
        cell_type = cell_type.lower()
        if cell_type == "lstm":
            self.cell = nn.LSTMCell(embed_size, hidden_size)
            self.uses_lstm = True
        elif cell_type == "gru":
            self.cell = nn.GRUCell(embed_size, hidden_size)
            self.uses_lstm = False
        else:
            raise ValueError("cell_type must be 'lstm' or 'gru'.")
        self.output = nn.Linear(hidden_size, self.vocab_size)

        if context_size is not None and context_size != hidden_size:
            self.h0_proj = nn.Linear(context_size, hidden_size)
        else:
            self.h0_proj = None

        self.register_buffer(
            "arities",
            torch.as_tensor(library.arities, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("generation_mask", generation_mask, persistent=False)
        self.register_buffer("nested_trig_mask", nested_trig_mask, persistent=False)
        self.register_buffer(
            "special_unary_argument_mask",
            special_unary_argument_mask,
            persistent=False,
        )
        self.register_buffer(
            "special_unary_context_only_mask",
            special_unary_context_only_mask,
            persistent=False,
        )

    def _init_state(self, h_0: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(h_0, tuple):
            h, c = h_0
        else:
            h = h_0
            c = torch.zeros_like(h_0)
        if self.h0_proj is not None:
            h = self.h0_proj(h)
            c = torch.zeros_like(h)
        if h.size(-1) != self.hidden_size:
            raise ValueError(
                f"h_0 has hidden size {h.size(-1)}, but decoder hidden_size is "
                f"{self.hidden_size}. Pass context_size=... when constructing "
                "DSORNNDecoder if a projection is needed."
            )
        return h, c

    def _syntax_mask(
        self,
        dangling: torch.Tensor,
        step: int,
        max_len: int,
        finished: torch.Tensor,
    ) -> torch.Tensor:
        """Hard grammar mask for prefix-tree closure."""

        device = dangling.device
        arities = self.arities.to(device)
        remaining_after = max_len - (step + 1)
        next_dangling = dangling.unsqueeze(1) + arities.unsqueeze(0) - 1

        valid = next_dangling >= 0
        valid &= next_dangling <= remaining_after
        if step + 1 < self.min_length:
            valid &= next_dangling > 0
        valid &= self.generation_mask.to(device).unsqueeze(0)
        valid[finished] = False
        return valid

    def _prior_logits(
        self,
        actions: torch.Tensor,
        step: int,
        dangling: torch.Tensor,
        finished: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.prior is None:
            return None

        batch_size = actions.size(0)
        device = actions.device
        if step == 0:
            prior_np = np.asarray(self.prior.initial_prior(), dtype=np.float32)
            prior_np = np.broadcast_to(prior_np, (batch_size, self.vocab_size)).copy()
        else:
            hist = actions[:, :step].detach().cpu().numpy().astype(np.int32)
            parent, sibling = _parents_siblings(hist, self.library)
            prior_np = self.prior(
                hist,
                parent,
                sibling,
                dangling.detach().cpu().numpy().astype(np.int32),
                finished.detach().cpu().numpy().astype(bool),
            ).astype(np.float32)
        return torch.as_tensor(prior_np, dtype=torch.float32, device=device)

    def _forbidden_mask(
        self,
        actions: torch.Tensor,
        step: int,
        dangling: torch.Tensor,
        forbidden: Tuple[Tuple[int, ...], ...],
        strict_prefix_ban: bool,
    ) -> torch.Tensor:
        device = actions.device
        mask = torch.ones(actions.size(0), self.vocab_size, dtype=torch.bool, device=device)
        if not forbidden:
            return mask

        prefixes = actions[:, :step].detach().cpu().tolist()
        arities = self.arities.detach().cpu().numpy()
        dangling_cpu = dangling.detach().cpu().numpy()

        forbidden_set = set(forbidden)
        for row, prefix in enumerate(prefixes):
            prefix_tuple = tuple(int(tok) for tok in prefix)
            for token_id in range(self.vocab_size):
                candidate = prefix_tuple + (token_id,)
                next_dangling = int(dangling_cpu[row]) + int(arities[token_id]) - 1
                if next_dangling == 0 and candidate in forbidden_set:
                    mask[row, token_id] = False
                elif strict_prefix_ban and any(term[: len(candidate)] == candidate for term in forbidden):
                    mask[row, token_id] = False
        return mask

    def _nested_trig_context_mask(
        self,
        context_stacks: Sequence[List[bool]],
        *,
        device,
        batch_size: int,
        finished: torch.Tensor,
        restrict_special_arguments: bool = True,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, self.vocab_size, dtype=torch.bool, device=device)
        if not self.forbid_nested_trig:
            return mask

        trig_mask = self.nested_trig_mask.to(device)
        if not bool(trig_mask.any()):
            return mask

        finished_cpu = finished.detach().cpu().tolist()
        for row in range(batch_size):
            if finished_cpu[row]:
                continue
            if context_stacks[row] and bool(context_stacks[row][-1]):
                mask[row, trig_mask] = False
                argument_mask = self.special_unary_argument_mask.to(device)
                if restrict_special_arguments and not bool(argument_mask.all()):
                    mask[row] &= argument_mask
        return mask

    def _special_unary_context_only_mask(
        self,
        context_stacks: Sequence[List[bool]],
        *,
        device,
        batch_size: int,
        finished: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, self.vocab_size, dtype=torch.bool, device=device)
        context_only_mask = self.special_unary_context_only_mask.to(device)
        if not bool(context_only_mask.any()):
            return mask

        finished_cpu = finished.detach().cpu().tolist()
        for row in range(batch_size):
            if finished_cpu[row]:
                continue
            inside_special_unary = bool(context_stacks[row][-1]) if context_stacks[row] else False
            if not inside_special_unary:
                mask[row, context_only_mask] = False
        return mask

    def _sub_structure_mask(
        self,
        sub_context_stacks: Sequence[List[_SubSlotContext]],
        *,
        device,
        batch_size: int,
        finished: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, self.vocab_size, dtype=torch.bool, device=device)
        if self.sub_token_id is None:
            return mask

        finished_cpu = finished.detach().cpu().tolist()
        for row in range(batch_size):
            if finished_cpu[row]:
                continue
            context = sub_context_stacks[row][-1] if sub_context_stacks[row] else _SubSlotContext()
            if context.inside_sub:
                mask[row, int(self.sub_token_id)] = False
            if context.sub_child_index == 1 and context.first_terminal is not None:
                mask[row, int(context.first_terminal)] = False
        return mask

    def _update_nested_trig_context(
        self,
        context_stacks: List[List[bool]],
        *,
        next_token: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        if not self.forbid_nested_trig:
            return

        next_cpu = next_token.detach().cpu().tolist()
        active_cpu = active.detach().cpu().tolist()
        arities_cpu = self.arities.detach().cpu().tolist()
        trig_cpu = self.nested_trig_mask.detach().cpu().tolist()

        for row, is_active in enumerate(active_cpu):
            if not is_active:
                continue
            current_inside_trig = context_stacks[row].pop() if context_stacks[row] else False
            token_id = int(next_cpu[row])
            child_inside_trig = bool(current_inside_trig or trig_cpu[token_id])
            for _ in range(int(arities_cpu[token_id])):
                context_stacks[row].append(child_inside_trig)

    def _update_sub_context(
        self,
        sub_context_stacks: List[List[_SubSlotContext]],
        *,
        next_token: torch.Tensor,
        active: torch.Tensor,
        next_sub_ids: List[int],
    ) -> None:
        if self.sub_token_id is None:
            return

        next_cpu = next_token.detach().cpu().tolist()
        active_cpu = active.detach().cpu().tolist()
        arities_cpu = self.arities.detach().cpu().tolist()
        self_cancel_terminals = {int(token_id) for token_id in self.sub_self_cancel_token_ids}

        for row, is_active in enumerate(active_cpu):
            if not is_active:
                continue
            current = sub_context_stacks[row].pop() if sub_context_stacks[row] else _SubSlotContext()
            token_id = int(next_cpu[row])
            arity = int(arities_cpu[token_id])

            if (
                current.sub_child_index == 0
                and current.sub_id is not None
                and arity == 0
                and token_id in self_cancel_terminals
            ):
                for slot_index in range(len(sub_context_stacks[row]) - 1, -1, -1):
                    slot = sub_context_stacks[row][slot_index]
                    if slot.sub_id == current.sub_id and slot.sub_child_index == 1:
                        sub_context_stacks[row][slot_index] = _SubSlotContext(
                            inside_sub=slot.inside_sub,
                            sub_id=slot.sub_id,
                            sub_child_index=slot.sub_child_index,
                            first_terminal=token_id,
                        )
                        break

            if arity <= 0:
                continue

            if token_id == int(self.sub_token_id) and arity == 2:
                sub_id = int(next_sub_ids[row])
                next_sub_ids[row] += 1
                sub_context_stacks[row].append(
                    _SubSlotContext(inside_sub=True, sub_id=sub_id, sub_child_index=1)
                )
                sub_context_stacks[row].append(
                    _SubSlotContext(inside_sub=True, sub_id=sub_id, sub_child_index=0)
                )
            else:
                child_inside_sub = bool(current.inside_sub)
                for _ in range(arity):
                    sub_context_stacks[row].append(_SubSlotContext(inside_sub=child_inside_sub))

    def forward(
        self,
        h_0: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        max_len: Optional[int] = None,
        forbidden_terms: Optional[Union[TokenSeq, Sequence[TokenSeq]]] = None,
        forbidden_terms_mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        forbidden_terms_encoding: str = "dso",
        greedy: bool = False,
        temperature: float = 1.0,
        strict_prefix_ban: bool = False,
        return_priors: bool = False,
    ) -> DecodeResult:
        """Sample a batch of new symbolic trees.

        ``forbidden_terms`` should contain previously accepted/rejected DSO
        prefix traversals. Exact completed duplicates are masked out by default.
        Set ``strict_prefix_ban=True`` only when sharing a prefix with an old
        term should also be forbidden.
        """

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        max_len = int(max_len or self.max_length)
        h, c = self._init_state(h_0)
        batch_size = h.size(0)
        device = h.device

        if forbidden_terms_encoding == "dso":
            forbidden = normalize_forbidden_terms(forbidden_terms, self.library, self.pad_value)
        elif forbidden_terms_encoding == "encoder":
            forbidden = self.token_map.forbidden_from_encoder(forbidden_terms, forbidden_terms_mask)
        else:
            raise ValueError("forbidden_terms_encoding must be 'dso' or 'encoder'.")

        actions = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_value,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, dtype=h.dtype, device=device)
        entropies = torch.zeros(batch_size, dtype=h.dtype, device=device)
        dangling = torch.ones(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_token = torch.full(
            (batch_size,),
            fill_value=self.start_token_id,
            dtype=torch.long,
            device=device,
        )
        prior_steps: List[torch.Tensor] = []
        nested_trig_context_stacks: List[List[bool]] = [[False] for _ in range(batch_size)]
        sub_context_stacks: List[List[_SubSlotContext]] = [
            [_SubSlotContext()] for _ in range(batch_size)
        ]
        next_sub_ids = [0 for _ in range(batch_size)]

        for step in range(max_len):
            if finished.all():
                break

            x = self.embedding(prev_token)
            if self.uses_lstm:
                h, c = self.cell(x, (h, c))
            else:
                h = self.cell(x, h)
            logits = self.output(h) / temperature

            syntax_mask = self._syntax_mask(dangling, step, max_len, finished)
            forbidden_mask = self._forbidden_mask(
                actions,
                step,
                dangling,
                forbidden,
                strict_prefix_ban=strict_prefix_ban,
            )
            nested_mask = self._nested_trig_context_mask(
                nested_trig_context_stacks,
                device=device,
                batch_size=batch_size,
                finished=finished,
            )
            context_only_mask = self._special_unary_context_only_mask(
                nested_trig_context_stacks,
                device=device,
                batch_size=batch_size,
                finished=finished,
            )
            sub_structure_mask = self._sub_structure_mask(
                sub_context_stacks,
                device=device,
                batch_size=batch_size,
                finished=finished,
            )
            valid_mask = syntax_mask & forbidden_mask & nested_mask & context_only_mask & sub_structure_mask

            relaxed_nested_mask = None
            invalid_rows = valid_mask.sum(dim=1) == 0
            if torch.any(invalid_rows & ~finished):
                relaxed_nested_mask = self._nested_trig_context_mask(
                    nested_trig_context_stacks,
                    device=device,
                    batch_size=batch_size,
                    finished=finished,
                    restrict_special_arguments=False,
                )
                relaxed_mask = (
                    syntax_mask
                    & forbidden_mask
                    & relaxed_nested_mask
                    & context_only_mask
                    & sub_structure_mask
                )
                can_relax = (relaxed_mask.sum(dim=1) > 0) & invalid_rows & ~finished
                if torch.any(can_relax):
                    valid_mask[can_relax] = relaxed_mask[can_relax]

            invalid_rows = valid_mask.sum(dim=1) == 0
            if torch.any(invalid_rows & ~finished):
                # If every syntactically valid completion is forbidden as an
                # observed duplicate, let the environment reject the duplicate
                # instead of crashing the whole rollout.
                no_forbidden_mask = syntax_mask & nested_mask & context_only_mask & sub_structure_mask
                can_relax = (no_forbidden_mask.sum(dim=1) > 0) & invalid_rows & ~finished
                if torch.any(can_relax):
                    valid_mask[can_relax] = no_forbidden_mask[can_relax]

            invalid_rows = valid_mask.sum(dim=1) == 0
            if torch.any(invalid_rows & ~finished):
                if relaxed_nested_mask is None:
                    relaxed_nested_mask = self._nested_trig_context_mask(
                        nested_trig_context_stacks,
                        device=device,
                        batch_size=batch_size,
                        finished=finished,
                        restrict_special_arguments=False,
                    )
                relaxed_no_forbidden_mask = (
                    syntax_mask & relaxed_nested_mask & context_only_mask & sub_structure_mask
                )
                can_relax = (relaxed_no_forbidden_mask.sum(dim=1) > 0) & invalid_rows & ~finished
                if torch.any(can_relax):
                    valid_mask[can_relax] = relaxed_no_forbidden_mask[can_relax]

            prior_logits = self._prior_logits(actions, step, dangling, finished)
            if prior_logits is not None:
                logits = logits + prior_logits.to(dtype=logits.dtype)
                prior_steps.append(prior_logits)

            logits = logits.masked_fill(~valid_mask, self.mask_value)
            invalid_rows = torch.isneginf(logits).all(dim=1) | (logits <= self.mask_value / 2).all(dim=1)
            if torch.any(invalid_rows & ~finished):
                bad = torch.nonzero(invalid_rows & ~finished, as_tuple=False).view(-1).tolist()
                raise RuntimeError(
                    f"All tokens were masked for batch rows {bad} at step {step}. "
                    "Try a larger max_len or relax the prior/forbidden prefix constraints."
                )

            dist = torch.distributions.Categorical(logits=logits)
            step_entropy = dist.entropy()
            if greedy:
                next_token = torch.argmax(logits, dim=-1)
                step_log_prob = dist.log_prob(next_token)
            else:
                next_token = dist.sample()
                step_log_prob = dist.log_prob(next_token)

            next_token = torch.where(finished, prev_token.clamp(max=self.vocab_size - 1), next_token)
            active = ~finished
            actions[active, step] = next_token[active]
            log_probs[active] = log_probs[active] + step_log_prob[active]
            entropies[active] = entropies[active] + step_entropy[active]
            lengths[active] = step + 1

            token_arity = self.arities.to(device)[next_token]
            self._update_nested_trig_context(
                nested_trig_context_stacks,
                next_token=next_token,
                active=active,
            )
            self._update_sub_context(
                sub_context_stacks,
                next_token=next_token,
                active=active,
                next_sub_ids=next_sub_ids,
            )
            dangling = torch.where(active, dangling + token_arity - 1, dangling)
            finished = finished | (dangling == 0)
            prev_token = next_token

        token_names: List[List[str]] = []
        trees: List[Optional[SymbolicTree]] = []
        for row in range(batch_size):
            length = int(lengths[row].item())
            seq = actions[row, :length].detach().cpu().tolist()
            names = [self.library[int(token_id)].name for token_id in seq]
            token_names.append(names)
            try:
                trees.append(tokens_to_tree(seq, self.library) if length > 0 else None)
            except ValueError:
                trees.append(None)

        stacked_priors = None
        if return_priors and prior_steps:
            stacked_priors = torch.stack(prior_steps, dim=1)
        mean_entropies = entropies / lengths.to(dtype=entropies.dtype).clamp(min=1.0)

        return DecodeResult(
            actions=actions,
            lengths=lengths,
            log_probs=log_probs,
            entropies=entropies,
            mean_entropies=mean_entropies,
            token_names=token_names,
            trees=trees,
            priors=stacked_priors,
        )

    def sample_trees(self, h_0, **kwargs) -> List[Optional[SymbolicTree]]:
        return self.forward(h_0, **kwargs).trees


def generate_from_encoder(
    encoder: nn.Module,
    decoder: DSORNNDecoder,
    theta_acc: torch.Tensor,
    acc_mask: torch.Tensor,
    theta_rej: torch.Tensor,
    rej_mask: torch.Tensor,
    *,
    h,
    k,
    K,
    **decode_kwargs,
) -> DecodeResult:
    """Run ``DualSetEncoder`` first, then decode new DSO trees from its h_0."""

    h_0 = encoder(theta_acc, acc_mask, theta_rej, rej_mask, h=h, k=k, K=K)
    forbid_observed = decode_kwargs.pop("forbid_observed", True)
    if forbid_observed and decode_kwargs.get("forbidden_terms") is None:
        forbidden_terms = (
            decoder.token_map.forbidden_from_encoder(theta_acc, acc_mask)
            + decoder.token_map.forbidden_from_encoder(theta_rej, rej_mask)
        )
        decode_kwargs["forbidden_terms"] = forbidden_terms
        decode_kwargs.setdefault("forbidden_terms_encoding", "dso")
    return decoder(h_0, **decode_kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)

    library = make_dso_library(
        input_names=("t", "x", "u"),
        function_set=("1", "n2", "mul", "n3"),
        protected=True,
    )
    decoder = DSORNNDecoder(
        library=library,
        embed_size=32,
        hidden_size=128,
        max_length=12,
        min_length=1,
        never_sample_tokens=("1",),
    )
    token_map = decoder.token_map

    # This tensor has the same shape as encoder.DualSetEncoder(...): [batch, 128].
    h0 = torch.randn(2, 128)

    # Encoder-side theta tensors use 0 as PAD and shift real DSO ids by +1.
    known_terms = token_map.dso_batch_to_encoder([
        ["1"],
        ["x"],
        ["t"],
        ["u"]
    ], pad_to=1)
    result = decoder(
        h0,
        forbidden_terms=known_terms,
        forbidden_terms_encoding="encoder",
        max_len=12,
    )

    print("=== DSO RNN decoder smoke test ===")
    print(f"DSO vocab size: {library.L}; encoder vocab size: {token_map.encoder_vocab_size}")
    print(f"Known terms in encoder ids: {known_terms.tolist()}")
    print(f"Known terms converted back to DSO ids: {token_map.forbidden_from_encoder(known_terms)}")
    for idx, (ids, names, tree, logp, entropy, mean_entropy) in enumerate(
        zip(
            result.actions,
            result.token_names,
            result.trees,
            result.log_probs,
            result.entropies,
            result.mean_entropies,
        )
    ):
        valid_ids = ids[: result.lengths[idx]].tolist()
        print(f"[{idx}] ids: {valid_ids}")
        print(f"[{idx}] prefix: {' -> '.join(names)}")
        print(f"[{idx}] tree: {tree}")
        print(f"[{idx}] log_prob: {float(logp):.4f}")
        print(f"[{idx}] entropy_sum: {float(entropy):.4f}")
        print(f"[{idx}] entropy_mean: {float(mean_entropy):.4f}")
