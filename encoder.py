"""Set encoder for accepted/rejected symbolic trees.

Encoder-side token convention:
    0      = PAD for nn.Embedding(padding_idx=0)
    1..L   = DSO token ids shifted by +1

Decoder/DSO-side token convention:
    0..L-1 = real DSO library token ids
    -1     = decoder output padding
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TreeEmbedding(nn.Module):
    """Embed one set of prefix-encoded symbolic trees with a BiLSTM."""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [batch_size, num_trees, seq_len]
        batch_size, num_trees, seq_len = tau.size()
        tau_flat = tau.reshape(-1, seq_len)
        embeds = self.embedding(tau_flat)
        _, (hidden, _) = self.bilstm(embeds)
        tree_embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return tree_embedding.view(batch_size, num_trees, -1)


class SetEncoder(nn.Module):
    """DeepSets-style mean pooling over a masked set of tree embeddings."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, tree_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mapped = self.mlp(tree_embeddings)
        mask = mask.unsqueeze(-1).to(dtype=mapped.dtype, device=mapped.device)
        mapped = mapped * mask
        counts = torch.sum(mask, dim=1).clamp(min=1)
        return torch.sum(mapped, dim=1) / counts


class DualSetEncoder(nn.Module):
    """Encode (Theta_acc, Theta_rej, h, k) into the shared actor/critic context."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        tree_hidden_size: int,
        set_hidden_size: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.tree_embedder = TreeEmbedding(vocab_size, embed_size, tree_hidden_size)
        self.acc_set_encoder = SetEncoder(tree_hidden_size, set_hidden_size)
        self.rej_set_encoder = SetEncoder(tree_hidden_size, set_hidden_size)
        self.context_linear = nn.Linear(set_hidden_size * 2 + 2, rnn_hidden_size)

    def forward(
        self,
        theta_acc: torch.Tensor,
        acc_mask: torch.Tensor,
        theta_rej: torch.Tensor,
        rej_mask: torch.Tensor,
        *,
        h,
        k,
        K,
    ) -> torch.Tensor:
        e_acc = self.tree_embedder(theta_acc)
        e_rej = self.tree_embedder(theta_rej)
        c_acc = self.acc_set_encoder(e_acc, acc_mask)
        c_rej = self.rej_set_encoder(e_rej, rej_mask)
        batch_size = c_acc.shape[0]

        def batch_scalar(value, name):
            value = torch.as_tensor(value, dtype=c_acc.dtype, device=c_acc.device)
            if value.numel() == 1:
                value = value.reshape(1).expand(batch_size)
            elif value.numel() == batch_size:
                value = value.reshape(batch_size)
            else:
                raise ValueError(f"{name} must be a scalar or have one value per batch row.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite.")
            return value

        h = batch_scalar(h, "h")
        k = batch_scalar(k, "k")
        K = batch_scalar(K, "K")
        if bool(((K <= 0) | (h < 0) | (h > k) | (k > K)).any()):
            raise ValueError("Search counters must satisfy 0 <= h <= k <= K with K > 0.")
        counters = torch.stack([h / K, k / K], dim=-1)
        return self.context_linear(torch.cat([c_acc, c_rej, counters], dim=-1))


if __name__ == "__main__":
    from decoder import DSORNNDecoder, generate_from_encoder, make_dso_library

    torch.manual_seed(0)

    tree_hidden = 32
    set_hidden = 64
    rnn_hidden = 128

    library = make_dso_library(
        input_names=("t", "x", "u"),
        function_set=("1", "n2", "mul", "n3"),
        protected=True,
    )
    decoder = DSORNNDecoder(
        library=library,
        embed_size=32,
        hidden_size=rnn_hidden,
        max_length=10,
        never_sample_tokens=("1",),
    )
    token_map = decoder.token_map

    encoder = DualSetEncoder(
        vocab_size=token_map.encoder_vocab_size,
        embed_size=16,
        tree_hidden_size=tree_hidden,
        set_hidden_size=set_hidden,
        rnn_hidden_size=rnn_hidden,
    )

    # DSO terms are written with library token names. The adapter shifts them
    # into encoder ids, where 0 remains reserved for PAD.
    theta_acc_dso = [
        ["1"],
        ["x"],
        ["t"],
        ["u"],
    ]
    theta_rej_dso = ["1"]

    acc_tensor = token_map.dso_batch_to_encoder(theta_acc_dso, pad_to=3).unsqueeze(0)
    rej_tensor = token_map.dso_batch_to_encoder(theta_rej_dso, pad_to=3).unsqueeze(0)
    acc_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    rej_mask = torch.tensor([[1.0]])

    h_0 = encoder(acc_tensor, acc_mask, rej_tensor, rej_mask, h=0, k=0, K=6)
    result = generate_from_encoder(
        encoder,
        decoder,
        acc_tensor,
        acc_mask,
        rej_tensor,
        rej_mask,
        h=0,
        k=0,
        K=6,
        max_len=10,
    )

    print("=== encoder/decoder token mapping check ===")
    print(f"DSO tokens: {library.names}")
    print(f"DSO vocab size: {library.L}")
    print(f"Encoder vocab size: {token_map.encoder_vocab_size}")
    print(f"Theta_acc encoder ids: {acc_tensor.tolist()}")
    print(f"Theta_acc back to DSO ids: {token_map.forbidden_from_encoder(acc_tensor, acc_mask)}")
    print(f"h_0 shape: {tuple(h_0.shape)}")
    print(f"Generated prefix: {result.token_names[0]}")
    print(f"Generated tree: {result.trees[0]}")
