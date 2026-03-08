"""
Per-expert auxiliary prediction heads.

Each expert type gets a lightweight auxiliary head that provides
expert-specific gradient signal, separate from the combined output's
main CE loss. This pushes Type A toward reconstruction fidelity
and Type B toward generative coherence.

Design:
  - Type A (EncoderDecoderExpert): reconstruction head — given the expert's
    output for active tokens, predict the target tokens directly.
    Gradient says "did YOUR output reconstruct accurately?"

  - Type B (DecoderOnlyExpert): coherence head — given the expert's output,
    predict the NEXT token (shifted by 1). Gradient says "does YOUR output
    maintain local coherence/fluency?"

  - Shared expert: no auxiliary head. Stays general.

The heads use the same factored embedding path as the main LM head
(d_model → inner_dim → vocab via weight-tied embedding) to keep
parameter overhead minimal (~17K per head vs ~4.2M for a direct projection).

One head per expert TYPE, shared across all layers. Two heads total
for the default config (1 Type A, 1 Type B).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAuxHead(nn.Module):
    """Lightweight auxiliary prediction head using factored embedding path.

    Projects d_model → inner_dim via a learned adapter, then uses the
    weight-tied embedding table (inner_dim → vocab) for token prediction.
    Total new params: LayerNorm(d_model) + Linear(d_model, inner_dim) ≈ 17K.
    """

    def __init__(self, d_model: int, inner_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.adapter = nn.Linear(d_model, inner_dim, bias=False)
        # vocab_proj will be set externally to share weights with embedding
        self.vocab_proj = None

    def set_vocab_proj(self, vocab_proj: nn.Linear):
        """Share the LM head's vocab projection (weight-tied with embedding)."""
        self.vocab_proj = vocab_proj

    def forward(self, expert_output: torch.Tensor) -> torch.Tensor:
        """
        expert_output: (n_active_tokens, d_model)
        Returns: (n_active_tokens, vocab_size) — logits
        """
        h = self.adapter(self.norm(expert_output))
        return self.vocab_proj(h)


class ExpertAuxLossComputer:
    """Computes per-expert auxiliary losses during MoE forward pass.

    Uses a single shared head per expert type (not per-layer).

    Usage in MoELayer.forward:
        aux_computer = ExpertAuxLossComputer(type_a_head, type_b_head)

        for expert_idx in range(n_experts):
            ...
            out = expert(tokens)
            aux_computer.add_expert_output(expert_type, out,
                                           active_positions, targets_flat)
            ...

        expert_aux_loss = aux_computer.get_loss()
    """

    def __init__(self, type_a_head: ExpertAuxHead, type_b_head: ExpertAuxHead,
                 ignore_index: int = -100):
        self.type_a_head = type_a_head
        self.type_b_head = type_b_head
        self.ignore_index = ignore_index
        self.losses = []

    def add_expert_output(self, expert_type: str, expert_output: torch.Tensor,
                          active_positions: torch.Tensor,
                          targets_flat: torch.Tensor):
        """Record one expert's output for auxiliary loss computation.

        expert_type: 'A' or 'B'
        expert_output: (n_active, d_model) — the expert's raw output (delta)
        active_positions: (n_active,) — indices into the flattened B*S dimension
        targets_flat: (B*S,) — target token ids for the full sequence
        """
        if len(active_positions) == 0:
            return

        if expert_type == 'A' and self.type_a_head is not None:
            # Type A objective: reconstruct target tokens at active positions
            logits = self.type_a_head(expert_output)
            targets = targets_flat[active_positions]

            valid = (targets != 0) & (targets != self.ignore_index)
            if valid.sum() > 0:
                loss = F.cross_entropy(logits[valid], targets[valid])
                self.losses.append(loss)

        elif expert_type == 'B' and self.type_b_head is not None:
            # Type B objective: predict next token (coherence)
            shifted_positions = active_positions + 1
            max_pos = targets_flat.shape[0]
            in_bounds = shifted_positions < max_pos
            if in_bounds.sum() > 0:
                logits = self.type_b_head(expert_output[in_bounds])
                targets = targets_flat[shifted_positions[in_bounds]]

                valid = (targets != 0) & (targets != self.ignore_index)
                if valid.sum() > 0:
                    loss = F.cross_entropy(logits[valid], targets[valid])
                    self.losses.append(loss)

    def get_loss(self) -> torch.Tensor:
        """Return mean auxiliary loss across all experts that fired."""
        if not self.losses:
            return torch.tensor(0.0)
        return sum(self.losses) / len(self.losses)
