"""
MoE layer: wires Router + heterogeneous experts.

Dispatch-and-combine with per-expert branching:
  - Type A experts receive (tokens, encoder_out, batch_indices, mask)
  - Type B experts receive (tokens,) only

Capacity: floor(top_k × capacity_factor × tokens / n_experts).
Train capacity_factor=1.25, eval=2.0. Overflow tokens pass through
residual only (no expert processing).
"""

import math
import torch
import torch.nn as nn

from model.config import ModelConfig
from model.experts import (
    CausalSelfAttention,
    EncoderDecoderExpert,
    DecoderOnlyExpert,
    Router,
    MOEManager,
)


class MoELayer(nn.Module):
    """Single MoE decoder layer: shared self-attn → router → expert dispatch → combine.

    The self-attention is shared across all experts (not per-expert).
    Only the expert-specific processing (cross-attn for Type A, FFN for both)
    is routed.
    """

    def __init__(self, config: ModelConfig, moe_manager: MOEManager):
        super().__init__()
        self.config = config
        self.moe_manager = moe_manager

        # Shared causal self-attention (pre-expert)
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config)

        # Router
        self.router = Router(config)

        # Heterogeneous experts
        experts = []
        self.expert_types = []  # 'A' or 'B' for each expert
        for i in range(config.n_type_a):
            experts.append(EncoderDecoderExpert(config))
            self.expert_types.append('A')
        for i in range(config.n_type_b_computed):
            experts.append(DecoderOnlyExpert(config))
            self.expert_types.append('B')
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor,
                encoder_out: torch.Tensor = None,
                encoder_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                encoder_available: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        encoder_out: (batch, enc_seq_len, d_model) or None
        encoder_mask: (batch, enc_seq_len) or None
        padding_mask: (batch, seq_len) — True=valid, False=pad
        encoder_available: (batch,) — binary per sample

        Returns: (batch, seq_len, d_model)
        """
        B, S, D = x.shape

        # 1. Shared causal self-attention
        x = x + self.self_attn(self.self_attn_norm(x), padding_mask)

        # 2. Flatten for routing
        x_flat = x.view(B * S, D)  # (B*S, D)

        # Build per-token encoder_available signal
        if encoder_available is not None:
            enc_avail_per_token = encoder_available.unsqueeze(1).expand(B, S).reshape(B * S, 1).float()
        else:
            enc_avail_per_token = torch.zeros(B * S, 1, device=x.device)

        # 3. Route
        expert_indices, expert_weights, router_logits = self.router(
            x_flat, enc_avail_per_token
        )
        # expert_indices: (B*S, top_k), expert_weights: (B*S, top_k)

        # Accumulate aux losses
        self.moe_manager.add_balance_loss(
            router_logits, expert_indices, self.config.n_experts
        )
        self.moe_manager.add_z_loss(router_logits)

        # 4. Capacity check
        n_tokens = B * S
        if self.training:
            capacity_factor = self.config.capacity_factor_train
        else:
            capacity_factor = self.config.capacity_factor_eval
        capacity = math.floor(
            self.config.top_k * capacity_factor * n_tokens / self.config.n_experts
        )

        # 5. Dispatch to experts (grouped by expert for efficiency)
        # Output starts as zeros (overflow tokens get residual only)
        expert_output = torch.zeros_like(x_flat)

        # Group all (position, top_k_slot) pairs by expert index
        expert_to_tokens = {}
        for k in range(self.config.top_k):
            for expert_idx in range(self.config.n_experts):
                token_mask = expert_indices[:, k] == expert_idx
                token_positions = token_mask.nonzero(as_tuple=True)[0]
                if len(token_positions) > 0:
                    entry = expert_to_tokens.setdefault(expert_idx, {'positions': [], 'slots': []})
                    entry['positions'].append(token_positions)
                    entry['slots'].append(torch.full_like(token_positions, k))

        # Process each expert once with all its tokens
        for expert_idx, token_info in expert_to_tokens.items():
            all_positions = torch.cat(token_info['positions'])
            all_slots = torch.cat(token_info['slots'])

            # Enforce capacity across all top-k slots combined
            if len(all_positions) > capacity:
                all_positions = all_positions[:capacity]
                all_slots = all_slots[:capacity]

            tokens = x_flat[all_positions]  # (n_selected, D)
            weights = expert_weights[all_positions, all_slots].unsqueeze(-1)

            expert = self.experts[expert_idx]
            expert_type = self.expert_types[expert_idx]

            if expert_type == 'A' and encoder_out is not None:
                # Process cross-attention per batch element to avoid
                # duplicating encoder_out for every token (VRAM optimization).
                # Instead of gathering (N, enc_S, D) which copies encoder for
                # every token, we loop over batch elements and share encoder KV.
                batch_indices = all_positions // S
                unique_batches = batch_indices.unique()
                out = torch.zeros_like(tokens)
                for b_idx in unique_batches:
                    mask = batch_indices == b_idx
                    b_tokens = tokens[mask]  # (n_b, D)
                    # Single encoder output repeated for tokens from this batch element.
                    # repeat() copies memory (expand() breaks MPS Metal stride checks)
                    # but the per-batch-element loop keeps each copy small.
                    n_b = mask.sum()
                    b_enc = encoder_out[b_idx:b_idx+1].repeat(n_b, 1, 1)
                    b_enc_mask = None
                    if encoder_mask is not None:
                        b_enc_mask = encoder_mask[b_idx:b_idx+1].repeat(n_b, 1)
                    out[mask] = expert(b_tokens, b_enc, b_enc_mask)
            else:
                out = expert(tokens)

            expert_output.index_add_(0, all_positions, weights * out)

        # 6. Reshape and add residual
        expert_output = expert_output.view(B, S, D)
        return x + expert_output
