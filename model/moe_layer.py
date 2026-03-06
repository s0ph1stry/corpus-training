"""
MoE layer v2: Hymba SSM+Attention fusion → shared expert → ReLU-routed experts.

Each decoder layer:
  1. Hymba block: shared pre-norm → parallel SSM + CausalAttn → per-channel fusion → residual
  2. Shared expert: always-on wide FFN (4x d_model) on all tokens
  3. ReLU-routed experts: Type A (cross-attn + FFN) and Type B (FFN only), 2x d_model

No capacity enforcement — ReLU provides natural sparsity (tokens can activate 0, 1, or all experts).
"""

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.ssm import Mamba2Block
from model.experts import (
    CausalSelfAttention,
    EncoderDecoderExpert,
    DecoderOnlyExpert,
    SharedExpert,
    Router,
    MOEManager,
)


class MoELayer(nn.Module):
    """Single MoE decoder layer: Hymba fusion → shared expert → routed experts.

    The Hymba block fuses SSM (sequential/local) and attention (global) pathways
    with learned per-channel mixing weights. This replaces the v1 shared self-attention.
    """

    def __init__(self, config: ModelConfig, moe_manager: MOEManager):
        super().__init__()
        self.config = config
        self.moe_manager = moe_manager

        # Hymba fusion: shared norm → parallel SSM + Attn → per-channel mix
        self.hymba_norm = nn.LayerNorm(config.d_model)
        self.ssm = Mamba2Block(config)
        self.self_attn = CausalSelfAttention(config)
        self.beta_ssm = nn.Parameter(0.7 * torch.ones(config.d_model))
        self.beta_attn = nn.Parameter(0.3 * torch.ones(config.d_model))

        # Shared expert (always-on, 4x d_model FFN)
        self.shared_expert = SharedExpert(config)

        # Router (ReLU gating, gates only routed experts)
        self.router = Router(config)

        # Routed experts (2x d_model FFN each)
        experts = []
        self.expert_types = []
        for _ in range(config.n_type_a):
            experts.append(EncoderDecoderExpert(config))
            self.expert_types.append('A')
        for _ in range(config.n_type_b_computed):
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

        # ─── 1. Hymba fusion: parallel SSM + Attention with per-channel mixing ───
        h = self.hymba_norm(x)
        ssm_out = self.ssm(h)
        attn_out = self.self_attn(h, padding_mask)
        fused = self.beta_ssm * ssm_out + self.beta_attn * attn_out
        x = x + fused  # pre-norm residual

        # ─── 2. Flatten for expert processing ───
        x_flat = x.view(B * S, D)

        # Build per-token encoder_available signal
        if encoder_available is not None:
            enc_avail_per_token = encoder_available.unsqueeze(1).expand(B, S).reshape(B * S, 1).float()
        else:
            enc_avail_per_token = torch.zeros(B * S, 1, device=x.device)

        # ─── 3. Shared expert (always-on) ───
        shared_delta = self.shared_expert(x_flat)

        # ─── 4. ReLU-routed experts ───
        gate_weights, gate_logits = self.router(x_flat, enc_avail_per_token)
        # gate_weights: (B*S, n_experts) — ReLU + L1-normalized

        # Track routing stats and optional z-loss
        self.moe_manager.add_routing_stats(gate_weights, self.config.n_experts)
        if self.config.router_z_loss_weight > 0:
            self.moe_manager.add_z_loss(gate_logits)

        # Dispatch to each routed expert
        routed_delta = torch.zeros_like(x_flat)

        for expert_idx in range(self.config.n_experts):
            # Select tokens with nonzero gate weight for this expert
            weights_i = gate_weights[:, expert_idx]  # (B*S,)
            active_mask = weights_i > 0
            active_positions = active_mask.nonzero(as_tuple=True)[0]

            if len(active_positions) == 0:
                continue

            tokens = x_flat[active_positions]  # (n_active, D)
            w = weights_i[active_positions].unsqueeze(-1)  # (n_active, 1)

            expert = self.experts[expert_idx]
            expert_type = self.expert_types[expert_idx]

            if expert_type == 'A' and encoder_out is not None:
                # Per-batch-element loop for cross-attention (MPS/VRAM optimization)
                batch_indices = active_positions // S
                unique_batches = batch_indices.unique()
                out = torch.zeros_like(tokens)
                for b_idx in unique_batches:
                    mask = batch_indices == b_idx
                    b_tokens = tokens[mask]
                    n_b = mask.sum()
                    # repeat() not expand() — MPS Metal stride checks
                    b_enc = encoder_out[b_idx:b_idx+1].repeat(n_b, 1, 1)
                    b_enc_mask = None
                    if encoder_mask is not None:
                        b_enc_mask = encoder_mask[b_idx:b_idx+1].repeat(n_b, 1)
                    out[mask] = expert(b_tokens, b_enc, b_enc_mask).to(out.dtype)
            else:
                out = expert(tokens)

            routed_delta.index_add_(
                0, active_positions, (w * out).to(routed_delta.dtype)
            )

        # ─── 5. Combine and reshape ───
        combined = (shared_delta + routed_delta).view(B, S, D)
        return x + combined
