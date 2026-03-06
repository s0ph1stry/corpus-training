"""
Expert types and router for the heterogeneous MoE.

Type A (EncoderDecoderExpert): cross-attn (K/V from encoder) → FFN (2x d_model)
Type B (DecoderOnlyExpert): FFN only (2x d_model)
SharedExpert: always-on FFN (4x d_model)

v2 Router: ReLU gating with learned encoder_available signal.
Variable sparsity — tokens can activate 0, 1, or all experts.

MOEManager: tracks routing statistics, optional z-loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.encoder import RotaryPositionalEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE. Shared by both expert types."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(config.head_dim, config.context_len * 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Causal mask
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Q from decoder hidden states, K/V from encoder output."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                encoder_mask: torch.Tensor = None):
        """
        x: (n_tokens, d_model) — dispatched decoder tokens (not batched)
        encoder_out: (n_tokens, enc_seq_len, d_model) — per-token encoder output
        encoder_mask: (n_tokens, enc_seq_len) — valid positions in encoder output
        """
        N, D = x.shape
        if encoder_out is None or encoder_out.shape[0] == 0:
            return torch.zeros_like(x)

        enc_S = encoder_out.shape[1]

        # Check for fully-masked encoder (encoder_available=False samples)
        # softmax over all -inf = NaN, so return zeros for those tokens
        if encoder_mask is not None:
            all_masked = ~encoder_mask.any(dim=-1)  # (N,) True if all positions masked
            if all_masked.all():
                return torch.zeros_like(x)

        # x is (N, D) -> add seq dim for attention: (N, 1, D)
        q = self.q_proj(x).view(N, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_out).view(N, enc_S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_out).view(N, enc_S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if encoder_mask is not None:
            attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, enc_S)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows with zeros (safe for partial masking)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, 1, D)
        return out.squeeze(1)  # (N, D)


class ExpertFFN(nn.Module):
    """Feed-forward network for an individual expert."""

    def __init__(self, config: ModelConfig, d_ff: int = None):
        super().__init__()
        ffn_dim = d_ff if d_ff is not None else config.d_ff
        self.net = nn.Sequential(
            nn.Linear(config.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderDecoderExpert(nn.Module):
    """Type A expert: cross-attention to encoder output + FFN.

    Used for reconstruction, comparison, truth-checking.
    The reflection mode — it can look back at a source.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(config.d_model)
        self.cross_attn = CrossAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = ExpertFFN(config, d_ff=config.d_model * 2)

    def forward(self, x: torch.Tensor,
                encoder_out: torch.Tensor = None,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (n_tokens, d_model) — dispatched tokens

        Returns DELTA only (no residual). The MoE dispatch layer adds the residual.
        """
        # Cross-attention delta
        ca = self.cross_attn(self.cross_attn_norm(x), encoder_out, encoder_mask)
        # FFN needs the cross-attended representation
        h = x + ca
        ff = self.ffn(self.ffn_norm(h))
        # Return delta only: ca + ff (dispatch adds x back via outer residual)
        return ca + ff


class DecoderOnlyExpert(nn.Module):
    """Type B expert: FFN only (self-attention is shared at the layer level).

    Used for autoregressive generation from internal knowledge.
    The action mode — it generates from what it knows.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = ExpertFFN(config, d_ff=config.d_model * 2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """x: (n_tokens, d_model) — dispatched tokens. Extra kwargs ignored.

        Returns DELTA only (no residual). The MoE dispatch layer adds the residual.
        """
        return self.ffn(self.ffn_norm(x))


class Router(nn.Module):
    """MoE router with ReLU gating and learned encoder_available signal.

    v2: ReLU replaces softmax+top-k. Variable sparsity — a token can activate
    0, 1, or all experts. Gate outputs are ReLU'd then L1-normalized.

    The encoder_available signal is projected from a 1-bit scalar through a learned
    Linear(1, d_model//4) + LayerNorm. (Grok audit patch, 2026-03-04)

    Runs in float32 for numerical stability regardless of model dtype.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts

        # Learned encoder_available projection (Grok patch: upgrade from 1-bit)
        self.enc_signal_dim = max(config.d_model // 4, 1)
        self.enc_signal_proj = nn.Linear(1, self.enc_signal_dim)
        self.enc_signal_norm = nn.LayerNorm(self.enc_signal_dim)

        # Gate with bias (init +0.1 to prevent dead ReLU at init)
        self.gate = nn.Linear(config.d_model + self.enc_signal_dim, config.n_experts, bias=True)
        nn.init.constant_(self.gate.bias, 0.1)

    def forward(self, hidden_states: torch.Tensor,
                encoder_available: torch.Tensor) -> tuple:
        """
        hidden_states: (n_tokens, d_model)
        encoder_available: (n_tokens, 1)

        Returns:
            gate_weights: (n_tokens, n_experts) — ReLU + L1-normalized weights
            gate_logits: (n_tokens, n_experts) — raw logits for z-loss / logging
        """
        # Project encoder_available through learned embedding
        enc_signal = self.enc_signal_proj(encoder_available.float())
        enc_signal = self.enc_signal_norm(enc_signal)

        # Gate
        gate_input = torch.cat([hidden_states.float(), enc_signal], dim=-1)
        gate_logits = self.gate(gate_input)  # (n_tokens, n_experts)

        # ReLU gating: variable sparsity
        scores = F.relu(gate_logits)

        # L1 normalize (tokens with all-zero scores get zero weights — residual only)
        gate_weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        return gate_weights, gate_logits


class SharedExpert(nn.Module):
    """Always-on shared expert with wide FFN (4x d_model).

    Runs on all tokens regardless of routing. Provides a stable gradient
    path while routed experts specialize.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_ff = config.d_ff_shared  # 4x d_model
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.up = nn.Linear(config.d_model, d_ff)
        self.down = nn.Linear(d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_tokens, d_model). Returns delta only."""
        h = self.ffn_norm(x)
        return self.dropout(self.down(F.gelu(self.up(h))))


class MOEManager:
    """Tracks routing statistics across MoE layers.

    v2: Balance loss removed (ReLU routing provides natural specialization).
    Only z-loss remains as optional regularizer (weight defaults to 0.0).

    Call reset() at the start of each forward pass.
    Call get_aux_loss() to retrieve accumulated loss for backprop.
    """

    def __init__(self, **kwargs):
        # Accept any kwargs for backward compat (alpha_start, alpha_end, ramp_steps)
        self.z_losses = []

        # Routing entropy tracking
        self.layer_entropies = []
        self.layer_expert_fracs = []
        self.entropy_history = []
        self.collapse_alert = False

    def reset(self):
        self.z_losses = []
        self.layer_entropies = []
        self.layer_expert_fracs = []

    def get_ramped_alpha(self, global_step: int = 0) -> float:
        """No-op — returns 0.0. Kept to prevent trainer crash."""
        return 0.0

    def add_routing_stats(self, gate_weights: torch.Tensor, n_experts: int):
        """Track per-expert activation fraction and routing entropy.

        gate_weights: (n_tokens, n_experts) — ReLU-normalized weights
        """
        # Per-expert activation fraction: fraction of tokens with nonzero weight
        active = (gate_weights > 0).float()
        fracs = active.mean(dim=0)  # (n_experts,)
        self.layer_expert_fracs.append(fracs.detach().cpu().tolist())

        # Routing entropy over the activation distribution
        # Treat fracs as a probability distribution (normalize)
        p = fracs / (fracs.sum() + 1e-10)
        entropy = -(p * (p + 1e-10).log()).sum().item()
        self.layer_entropies.append(entropy)

    def add_z_loss(self, router_logits: torch.Tensor):
        """Router z-loss: penalizes large logits to prevent router instability.

        L_z = (1/n) * sum(log(sum(exp(logits)))^2)
        """
        log_z = torch.logsumexp(router_logits.float(), dim=-1)
        z_loss = (log_z ** 2).mean()
        self.z_losses.append(z_loss)

    def get_aux_loss(self, global_step: int = 0,
                     z_weight: float = 0.0) -> torch.Tensor:
        """Return z-loss (or 0.0 if weight is 0 or no z-losses accumulated)."""
        total = torch.tensor(0.0)
        if z_weight > 0 and self.z_losses:
            device = self.z_losses[0].device
            total = total.to(device)
            total = total + z_weight * sum(self.z_losses) / len(self.z_losses)
        return total

    def get_routing_entropy(self) -> dict:
        """Return per-layer routing entropy from the last forward pass."""
        return {f'routing_entropy/layer_{i}': e
                for i, e in enumerate(self.layer_entropies)}

    def get_expert_fracs(self) -> dict:
        """Return per-layer, per-expert activation fractions.

        Keys like 'expert_frac/layer_0/expert_0', values are fraction of tokens
        with nonzero gate weight for that expert.
        """
        result = {}
        for layer_i, fracs in enumerate(self.layer_expert_fracs):
            for expert_j, f in enumerate(fracs):
                result[f'expert_frac/layer_{layer_i}/expert_{expert_j}'] = f
        return result

    def check_collapse(self, threshold: float = 0.4, window: int = 3) -> bool:
        """Check if routing entropy has been below threshold for window checkpoints."""
        if self.layer_entropies:
            mean_entropy = sum(self.layer_entropies) / len(self.layer_entropies)
            self.entropy_history.append(mean_entropy)

        if len(self.entropy_history) < window:
            return False

        recent = self.entropy_history[-window:]
        self.collapse_alert = all(e < threshold for e in recent)
        return self.collapse_alert
