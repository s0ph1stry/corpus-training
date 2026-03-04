"""
Expert types and router for the heterogeneous MoE.

Type A (EncoderDecoderExpert): causal self-attn → cross-attn (K/V from encoder) → FFN
Type B (DecoderOnlyExpert): causal self-attn → FFN

Router: Linear(d_model + 1, n_experts), where +1 is the encoder_available signal.
The router learns when to reflect (Type A) and when to generate (Type B).

MOEManager: accumulates auxiliary losses (balance + z-loss) across layers.
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

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
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
        self.ffn = ExpertFFN(config)

    def forward(self, x: torch.Tensor,
                encoder_out: torch.Tensor = None,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (n_tokens, d_model) — dispatched tokens
        encoder_out: (n_tokens, enc_seq_len, d_model)
        encoder_mask: (n_tokens, enc_seq_len)
        """
        # Cross-attention with residual
        x = x + self.cross_attn(self.cross_attn_norm(x), encoder_out, encoder_mask)
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyExpert(nn.Module):
    """Type B expert: FFN only (self-attention is shared at the layer level).

    Used for autoregressive generation from internal knowledge.
    The action mode — it generates from what it knows.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = ExpertFFN(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """x: (n_tokens, d_model) — dispatched tokens. Extra kwargs ignored."""
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Router(nn.Module):
    """MoE router with learned encoder_available signal projection.

    Input: hidden_state (d_model) concatenated with encoder signal projection (d_model//4).
    Output: top-k expert selection with softmax weights.

    The encoder_available signal is projected from a 1-bit scalar through a learned
    Linear(1, d_model//4) + LayerNorm, giving the router a richer signal than raw
    binary concatenation. (Grok audit patch, 2026-03-04)

    Runs in float32 for numerical stability regardless of model dtype.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        # Learned encoder_available projection (Grok patch: upgrade from 1-bit)
        self.enc_signal_dim = max(config.d_model // 4, 1)
        self.enc_signal_proj = nn.Linear(1, self.enc_signal_dim)
        self.enc_signal_norm = nn.LayerNorm(self.enc_signal_dim)

        self.gate = nn.Linear(config.d_model + self.enc_signal_dim, config.n_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                encoder_available: torch.Tensor) -> tuple:
        """
        hidden_states: (batch * seq_len, d_model) — flattened token representations
        encoder_available: (batch * seq_len, 1) — binary signal per token

        Returns:
            expert_indices: (n_tokens, top_k) — selected expert indices
            expert_weights: (n_tokens, top_k) — softmax weights for selected experts
            router_logits: (n_tokens, n_experts) — raw logits for aux loss
        """
        # Project encoder_available through learned embedding
        enc_signal = self.enc_signal_proj(encoder_available.float())
        enc_signal = self.enc_signal_norm(enc_signal)

        # Concatenate with hidden state
        gate_input = torch.cat([hidden_states.float(), enc_signal], dim=-1)
        router_logits = self.gate(gate_input)  # (n_tokens, n_experts)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights.float(), dim=-1)

        return topk_indices, topk_weights, router_logits


class MOEManager:
    """Accumulates auxiliary losses and routing statistics across MoE layers.

    Call reset() at the start of each forward pass.
    Call get_aux_loss() to retrieve the accumulated loss for backprop.

    Grok audit patches (2026-03-04):
    - Alpha ramping: balance_weight ramps from alpha_start to alpha_end over ramp_steps
    - Routing entropy monitoring: tracks per-layer entropy, alerts on collapse
    """

    def __init__(self, alpha_start: float = 0.01, alpha_end: float = 0.07,
                 ramp_steps: int = 10000):
        self.balance_losses = []
        self.z_losses = []

        # Alpha ramp parameters (Grok patch)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.ramp_steps = ramp_steps

        # Routing entropy tracking (Grok patch)
        self.layer_entropies = []  # per-layer routing entropy for current forward pass
        self.entropy_history = []  # list of per-checkpoint mean entropies
        self.collapse_alert = False

    def reset(self):
        self.balance_losses = []
        self.z_losses = []
        self.layer_entropies = []

    def get_ramped_alpha(self, global_step: int) -> float:
        """Linear ramp from alpha_start to alpha_end over ramp_steps."""
        if global_step >= self.ramp_steps:
            return self.alpha_end
        t = global_step / self.ramp_steps
        return self.alpha_start + t * (self.alpha_end - self.alpha_start)

    def add_balance_loss(self, router_logits: torch.Tensor,
                         expert_indices: torch.Tensor,
                         n_experts: int):
        """Switch Transformer load-balancing loss.

        Encourages each expert to receive roughly equal traffic.
        L_balance = n_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = average routing probability for expert i
        """
        probs = F.softmax(router_logits.float(), dim=-1)

        # f_i: fraction of tokens dispatched to each expert
        one_hot = F.one_hot(expert_indices[:, 0], n_experts).float()
        f = one_hot.mean(dim=0)

        # P_i: average probability assigned to each expert
        P = probs.mean(dim=0)

        balance_loss = n_experts * (f * P).sum()
        self.balance_losses.append(balance_loss)

        # Track routing entropy (Grok patch)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        self.layer_entropies.append(entropy.item())

    def add_z_loss(self, router_logits: torch.Tensor):
        """Router z-loss: penalizes large logits to prevent router collapse.

        L_z = (1/n) * sum(log(sum(exp(logits)))^2)
        """
        log_z = torch.logsumexp(router_logits.float(), dim=-1)
        z_loss = (log_z ** 2).mean()
        self.z_losses.append(z_loss)

    def get_aux_loss(self, global_step: int = 0,
                     z_weight: float = 0.001) -> torch.Tensor:
        """Return weighted sum of all accumulated auxiliary losses.

        balance_weight is computed from alpha ramp schedule based on global_step.
        """
        balance_weight = self.get_ramped_alpha(global_step)
        total = torch.tensor(0.0)
        if self.balance_losses:
            device = self.balance_losses[0].device
            total = total.to(device)
            total = total + balance_weight * sum(self.balance_losses) / len(self.balance_losses)
        if self.z_losses:
            device = self.z_losses[0].device
            total = total.to(device)
            total = total + z_weight * sum(self.z_losses) / len(self.z_losses)
        return total

    def get_routing_entropy(self) -> dict:
        """Return per-layer routing entropy from the last forward pass."""
        return {f'routing_entropy/layer_{i}': e
                for i, e in enumerate(self.layer_entropies)}

    def check_collapse(self, threshold: float = 0.4, window: int = 3) -> bool:
        """Check if routing entropy has been below threshold for window checkpoints.

        Call this at checkpoint time with the mean entropy of the evaluation batch.
        Returns True if collapse is detected.
        """
        if self.layer_entropies:
            mean_entropy = sum(self.layer_entropies) / len(self.layer_entropies)
            self.entropy_history.append(mean_entropy)

        if len(self.entropy_history) < window:
            return False

        recent = self.entropy_history[-window:]
        self.collapse_alert = all(e < threshold for e in recent)
        return self.collapse_alert
