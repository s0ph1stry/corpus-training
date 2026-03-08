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
from model.expert_aux_heads import ExpertAuxHead, ExpertAuxLossComputer
from training.contrastive_losses import ExpertSimilarityLoss


class MoELayer(nn.Module):
    """Single MoE decoder layer: Hymba fusion → shared expert → routed experts.

    The Hymba block fuses SSM (sequential/local) and attention (global) pathways
    with learned per-channel mixing weights. This replaces the v1 shared self-attention.
    """

    def __init__(self, config: ModelConfig, moe_manager: MOEManager,
                 layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.moe_manager = moe_manager
        self.layer_idx = layer_idx

        # Hymba fusion: shared norm → parallel SSM + Attn → per-channel mix
        # Per-layer initialization: early layers SSM-heavy (read), late layers
        # attention-heavy (think). Creates a read→think gradient through the stack.
        self.hymba_norm = nn.LayerNorm(config.d_model)
        self.ssm = Mamba2Block(config)
        self.self_attn = CausalSelfAttention(config)
        n_layers = max(config.n_dec_layers - 1, 1)
        ssm_init = 0.8 - 0.2 * (layer_idx / n_layers)   # 0.8 → 0.6
        attn_init = 1.0 - ssm_init                        # 0.2 → 0.4
        self.beta_ssm = nn.Parameter(ssm_init * torch.ones(config.d_model))
        self.beta_attn = nn.Parameter(attn_init * torch.ones(config.d_model))

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

        # Per-expert aux heads are shared across layers, owned by the model.
        # MoELayer receives references via set_aux_heads().
        self.use_expert_aux = config.expert_aux_heads
        self._type_a_aux_head = None
        self._type_b_aux_head = None

        # Expert similarity loss (v2.4, SimSMoE): CKA between expert outputs
        self._use_cka = config.similarity_loss_weight > 0
        self.cka_loss = None
        if self._use_cka:
            self.cka_loss = ExpertSimilarityLoss(
                config.d_model,
                n_proj=config.similarity_n_proj,
                threshold=config.similarity_threshold,
            )

        # Routing contrastive loss (v2.4, ProMoE): expert prototypes
        self._use_rcl = config.rcl_weight > 0
        if self._use_rcl:
            self.expert_prototypes = nn.Parameter(
                torch.randn(config.n_experts, config.d_model) * 0.02
            )

    def set_aux_heads(self, type_a_head: ExpertAuxHead, type_b_head: ExpertAuxHead):
        """Set shared auxiliary heads (called by HeteroMoETransformer after init)."""
        self._type_a_aux_head = type_a_head
        self._type_b_aux_head = type_b_head

    def forward(self, x: torch.Tensor,
                encoder_out: torch.Tensor = None,
                encoder_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                encoder_available: torch.Tensor = None,
                decoder_targets: torch.Tensor = None,
                mode_ids: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        encoder_out: (batch, enc_seq_len, d_model) or None
        encoder_mask: (batch, enc_seq_len) or None
        padding_mask: (batch, seq_len) — True=valid, False=pad
        encoder_available: (batch,) — binary per sample
        decoder_targets: (batch, seq_len) — target tokens, needed for expert aux heads
        mode_ids: (batch,) — UL2 mode IDs (R=0, S=1, X=2) for mode-conditioned routing

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
        jitter = self.config.router_jitter_noise if self.training else 0.0

        # Expand mode_ids to per-token if provided
        mode_ids_flat = None
        if mode_ids is not None:
            mode_ids_flat = mode_ids.unsqueeze(1).expand(B, S).reshape(B * S)

        gate_weights, gate_logits = self.router(x_flat, enc_avail_per_token,
                                                jitter_noise=jitter,
                                                mode_ids=mode_ids_flat)
        # gate_weights: (B*S, n_experts) — ReLU + L1-normalized

        # Hard S-mode gate (v2.3): structurally zero Type A weights when no encoder
        if self.config.hard_s_mode_gate and encoder_available is not None:
            s_mode_mask = (enc_avail_per_token.squeeze(-1) == 0)  # (B*S,)
            if s_mode_mask.any():
                for ei in range(self.config.n_type_a):
                    gate_weights[s_mode_mask, ei] = 0.0
                # Renormalize remaining weights
                gate_sum = gate_weights[s_mode_mask].sum(dim=-1, keepdim=True) + 1e-6
                gate_weights[s_mode_mask] = gate_weights[s_mode_mask] / gate_sum

        # Track routing stats, liveness loss, and optional z-loss
        self.moe_manager.add_routing_stats(
            gate_weights, self.config.n_experts,
            min_frac=self.config.liveness_min_frac,
            gate_logits=gate_logits,
        )
        if self.config.router_z_loss_weight > 0:
            self.moe_manager.add_z_loss(gate_logits)

        # Dispatch to each routed expert
        routed_delta = torch.zeros_like(x_flat)

        # Set up per-expert aux loss computation if enabled
        aux_computer = None
        if (self.use_expert_aux and decoder_targets is not None and self.training
                and self._type_a_aux_head is not None):
            targets_flat = decoder_targets.view(B * S)
            aux_computer = ExpertAuxLossComputer(
                self._type_a_aux_head, self._type_b_aux_head
            )

        # Collect raw expert outputs for CKA computation (v2.4)
        collect_for_cka = self.training and self._use_cka
        expert_raw = {}  # expert_idx -> (active_positions, raw_output)

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
                    bi = b_idx.item()
                    mask = batch_indices == b_idx
                    b_tokens = tokens[mask]
                    n_b = mask.sum().item()
                    # repeat() not expand() — MPS Metal stride checks
                    b_enc = encoder_out[bi:bi+1].repeat(n_b, 1, 1)
                    b_enc_mask = None
                    if encoder_mask is not None:
                        b_enc_mask = encoder_mask[bi:bi+1].repeat(n_b, 1)
                    out[mask] = expert(b_tokens, b_enc, b_enc_mask).to(out.dtype)
            else:
                out = expert(tokens)

            # Store raw output for CKA (before weighting)
            if collect_for_cka:
                expert_raw[expert_idx] = (active_positions, out)

            # Per-expert auxiliary loss: tap output before weighted combination
            if aux_computer is not None:
                aux_computer.add_expert_output(
                    expert_type, out.detach() if not self.training else out,
                    active_positions, targets_flat
                )

            routed_delta.index_add_(
                0, active_positions, (w * out).to(routed_delta.dtype)
            )

        # Accumulate expert aux loss into MOEManager
        if aux_computer is not None:
            expert_aux_loss = aux_computer.get_loss()
            if expert_aux_loss.requires_grad:
                self.moe_manager.expert_aux_losses.append(expert_aux_loss)

        # ─── CKA expert similarity loss (v2.4, SimSMoE) ───
        if collect_for_cka and len(expert_raw) == self.config.n_experts:
            pos_0, out_0 = expert_raw[0]
            pos_1, out_1 = expert_raw[1]

            # Find co-activated tokens (active for both experts)
            co_active = (gate_weights[:, 0] > 0) & (gate_weights[:, 1] > 0)
            n_co = co_active.sum().item()

            if n_co >= self.cka_loss.min_tokens:
                co_positions = co_active.nonzero(as_tuple=True)[0]
                # Map co_positions to indices within each expert's output
                idx_0 = torch.searchsorted(pos_0, co_positions)
                idx_1 = torch.searchsorted(pos_1, co_positions)
                a_co = out_0[idx_0]
                b_co = out_1[idx_1]

                cka_val = self.cka_loss(a_co, b_co)
                if cka_val.requires_grad:
                    self.moe_manager.similarity_losses.append(cka_val)

        # ─── Routing contrastive loss (v2.4, ProMoE) ───
        if self.training and self._use_rcl:
            self.moe_manager.add_rcl_loss(
                gate_weights, x_flat, self.expert_prototypes,
                tau=self.config.rcl_tau,
            )

        # ─── 5. Combine and reshape ───
        combined = (shared_delta + routed_delta).view(B, S, D)
        return x + combined
