"""
Model configuration for the heterogeneous MoE transformer.

Two preset scales:
  - Tiny (4-8M params): trainable on Mac M-series, for validation
  - Small (125-250M params): Colab A100, the real experiment

Factored embeddings: vocab × inner_dim + inner_dim × d_model
saves ~74% of embedding parameters at 16K vocab.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    # Embedding
    vocab_size: int = 16233  # updated after author tokens added (202 texts, 233 author tokens + <complete>)
    embedding_inner_dim: int = 64  # factored embedding bottleneck
    pad_token_id: int = 0

    # Transformer dimensions
    d_model: int = 256
    d_ff: int = 1024  # FFN inner dim, typically 4 * d_model
    n_heads: int = 4
    dropout: float = 0.1

    # Encoder (bidirectional, shallow)
    n_enc_layers: int = 2

    # Decoder (causal, MoE)
    n_dec_layers: int = 4

    # MoE
    n_experts: int = 2  # routed experts per MoE layer
    n_type_a: int = 1  # encoder-decoder experts (n_type_b = n_experts - n_type_a)
    top_k: int = 1  # vestigial (v2 uses ReLU routing)
    capacity_factor_train: float = 1.25  # vestigial (v2 uses natural ReLU sparsity)
    capacity_factor_eval: float = 2.0  # vestigial
    aux_loss_weight: float = 0.0  # balance loss disabled in v2
    router_z_loss_weight: float = 0.0  # z-loss disabled by default in v2

    # Expert liveness (v2.1): prevent dead ReLU fixed point in routing
    router_jitter_noise: float = 0.1   # std of Gaussian noise on gate logits during training
    liveness_min_frac: float = 0.05    # minimum expert activation fraction (5%)
    liveness_loss_weight: float = 0.1  # weight for liveness penalty

    # SSM (Mamba-2 SSD blocks)
    ssm_d_state: int = 16
    ssm_dt_rank: int = 16

    # Shared expert FFN width (always-on, 4x d_model)
    d_ff_shared: int = 1024

    # Sequence
    context_len: int = 512

    # Training
    gradient_checkpointing: bool = False

    @property
    def n_type_b_computed(self) -> int:
        return self.n_experts - self.n_type_a

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def embedding_params(self) -> int:
        """Factored embedding parameter count."""
        return self.vocab_size * self.embedding_inner_dim + self.embedding_inner_dim * self.d_model

    @property
    def full_embedding_params(self) -> int:
        """Non-factored embedding parameter count (for comparison)."""
        return self.vocab_size * self.d_model

    def estimate_params(self) -> dict:
        """Estimate total parameter count by component."""
        d = self.d_model
        d_ff_routed = self.d_ff  # routed expert FFN width (2x d_model in v2 tiny)

        # Factored embeddings (shared encoder/decoder, weight-tied with LM head)
        embed = self.embedding_params

        # Encoder layers: self-attn + FFN + 2 layer norms
        enc_attn = d * d * 4  # Q, K, V, O projections
        enc_ffn = d * self.d_ff_shared * 2  # encoder uses full-width FFN
        enc_norm = d * 4  # 2 norms × (weight + bias)
        enc_total = (enc_attn + enc_ffn + enc_norm) * self.n_enc_layers

        # Decoder MoE layers (v2: Hymba fusion + shared expert + routed experts)

        # SSM (Mamba-2 SSD block) per layer
        # in_proj: d * (d + d + 2*d_state*n_heads + dt_rank)
        n_heads = self.n_heads
        d_state = self.ssm_d_state
        dt_rank = self.ssm_dt_rank
        ssm_in = d * (d + d + 2 * d_state * n_heads + dt_rank)
        ssm_dt = dt_rank * d  # dt_proj
        ssm_out = d * d  # out_proj
        ssm_misc = n_heads + n_heads + d  # A_log + D + RMSNorm
        ssm_per_layer = ssm_in + ssm_dt + ssm_out + ssm_misc

        # Causal self-attention per layer
        dec_self_attn = d * d * 4  # Q, K, V, O

        # Hymba norm + learnable gates
        hymba_norm = d * 2  # LayerNorm weight + bias
        hymba_gates = d * 2  # beta_ssm + beta_attn

        # Shared expert (always-on, 4x FFN)
        shared_expert = (d * self.d_ff_shared * 2  # up + down projections
                         + d * 2)  # LayerNorm

        # Routed Type A expert: cross-attn + FFN + 2 norms
        type_a_cross_attn = d * d * 4
        type_a_ffn = d * d_ff_routed * 2
        type_a_norm = d * 4
        type_a_per_expert = type_a_cross_attn + type_a_ffn + type_a_norm

        # Routed Type B expert: FFN + 1 norm
        type_b_ffn = d * d_ff_routed * 2
        type_b_norm = d * 2
        type_b_per_expert = type_b_ffn + type_b_norm

        # Router per layer
        enc_signal_dim = max(d // 4, 1)
        router_per_layer = (1 * enc_signal_dim  # enc_signal_proj
                            + enc_signal_dim * 2  # enc_signal_norm
                            + (d + enc_signal_dim) * self.n_experts  # gate (with bias)
                            + self.n_experts)  # gate bias

        # Total per decoder layer
        dec_per_layer = (
            ssm_per_layer + dec_self_attn + hymba_norm + hymba_gates +
            shared_expert +
            type_a_per_expert * self.n_type_a +
            type_b_per_expert * self.n_type_b_computed +
            router_per_layer
        )
        dec_total = dec_per_layer * self.n_dec_layers

        # Final layer norm
        final_norm = d * 2

        # LM head is weight-tied with embedding, so not counted
        total = embed + enc_total + dec_total + final_norm

        return {
            'embedding': embed,
            'encoder': enc_total,
            'decoder': dec_total,
            'decoder_per_layer': dec_per_layer,
            'ssm_per_layer': ssm_per_layer,
            'shared_expert_per_layer': shared_expert,
            'final_norm': final_norm,
            'total': total,
            'total_M': total / 1e6,
        }

    @classmethod
    def from_tokenizer(cls, tokenizer_dir: str, preset: str = 'tiny') -> 'ModelConfig':
        """Load vocab_size from tokenizer config and apply a preset."""
        # Prefer extended config with author tokens; fall back to base
        config_path = Path(tokenizer_dir) / 'config_with_authors.json'
        if not config_path.exists():
            config_path = Path(tokenizer_dir) / 'config.json'
        with open(config_path) as f:
            tok_config = json.load(f)

        if preset == 'tiny':
            cfg = TinyConfig()
        elif preset == 'small':
            cfg = SmallConfig()
        else:
            cfg = cls()

        cfg.vocab_size = tok_config['vocab_size']
        cfg.embedding_inner_dim = tok_config.get('embedding_inner_dim', 64)
        return cfg


def TinyConfig(**overrides) -> ModelConfig:
    """~13M params. Mac M-series trainable. v2: Hymba SSM+Attn, ReLU routing, shared expert."""
    defaults = dict(
        d_model=256,
        d_ff=512,             # routed expert FFN (2x d_model, was 1024)
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=6,       # was 4
        n_experts=2,
        n_type_a=1,
        top_k=1,              # vestigial
        context_len=512,
        dropout=0.1,
        gradient_checkpointing=False,
        d_ff_shared=1024,     # shared expert (4x d_model)
        ssm_d_state=16,
        ssm_dt_rank=16,
        aux_loss_weight=0.0,          # balance loss disabled
        router_z_loss_weight=0.0,     # z-loss disabled by default
        router_jitter_noise=0.1,      # prevent dead ReLU in routing
        liveness_min_frac=0.05,       # 5% minimum expert activation
        liveness_loss_weight=0.1,     # liveness penalty weight
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def SmallConfig(**overrides) -> ModelConfig:
    """125-250M params. Colab A100 target. The real experiment.

    Budget constraint: 8 experts × full FFN × 12 layers = 600M+.
    Use 4 experts (1A+3B) with d_ff=2048 and 8 decoder layers to hit ~160M.
    Scale up experts if compute budget allows.
    """
    defaults = dict(
        d_model=768,
        d_ff=2048,
        n_heads=12,
        n_enc_layers=4,
        n_dec_layers=8,
        n_experts=4,
        n_type_a=1,
        top_k=2,
        context_len=1024,
        dropout=0.1,
        gradient_checkpointing=True,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


if __name__ == '__main__':
    for name, cfg_fn in [('Tiny', TinyConfig), ('Small', SmallConfig)]:
        cfg = cfg_fn()
        params = cfg.estimate_params()
        print(f"\n{name} Config:")
        print(f"  d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
              f"enc_layers={cfg.n_enc_layers}, dec_layers={cfg.n_dec_layers}")
        print(f"  experts={cfg.n_experts} (A={cfg.n_type_a}, B={cfg.n_type_b_computed}), "
              f"top_k={cfg.top_k}")
        print(f"  context_len={cfg.context_len}")
        print(f"  Parameter breakdown:")
        for k, v in params.items():
            if k == 'total_M':
                print(f"    {k}: {v:.2f}M")
            else:
                print(f"    {k}: {v:,}")
        print(f"  Embedding savings: {(1 - cfg.embedding_params / cfg.full_embedding_params) * 100:.1f}%")
