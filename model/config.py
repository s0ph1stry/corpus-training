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
    n_experts: int = 2  # total experts per MoE layer
    n_type_a: int = 1  # encoder-decoder experts (n_type_b = n_experts - n_type_a)
    top_k: int = 1
    capacity_factor_train: float = 1.25
    capacity_factor_eval: float = 2.0
    aux_loss_weight: float = 0.01  # load balancing
    router_z_loss_weight: float = 0.001

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
        # Factored embeddings (shared encoder/decoder, weight-tied with LM head)
        embed = self.embedding_params

        # Encoder layers: self-attn + FFN + 2 layer norms
        enc_attn = self.d_model * self.d_model * 4  # Q, K, V, O projections
        enc_ffn = self.d_model * self.d_ff * 2  # up + down
        enc_norm = self.d_model * 4  # 2 norms × (weight + bias)
        enc_total = (enc_attn + enc_ffn + enc_norm) * self.n_enc_layers

        # Decoder MoE layers
        # Self-attention (shared across experts in a layer)
        dec_self_attn = self.d_model * self.d_model * 4
        dec_self_norm = self.d_model * 2

        # Type A expert: cross-attn + FFN + 2 norms
        type_a_cross_attn = self.d_model * self.d_model * 4  # Q, K, V, O
        type_a_ffn = self.d_model * self.d_ff * 2
        type_a_norm = self.d_model * 4  # cross-attn norm + FFN norm
        type_a_per_expert = type_a_cross_attn + type_a_ffn + type_a_norm

        # Type B expert: FFN + 1 norm
        type_b_ffn = self.d_model * self.d_ff * 2
        type_b_norm = self.d_model * 2
        type_b_per_expert = type_b_ffn + type_b_norm

        # Router per layer: (d_model + 1) × n_experts
        router_per_layer = (self.d_model + 1) * self.n_experts

        # Total per decoder layer
        dec_per_layer = (
            dec_self_attn + dec_self_norm +
            type_a_per_expert * self.n_type_a +
            type_b_per_expert * self.n_type_b_computed +
            router_per_layer
        )
        dec_total = dec_per_layer * self.n_dec_layers

        # Final layer norm
        final_norm = self.d_model * 2

        # LM head is weight-tied with embedding, so not counted
        total = embed + enc_total + dec_total + final_norm

        return {
            'embedding': embed,
            'encoder': enc_total,
            'decoder': dec_total,
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
    """4-8M params. Mac M-series trainable. For architecture validation."""
    defaults = dict(
        d_model=256,
        d_ff=1024,
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=4,
        n_experts=2,
        n_type_a=1,
        top_k=1,
        context_len=512,
        dropout=0.1,
        gradient_checkpointing=False,
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
