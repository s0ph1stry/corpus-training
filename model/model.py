"""
HeteroMoETransformer: the full model.

shared factored embeddings → encoder (when needed) → MoE decoder stack → LM head

The encoder runs only on samples with encoder_available=True.
Type A experts for encoder_available=False samples get zeroed cross-attention
K/V — the router should learn to avoid routing there.

Weight tying: LM head shares weights with the embedding projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.encoder import Encoder
from model.moe_layer import MoELayer
from model.experts import MOEManager
from model.expert_aux_heads import ExpertAuxHead


class FactoredEmbedding(nn.Module):
    """Factored token embedding: vocab × inner_dim + inner_dim × d_model.

    Saves ~74% of embedding parameters at 16K vocab, inner_dim=64, d_model=256.
    """

    def __init__(self, vocab_size: int, inner_dim: int, d_model: int,
                 pad_token_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, inner_dim, padding_idx=pad_token_id)
        self.projection = nn.Linear(inner_dim, d_model, bias=False)
        self.inner_dim = inner_dim
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) -> (batch, seq_len, d_model)"""
        return self.projection(self.embedding(token_ids))


class HeteroMoETransformer(nn.Module):
    """Heterogeneous Mixture-of-Experts Transformer.

    Two expert types mirror the corpus's own structure:
      - Type A (enc-dec): reflection, comparison, reconstruction
      - Type B (dec-only): generation from internal knowledge

    The router learns when to reflect and when to act.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.moe_manager = MOEManager()
        self.global_step = 0  # updated by trainer

        # Shared factored embedding
        self.embedding = FactoredEmbedding(
            config.vocab_size, config.embedding_inner_dim,
            config.d_model, config.pad_token_id
        )

        # Encoder (bidirectional, shallow)
        self.encoder = Encoder(config)

        # Decoder MoE stack
        self.decoder_layers = nn.ModuleList([
            MoELayer(config, self.moe_manager, layer_idx=i)
            for i in range(config.n_dec_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head — reverse of factored embedding path
        # inner_dim -> vocab (weight-tied with embedding table)
        self.lm_head_vocab = nn.Linear(config.embedding_inner_dim, config.vocab_size, bias=False)
        self.lm_head_vocab.weight = self.embedding.embedding.weight
        # d_model -> inner_dim: uses transposed embedding.projection in forward()

        # Per-expert auxiliary heads (v2.2): shared across layers
        self.type_a_aux_head = None
        self.type_b_aux_head = None
        if config.expert_aux_heads:
            self.type_a_aux_head = ExpertAuxHead(config.d_model, config.embedding_inner_dim)
            self.type_b_aux_head = ExpertAuxHead(config.d_model, config.embedding_inner_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Wire up aux heads AFTER init (needs weight-tied vocab proj)
        if config.expert_aux_heads:
            self.type_a_aux_head.set_vocab_proj(self.lm_head_vocab)
            self.type_b_aux_head.set_vocab_proj(self.lm_head_vocab)
            for layer in self.decoder_layers:
                layer.set_aux_heads(self.type_a_aux_head, self.type_b_aux_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self,
                decoder_input_ids: torch.Tensor,
                encoder_input_ids: torch.Tensor = None,
                encoder_available: torch.Tensor = None,
                decoder_padding_mask: torch.Tensor = None,
                encoder_padding_mask: torch.Tensor = None,
                decoder_targets: torch.Tensor = None,
                mode_ids: torch.Tensor = None) -> dict:
        """
        decoder_input_ids: (batch, dec_seq_len) — token IDs for decoder
        encoder_input_ids: (batch, enc_seq_len) — token IDs for encoder (can be None)
        encoder_available: (batch,) — binary, 1 if encoder input exists for this sample
        decoder_padding_mask: (batch, dec_seq_len) — True=valid
        encoder_padding_mask: (batch, enc_seq_len) — True=valid
        decoder_targets: (batch, dec_seq_len) — target tokens for expert aux heads
        mode_ids: (batch,) — UL2 mode IDs (R=0, S=1, X=2) for mode-conditioned routing

        Returns dict with:
            logits: (batch, dec_seq_len, vocab_size)
            aux_loss: scalar tensor
        """
        self.moe_manager.reset()

        B = decoder_input_ids.shape[0]
        device = decoder_input_ids.device

        # Default masks
        if encoder_available is None:
            encoder_available = torch.zeros(B, device=device)

        if decoder_padding_mask is None:
            decoder_padding_mask = decoder_input_ids != self.config.pad_token_id

        # Embed decoder input
        dec_hidden = self.embedding(decoder_input_ids)  # (B, dec_S, D)

        # Run encoder on samples that have encoder input
        encoder_out = None
        encoder_mask = None

        if encoder_input_ids is not None and encoder_available.any():
            if encoder_available.all():
                # Pure denoising batch — run encoder on all samples directly
                if encoder_padding_mask is not None:
                    enc_pad_mask = encoder_padding_mask
                else:
                    enc_pad_mask = encoder_input_ids != self.config.pad_token_id

                enc_embedded = self.embedding(encoder_input_ids)
                encoder_out = self.encoder(enc_embedded, enc_pad_mask)
                encoder_mask = enc_pad_mask
            else:
                # Mixed batch — only run encoder on samples that need it
                enc_mask = encoder_available.bool()
                enc_input = encoder_input_ids[enc_mask]

                if encoder_padding_mask is not None:
                    enc_pad_mask = encoder_padding_mask[enc_mask]
                else:
                    enc_pad_mask = enc_input != self.config.pad_token_id

                enc_embedded = self.embedding(enc_input)
                enc_hidden = self.encoder(enc_embedded, enc_pad_mask)

                # Build full-batch encoder output (zeros for samples without encoder)
                enc_S = encoder_input_ids.shape[1]
                encoder_out = torch.zeros(B, enc_S, self.config.d_model, device=device)
                encoder_mask = torch.zeros(B, enc_S, dtype=torch.bool, device=device)

                encoder_out[enc_mask] = enc_hidden
                encoder_mask[enc_mask] = enc_pad_mask

        # Run decoder MoE stack
        for layer in self.decoder_layers:
            dec_hidden = layer(
                dec_hidden,
                encoder_out=encoder_out,
                encoder_mask=encoder_mask,
                padding_mask=decoder_padding_mask,
                encoder_available=encoder_available,
                decoder_targets=decoder_targets,
                mode_ids=mode_ids,
            )

        # Final norm + LM head
        dec_hidden = self.final_norm(dec_hidden)
        # d_model -> inner_dim via transposed embedding projection (weight tying)
        # projection.weight is (d_model, inner_dim), transpose to (inner_dim, d_model) for F.linear
        inner = F.linear(dec_hidden, self.embedding.projection.weight.t())
        logits = self.lm_head_vocab(inner)

        # Aux loss from all MoE layers
        aux_loss = self.moe_manager.get_aux_loss(
            global_step=self.global_step,
            z_weight=self.config.router_z_loss_weight,
            liveness_weight=self.config.liveness_loss_weight,
            expert_aux_weight=self.config.expert_aux_weight if self.config.expert_aux_heads else 0.0,
            similarity_weight=self.config.similarity_loss_weight,
            rcl_weight=self.config.rcl_weight,
        )

        return {
            'logits': logits,
            'aux_loss': aux_loss,
        }

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['embedding'] = sum(p.numel() for p in self.embedding.parameters())
        counts['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        counts['decoder'] = sum(p.numel() for p in self.decoder_layers.parameters())
        counts['final_norm'] = sum(p.numel() for p in self.final_norm.parameters())
        # LM head is weight-tied, so not counted separately
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['total_M'] = counts['total'] / 1e6
        return counts

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'HeteroMoETransformer':
        return cls(config)
