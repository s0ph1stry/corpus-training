"""
Loss functions for the heterogeneous MoE.

Phase 1: reconstruction loss at masked positions
Phase 2: standard next-token cross-entropy
Mixed: weighted combination for Phase 2 batches with denoising fraction
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor,
                        pad_id: int = 0, ignore_index: int = -100) -> torch.Tensor:
    """
    Cross-entropy at reconstruction positions only (Phase 1).

    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) — pad_id or ignore_index at non-target positions

    Returns scalar loss.
    """
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)

    # Mask: only compute loss where target is meaningful
    valid = (targets_flat != pad_id) & (targets_flat != ignore_index)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index,
                           reduction='none')

    # Also ignore pad positions
    loss = loss * valid.float()
    return loss.sum() / valid.float().sum()


def generative_loss(logits: torch.Tensor, targets: torch.Tensor,
                    ignore_index: int = -100) -> torch.Tensor:
    """
    Standard next-token cross-entropy (Phase 2).

    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) — shifted by 1 relative to input

    Returns scalar loss.
    """
    B, S, V = logits.shape
    return F.cross_entropy(
        logits.view(-1, V), targets.view(-1),
        ignore_index=ignore_index,
    )


def phase2_mixed_loss(logits: torch.Tensor, targets: torch.Tensor,
                      encoder_available: torch.Tensor,
                      pad_id: int = 0,
                      ignore_index: int = -100) -> dict:
    """
    Weighted loss for mixed denoising + generative batches (Phase 2).

    Computes separate losses for denoising and generative samples,
    returns both plus the combined loss.

    encoder_available: (batch,) — 1.0 for denoising samples, 0.0 for generative
    """
    B, S, V = logits.shape
    enc_mask = encoder_available > 0.5
    gen_mask = ~enc_mask

    result = {}

    # Denoising loss (reconstruction)
    if enc_mask.any():
        enc_logits = logits[enc_mask]
        enc_targets = targets[enc_mask]
        result['denoise_loss'] = reconstruction_loss(
            enc_logits, enc_targets, pad_id, ignore_index
        )
    else:
        result['denoise_loss'] = torch.tensor(0.0, device=logits.device)

    # Generative loss
    if gen_mask.any():
        gen_logits = logits[gen_mask]
        gen_targets = targets[gen_mask]
        result['gen_loss'] = generative_loss(gen_logits, gen_targets, ignore_index)
    else:
        result['gen_loss'] = torch.tensor(0.0, device=logits.device)

    # Combined: weight by proportion present
    n_enc = enc_mask.sum().float()
    n_gen = gen_mask.sum().float()
    total = n_enc + n_gen

    if total > 0:
        result['loss'] = (n_enc / total * result['denoise_loss'] +
                          n_gen / total * result['gen_loss'])
    else:
        result['loss'] = torch.tensor(0.0, device=logits.device)

    return result


def compute_per_text_loss(logits: torch.Tensor, targets: torch.Tensor,
                          text_names: list,
                          ignore_index: int = -100) -> dict:
    """Compute per-text loss for online difficulty adjustment.

    Returns dict of text_name -> mean loss for that text's samples.
    """
    B, S, V = logits.shape
    per_sample_loss = F.cross_entropy(
        logits.view(-1, V), targets.view(-1),
        ignore_index=ignore_index,
        reduction='none',
    ).view(B, S)

    # Mean loss per sample (ignoring padding)
    valid = targets != ignore_index
    per_sample_mean = (per_sample_loss * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1)

    result = {}
    counts = {}
    for i, name in enumerate(text_names):
        if name:
            loss_val = per_sample_mean[i].item()
            if name in result:
                result[name] += loss_val
                counts[name] += 1
            else:
                result[name] = loss_val
                counts[name] = 1

    # Compute proper mean
    for name in result:
        result[name] /= counts[name]

    return result
