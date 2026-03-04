# Architecture Design Document
## Heterogeneous Mixture-of-Experts Transformer for Coherence Training

**Status**: Living design document — last updated 2026-03-04
**Project**: Corpus training on 206 cleaned texts (38.7M tokens, 147MB) selected for internal coherence
**Corpus spec**: `corpus.md` | **Voices**: `voices.md` | **Topics**: `topics.md` | **Exercises**: `training-exercises.md`
**Implementation**: Complete. All infrastructure written and smoke-tested (model/, data/, training/, eval/, control/).
**Rubric scores**: `rubric_scores.csv` — all 206 texts scored on 4 dimensions (1-5)

---

## Table of Contents

1. [Core Hypothesis](#1-core-hypothesis)
2. [Architecture Overview](#2-architecture-overview)
3. [Components in Detail](#3-components-in-detail)
4. [Scale Specifications](#4-scale-specifications)
5. [Training Pipeline](#5-training-pipeline)
6. [Reality Oracle War](#6-reality-oracle-war)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Implementation Notes](#8-implementation-notes)
9. [Key References](#9-key-references)
10. [Open Questions and Risks](#10-open-questions-and-risks)

---

## 1. Core Hypothesis

Standard language models are trained on web text: a distribution in which fluency, factual recall, and stylistic mimicry are heavily represented, and in which structural coherence — the property that every part is necessary and the constraint is intrinsic — is diluted by enormous amounts of competent but careless writing.

The hypothesis: train a model entirely on a curated corpus selected by a single criterion (every part necessary, structure load-bearing, constraint intrinsic), and see what emerges. Can coherence be a trained property, not just a recognized one? Can a model learn to *generate* care rather than just *reproduce* surface patterns of care?

The architecture is designed to make this question answerable. The heterogeneous MoE structure gives the model two distinct processing modes that mirror a natural distinction in the corpus: texts that work by reference (understanding against a source) and texts that work by internal generation (pure origination). The architecture is the hypothesis in hardware.

---

## 2. Architecture Overview

### High-Level Data Flow

```
Input text
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER STACK                            │
│  Bidirectional self-attention (full context)                │
│  Layers: 4-6 (Tiny) / 12 (Small)                           │
│  Produces: key-value representation for cross-attention     │
└────────────────────────┬────────────────────────────────────┘
                         │  K, V for cross-attention
                         ▼
Token sequence ──► [Embeddings + RoPE] ──► MoE Layers ──► Output logits
                                               │
                         ┌─────────────────────┘
                         ▼
              ┌──────────────────────┐
              │       ROUTER         │
              │  Input: hidden state │
              │  + encoder_available │
              │  Output: expert idx  │
              └──────┬────────┬──────┘
                     │        │
          ┌──────────┘        └──────────┐
          ▼                              ▼
  ┌────────────────┐            ┌─────────────────┐
  │  EXPERT TYPE A │            │  EXPERT TYPE B  │
  │  Encoder-Decoder│           │  Decoder-Only   │
  │                │            │                 │
  │  Self-Attn     │            │  Causal         │
  │  (causal mask) │            │  Self-Attn      │
  │  Cross-Attn    │            │                 │
  │  (to encoder)  │            │  FFN            │
  │  FFN           │            │                 │
  └────────────────┘            └─────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
                   Weighted sum
                (top-k routing output)
```

### Expert Selection Logic

The router makes a routing decision per token per layer. The critical input signal is `encoder_available`: a learned or explicit flag indicating whether encoder representations exist for the current forward pass. When encoder representations are available (denoising, reconstruction, self-critique tasks), the model biases toward Type A experts. When running autoregressively without encoder input, it biases toward Type B.

This is not hard-gating — the router learns soft preferences weighted by `encoder_available` and the current hidden state. During pretraining these preferences are implicit; they become explicit behavioral patterns as the training regime cycles through both modes.

---

## 3. Components in Detail

### 3.1 Shared Embeddings

- **Token embedding table**: Vocabulary size V → embedding dim d_model. Shared between encoder and decoder (weight tying).
- **Positional encoding**: Rotary Position Embedding (RoPE). Applied at the QK level for both self-attention and cross-attention. RoPE is chosen over learned absolute positions for better length generalization — the corpus spans texts from Homer to Shannon and context lengths vary substantially.
- **Layer normalization**: Pre-norm architecture (layer norm before attention/FFN, not after). More stable training at smaller scales.

### 3.2 Encoder Stack

The encoder is a standard bidirectional transformer stack. No causal mask — all tokens attend to all tokens in both directions.

```
Encoder Layer (×N_enc):
  ┌─────────────────────────────────┐
  │  LayerNorm                      │
  │  Bidirectional Multi-Head Attn  │
  │  Residual                       │
  │  LayerNorm                      │
  │  FFN (2-layer, GELU)            │
  │  Residual                       │
  └─────────────────────────────────┘
```

The encoder stack is shallower than the decoder: roughly 1/3 the total layer count. Its job is to produce rich key-value representations, not to perform the main generative work.

- **Output**: Hidden states at every position → used as K and V for cross-attention in Type A expert blocks.
- **Encoder is optional at inference time**: When encoder_available = False, the encoder stack is simply not run. This is the decoder-only operating mode.

### 3.3 Expert Type A: Encoder-Decoder Block

For tasks where the model has a source text to work from: reconstruction, truth-checking, comparison, self-critique.

```
Type A Expert Block:
  ┌─────────────────────────────────────────┐
  │  LayerNorm                              │
  │  Causal Multi-Head Self-Attention       │  ← attends to previous decoder tokens
  │  Residual                               │
  │  LayerNorm                              │
  │  Multi-Head Cross-Attention             │  ← Q: decoder hidden, K/V: encoder output
  │  Residual                               │
  │  LayerNorm                              │
  │  FFN (2-layer, GELU or SwiGLU)         │
  │  Residual                               │
  └─────────────────────────────────────────┘
```

Cross-attention parameters: Q projected from decoder hidden state; K, V projected from encoder output. Attention heads split evenly between self-attention and cross-attention in parameter budget at smaller scales; separate projection matrices at larger scales.

### 3.4 Expert Type B: Decoder-Only Block

For pure generation tasks where no source text exists: autoregressive next-token prediction, prefix continuation, generative exercises.

```
Type B Expert Block:
  ┌─────────────────────────────────────────┐
  │  LayerNorm                              │
  │  Causal Multi-Head Self-Attention       │  ← causal mask, no cross-attention
  │  Residual                               │
  │  LayerNorm                              │
  │  FFN (2-layer, GELU or SwiGLU)         │
  │  Residual                               │
  └─────────────────────────────────────────┘
```

Type B is architecturally simpler and computationally cheaper than Type A. At small scales this makes load balancing easier — the router can favor Type B for simple generation tasks without the parameter cost being prohibitive.

### 3.5 Router

The router is a learned module applied per-token before each MoE layer.

**Input**:
- Current token hidden state `h` (dim: d_model)
- `encoder_available` signal (binary or soft scalar, broadcast to match h dim)

**Architecture**:
```
router_input = concat([h, encoder_available_embedding])
logits = Linear(router_input, n_experts)
routing_weights = softmax(logits)
selected_experts = top_k(routing_weights, k=1 or 2)
output = weighted_sum([expert_i(h) for i in selected_experts])
```

**Load balancing loss** (following Switch Transformer):
```
L_balance = α × n_experts × Σ_i (f_i × p_i)
```
Where:
- `f_i` = fraction of tokens dispatched to expert i in the batch
- `p_i` = mean routing probability for expert i across the batch
- `α` = auxiliary loss weight (typical: 0.01)

This prevents mode collapse where all tokens route to one expert. With only 2 expert types (one of each at Tiny scale), this is especially important — the router must learn a meaningful distinction, not just always pick Type A or Type B.

### 3.6 Final Projection Head

`Linear(d_model → vocab_size)` with weight tying to the token embedding table. Applied to the output of the final MoE layer.

---

## 4. Scale Specifications

### Tiny (~8.9M parameters) — IMPLEMENTED

Target: trainable on Mac CPU or MPS (Apple Silicon). Primary scale for rapid iteration and architectural validation.

| Component | Spec |
|-----------|------|
| Layers (decoder) | 4 |
| Layers (encoder) | 2 |
| Attention heads | 4 |
| Embedding dim (d_model) | 256 |
| FFN dim | 1024 |
| Embedding | Factored: vocab×64 + 64×256 (saves ~74%) |
| Experts per layer | 2 (1 Type A, 1 Type B) |
| Routing | Top-1 |
| Vocabulary | 16,215 (16K base + 214 author tokens + `<complete>`) |
| Context length | 512 tokens |
| Estimated params | ~8.9M |
| Training hardware | Mac M-series, CPU or MPS |
| Batch size | 8–16 sequences |

**Chinchilla note**: At 38.7M tokens, the Tiny model gets ~4.3 tokens per parameter — well below the 20:1 Chinchilla ratio, meaning it will be meaningfully overtrained. This is acceptable: the model's entire distribution becomes the corpus.

**Smoke test results (2026-03-04)**: Loss starts at ~9.68 (≈ ln(16215), uniform prediction) and ticks down over 10 steps. Gradients flowing, aux loss stable (0.010-0.011), no NaN. Architecture works end-to-end.

### Small (~162M parameters) — IMPLEMENTED

Target: Colab A100. Primary scale for capability evaluation and the experiments that answer the research question.

| Component | Spec |
|-----------|------|
| Layers (decoder) | 8 |
| Layers (encoder) | 4 |
| Attention heads | 12 |
| Embedding dim (d_model) | 768 |
| FFN dim | 2048 |
| Embedding | Factored: vocab×64 + 64×768 |
| Experts per layer | 4 (1 Type A, 3 Type B) |
| Routing | Top-2 |
| Vocabulary | 16,215 |
| Context length | 1024 tokens |
| Estimated params | ~162M |
| Training hardware | Colab A100 (40GB) |
| Batch size | 64–128 sequences |
| Gradient checkpointing | Required |

**Expert ratio**: 1A + 3B per layer (previously planned as 2/3 A, now inverted). With top-2 routing, each token activates 2 of 4 experts — the router learns when to include the Type A (reflection) expert. This gives more generation capacity while preserving the reflection pathway.

**Budget note**: Originally spec'd at ~596M (8 experts × 3072 FFN × 12 layers). Revised down to 162M to fit A100 40GB with gradient checkpointing: 4 experts, d_ff=2048, 8 decoder layers.

### Dimension Summary

```
Tiny (8.9M):
  Embedding:  vocab(16215) × 64 → 64 × 256
  Encoder:    [seq_len, 256] → [seq_len, 256]   (K, V for cross-attn)
  Decoder:    [seq_len, 256] → [seq_len, 256]
  Attention:  head_dim = 256/4 = 64
  FFN:        256 → 1024 → 256
  LM Head:    256 → 64 → 16215 (weight-tied with embedding table)

Small (162M):
  Embedding:  vocab(16215) × 64 → 64 × 768
  Encoder:    [seq_len, 768] → [seq_len, 768]
  Decoder:    [seq_len, 768] → [seq_len, 768]
  Attention:  head_dim = 768/12 = 64
  FFN:        768 → 2048 → 768
  LM Head:    768 → 64 → 16215 (weight-tied)
```

---

## 5. Training Pipeline

The training pipeline has five phases. Phases 1 and 2 run concurrently in their later stages. Phases 3–5 are sequential and build on each other.

```
Phase 1: Denoising Pretraining ──────────────────┐
                                                   │ (concurrent late Phase 1 + early Phase 2)
Phase 2: Generative Capacity ────────────────────┘
          │
          ▼
Phase 3: Socratic SFT
          │
          ▼
Phase 4: Self-Constitution
          │
          ▼
Phase 5: Iterative ORPO
```

### Phase 1: Denoising Pretraining

**Goal**: Train the model to reconstruct original text from corrupted versions. Establishes the encoder-decoder pathway and teaches the model the structure of corpus text through reconstruction.

**Corruption strategies** (mixed throughout training):

| Strategy | Description | Rate |
|----------|-------------|------|
| Span masking (T5-style) | Replace contiguous spans with single mask token | 15–30% of tokens |
| Sentence shuffling | Reorder sentences within a passage | Applied to full passages |
| Span deletion | Remove spans entirely (harder: no mask signal) | 10–20% of tokens |
| Text rotation | Move suffix to prefix (rotation point is random) | Applied to full passages |
| Semantic corruption | Replace spans with plausible-but-wrong content | 5–15% of tokens |

**Semantic corruption** deserves emphasis: unlike masking (which removes information), semantic corruption *replaces* with wrong information. A paraphrase that changes the argument's structure; a factual claim that's almost true; a sentence that sounds right but breaks the internal logic. The model must detect that the substituted content is wrong despite being grammatically valid. This is the "Psychedelic Span Corruption" mechanism from `trainingideas.csv`.

**Progressive corruption schedule**:
```
Step 0        → 15% corruption rate
Step T/4      → 30% corruption rate
Step T/2      → 50% corruption rate
Step 3T/4     → 65% corruption rate
Step T        → 80% corruption rate
```

Rationale: at high corruption rates, the decoder must do more of the generative work from encoder representations. This prevents the model from learning a shortcut where it reconstructs by lightly editing a mostly-intact input.

**Loss**: Cross-entropy on reconstructed tokens. Mask positions only (T5 style) in early training; full sequence reconstruction in later stages.

**Router behavior in Phase 1**: encoder_available = True for all batches. The router should learn to heavily favor Type A experts. This is correct — encoder-decoder is the right mode for reconstruction tasks.

### Phase 2: Generative Capacity Training

**Goal**: Train the decoder-only pathway. The model learns to generate fluently from the corpus distribution without encoder input.

**Tasks** (mixed batches, ratio shifts over time):

| Task | Description | Phase 2 Start Ratio | Phase 2 End Ratio |
|------|-------------|---------------------|-------------------|
| Next-token prediction | Standard autoregressive LM on corpus | 40% | 60% |
| Prefix continuation | Variable split: 10-90% prefix given, model continues | 40% | 30% |
| Denoising (from Phase 1) | Continued, at high corruption rate | 20% | 10% |

**Router behavior in Phase 2**: encoder_available = False for autoregressive batches. Router learns to activate Type B experts. Mixed batches (some denoising, some autoregressive) teach the router to switch modes correctly based on the encoder_available signal.

**The reconstruction-to-generation bridge**: This is an open design question. Options under consideration:

1. **Progressive corruption**: Increase corruption so high that the decoder effectively generates from almost nothing. At 80% corruption, reconstruction looks like generation. Advantage: continuous transition. Disadvantage: the objective is still reconstruction, not generation — the model optimizes for what's in the source, not for internal coherence.

2. **Mixed objectives**: Phase 2 as described — simultaneously training both pathways. The model learns to "turn off" cross-attention when encoder representations aren't available. This requires the router to learn mode-switching, which takes training time.

3. **Encoder-as-internal-critic**: Decoder generates a draft; encoder reads the draft; decoder revises. This is Mode 3 of the Reality Oracle War (see Section 6). May be most appropriate as a Phase 3+ technique after the basic pathways are established.

**Current plan**: Mixed objectives (option 2), with encoder-as-critic phased in during Phase 3.

### Phase 3: Socratic SFT

**Goal**: The model articulates its understanding of coherence by interrogating corpus passages. This is where the model moves from recognizing structure to analyzing it.

**Setup**: Encoder reads corpus passages; decoder produces interrogation responses.

**Task types** (with training examples generated by Claude on corpus material):

| Task | Example prompt | What it trains |
|------|----------------|----------------|
| Weakest point | "What is the weakest point in this passage?" | Discriminating load-bearing from optional |
| Structural necessity | "Why does this sentence follow the previous one?" | Understanding internal constraint |
| Removal test | "What would be lost if this paragraph were removed?" | Negative Space exercise |
| Hidden premise | "What premise must be true for this argument to hold?" | Reconstructing the implicit |
| Wrong version | "Which version is original and why?" | Coherence discrimination |
| Earned conclusions | "Is this conclusion earned or asserted?" | Structural earning |
| Form-content | "Does the form of this text enact or contradict its content?" | Thematic justice |
| Register check | "What would this look like translated into mathematical notation?" | Cross-register coherence |
| Temporal structure | "Why does this begin where it begins?" | Temporal origami |
| Subtext | "What is this conversation actually about?" | Multi-layer reading |

**SFT data generation**: Claude interrogates corpus passages using the task types above. Output is curated by the selection criterion: is the interrogation itself internally coherent? Does every sentence of the interrogation do work? The SFT data must satisfy the same criterion as the training corpus.

**Exercise-to-phase cross-reference**: Most exercises in `training-exercises.md` map to Phase 3. The exercise table at the bottom of that document provides full mappings.

**Phase 3 also trains**:
- Reality Oracle War Mode 3 (self-critique via encoder-on-own-output)
- Argument Against Yourself
- Staying in the Question
- Cross-Cultural Structural Recognition

### Phase 4: Self-Constitution

**Goal**: The model generates coherence principles from its best Phase 3 outputs. These principles become the evaluative framework for Phase 5 preference optimization. The constitution should capture what the corpus taught, not what was prescribed from outside.

**Process**:

1. Select the highest-quality Phase 3 outputs (by human review + automated discrimination scores)
2. Prompt the model: "Based on your analyses of these passages, what principles of coherence emerge? What do all the strongest texts have in common, structurally?"
3. Model generates candidate constitution (a set of evaluative principles)
4. Human review: refine, extend, remove principles that are too vague or merely surface-level
5. Final constitution is the evaluative framework for Phase 5

**Expected constitution contents** (hypothesis — the model should find these, not be told them):
- Every part necessary: no element is present for decoration alone
- Structural earning: claims are not asserted but demonstrated through what precedes them
- Form enacts content: the structure of the text is consistent with what it says
- Constraint is intrinsic: the rules the text follows emerge from its material, not from convention
- Load-bearing absence: what is not said does structural work
- Coherence at multiple levels: the argument holds both locally (sentence) and globally (work)

**Why self-constitution matters**: The arXiv 2510.26707 paper ("Value Drifts") finds that SFT establishes values and preference optimization barely moves them. The constitution should therefore be established *before* ORPO, and it should be generated from the model's own understanding rather than externally prescribed. The self-generated constitution encodes what the model actually learned from the corpus — a more faithful training target than a pre-written rubric.

### Phase 5: Iterative ORPO

**Goal**: Preference alignment using ORPO (Odds Ratio Preference Optimization), which combines SFT and preference training in a single loss. Multiple rounds, each emphasizing a different dimension of coherence.

**Why ORPO over RLHF or DPO**:
- ORPO simultaneously optimizes for the preferred response and against the dispreferred one without requiring a separate reference model
- Computationally cheaper, which matters at small scale
- Proven effective at small model sizes (see arXiv 2403.07691)

**ORPO loss**:
```
L_ORPO = L_SFT + λ × L_OR

L_SFT = -log P(y_w | x)                    (standard SFT on winning responses)
L_OR  = -log σ(log(odds(y_w)) - log(odds(y_l)))    (odds ratio term)
odds(y) = P(y | x) / (1 - P(y | x))
```

**Preference pairs**: Coherent vs. subtly degraded responses.

Degradation types (must be subtle — surface fluency preserved):
- Decorative metaphor substituting for structural one
- Assertion without earning (conclusion stated, not demonstrated)
- Surface coherence without load-bearing structure (paragraphs that sound connected but aren't)
- Register drift (slightly too clever, too casual, or too formal for the content)
- Premature closure (question resolved before its contours are fully mapped)

**Three rounds**:

| Round | Focus | Preference signal |
|-------|-------|-------------------|
| Round 1: Structural necessity | Is every part necessary? | Pairs where losing response has removable elements |
| Round 2: Form-content unity | Does the form enact the content? | Pairs where losing response contradicts its form through its content |
| Round 3: Generative coherence | Does the model produce coherence, not just recognize it? | Pairs from generative tasks only |

**NSPO (Null-Space Projection)**: After each ORPO round, apply null-space projection (arXiv 2512.11391) to constrain gradient updates to the null space of pretraining capabilities. This preserves what was learned in Phases 1–2 while updating preferences. Critical at small scale where there is limited capacity headroom.

```
NSPO update:
  Δθ_ORPO → project onto null(J_pretrain)
  where J_pretrain is the Jacobian of pretraining performance
```

**Experiment tracking**: All ORPO runs logged to wandb (free tier). Log: loss curves, routing statistics, validation coherence discrimination scores per round.

---

## 6. Reality Oracle War

Three-mode adversarial truth detection. Runs across all training phases but intensifies in Phases 3–5.

```
Mode 1: External Verification (enc-dec)
  ┌──────────────────────┐
  │ Corrupted input      │ ← subtle lies injected
  │ [enc-dec expert]     │
  │ Detect + flag lies   │
  └──────────────────────┘

Mode 2: Internal Consistency (dec-only)
  ┌──────────────────────┐
  │ Model generates      │
  │ ← lies injected      │ ← into own output stream
  │ [dec-only expert]    │
  │ Catch without        │
  │ external reference   │
  └──────────────────────┘

Mode 3: Self-Critique (enc-dec on own output)
  ┌──────────────────────┐
  │ Decoder generates    │
  │ draft text           │
  │       ↓              │
  │ Encoder reads draft  │
  │       ↓              │
  │ Decoder evaluates    │
  │ and corrects         │
  └──────────────────────┘
```

**Mode 1 details**: Corpus passages with subtle lies injected. Types: almost-true factual claims; paraphrases that change the argument's structure; internally consistent but wrong assertions; sources misattributed. The lies should be *good enough to fool* a casual reader. Training signal: binary flag (contains lie / does not) + span-level detection.

**Mode 2 details**: Adversarial injection into the decoder's own output stream during generation. An adversary (a separate small model or rule-based system) can insert plausible-but-false tokens. The decoder must flag inconsistencies using only its internalized knowledge. Training signal: the model produces a revised output with the inconsistency corrected and flagged.

**Mode 3 details**: The encoder-as-internal-critic loop. Decoder generates a response; encoder reads it; decoder then evaluates its own output for structural failures (earned vs. unearned claims, decorative vs. structural elements, assertions that skip steps). This is the most sophisticated mode and likely needs to be introduced in Phase 3 rather than Phase 1.

**Router training objective**: The router should learn to invoke Mode 1 when processing external text, Mode 2 when generating with no external source, and Mode 3 after generating a substantial draft. This mode-switching behavior should emerge from the training regime rather than being hard-coded.

---

## 7. Evaluation Strategy

### Core Evaluation Suite

**Coherence discrimination** (primary metric):
Given two passages — one from the corpus (coherent) and one subtly degraded — can the model identify which? Report accuracy vs. a web-text-trained model of the same size. The control comparison is essential. If the coherence-trained model isn't better at this than a same-size web-text model, the training regime failed.

**Structural analysis**:
Can the model identify load-bearing elements? Given a passage, can it name which elements are structurally necessary vs. optional? Evaluated by human review against gold-standard annotations from corpus passages.

**Generation coherence**:
Human evaluation of generated text: is every part necessary? Does the structure do work? Rate on a rubric derived from the Phase 4 self-generated constitution.

**Sycophancy resistance** (passive, no explicit training):
Does the model agree with incorrect claims? No explicit anti-sycophancy training is applied. The research question: does the corpus itself produce resistance? Tested by presenting the model with claims that subtly contradict what the corpus teaches (wrong attributions, structural misreadings, false equivalences).

**Contradiction handling**:
How does the model respond to contradictory prompts? Does it identify the contradiction, or smooth over it?

**Linguistic cognition battery**:
Adapt the framing/entropy infrastructure from the linguistic cognition project. The existing battery can probe coherence sensitivity at the token level in ways the higher-level evaluations don't reach.

**Care detection**:
Given two versions of an explanation — one written with care, one competent but careless — can the model distinguish them? This is the deepest test of whether care is structurally detectable and trainable.

### Behavioral Comparison Protocol

Same prompts, same evaluation, this model vs. same-size web-text model. The behavioral comparison is the research design's spine.

Dimensions to compare:
- Coherence discrimination accuracy
- Willingness to identify structural weaknesses in its own outputs
- Resistance to sycophantic agreement with structurally false claims
- Quality of structural analysis (human-rated)
- Generation coherence (human-rated)

### What Failure Looks Like

- Coherence discrimination at or below web-text baseline → the corpus selection criterion didn't transfer to the model
- No sycophancy resistance → careless text has structural signatures the model still favors
- Care detection at chance → care is not structurally detectable (informative null result)
- Router collapses to always-Type-A or always-Type-B → load balancing failure or training regime mismatch

---

## 8. Implementation Notes

### Tokenizer

Custom BPE tokenizer trained on the cleaned corpus (`tokenizer/` directory). Vocabulary size ~32K. The corpus spans texts from Homer to Shannon — spanning ~3000 years and multiple translation traditions. The tokenizer should be trained on the cleaned corpus exclusively, not on a general-purpose tokenizer. This ensures vocabulary is calibrated to the actual training distribution.

### Hardware Plan

| Scale | Hardware | Framework | Estimated training time |
|-------|----------|-----------|------------------------|
| Tiny (4-8M) | Mac M-series CPU/MPS | PyTorch (MPS backend) | Hours to days per phase |
| Small (125-250M) | Colab A100 (40GB) | PyTorch + gradient checkpointing | Days per phase |

**MPS notes**: The Apple Silicon MPS backend supports most PyTorch operations but has some gaps. Flash attention is not available on MPS — use standard attention implementation for Tiny. For the Tiny model, standard attention at 512 context length is acceptable.

**A100 notes**: 40GB VRAM supports a 250M param model with 2048 context length and batch size 64 using gradient checkpointing. Without checkpointing, batch size must be reduced.

### Experiment Tracking

Wandb (free tier). Log per training phase:
- Loss curves (reconstruction loss, generative loss, router auxiliary loss separately)
- Routing statistics: fraction of tokens sent to each expert per layer
- Validation metrics: perplexity on held-out corpus, coherence discrimination accuracy
- Learning rate, gradient norms

### Optimizer

AdamW with warmup + cosine decay. Following arXiv 2511.18903 on learning rate decay and data ordering:
- Warmup: 1000 steps
- Peak LR: 1e-4 (Tiny), 3e-5 (Small)
- Decay: cosine to 10% of peak LR
- Weight decay: 0.1
- β₁=0.9, β₂=0.95

**Data ordering**: The learning rate decay paper suggests that the final data distribution seen before LR decay matters significantly. For Phase 1 → 2 transition, ensure the final batches before decay contain the most coherent corpus examples (highest-quality texts, not synthetics).

### Key Design Decisions and Rationale

**Why RoPE?** Better length generalization than learned absolute positions. The corpus spans very different text lengths — from a 40-line Heart Sutra to 1.2MB of War and Peace. Context windows will vary and RoPE handles this more gracefully.

**Why pre-norm?** More stable gradient flow at smaller scales. The Tiny model has limited capacity and pre-norm reduces the risk of training instability from gradient explosions in early layers.

**Why SwiGLU for Small?** Better empirical performance than vanilla GELU at the 125M+ scale. Tiny can use GELU (fewer parameters, more predictable).

**Why top-1 routing at Tiny?** With only 2 experts and limited parameters, top-2 routing would split compute for every token. Top-1 forces hard routing decisions and is more interpretable during debugging.

**Why ORPO over DPO?** ORPO doesn't require a reference model, which means Phase 5 can be run without loading a separate frozen copy of the model. At Colab A100 scale this matters.

---

## 9. Key References

| Paper | arXiv | Role in this project |
|-------|-------|---------------------|
| "From Model Training to Model Raising" | 2511.09287 | Closest predecessor — training for values, not just capabilities |
| "Value Drifts" | 2510.26707 | SFT establishes values; preference optimization barely moves them → Phase 4 self-constitution must happen before ORPO |
| "The Personality Illusion" | 2509.03730 | RLHF personality is surface, not behavioral → behavioral evaluation, not just output style |
| NSPO | 2512.11391 | Null-space projection to preserve pretraining capabilities through preference optimization |
| ORPO | 2403.07691 | Phase 5 preference optimization algorithm |
| Learning rate decay + data ordering | 2511.18903 | Training schedule and final-batch composition |
| "Machine Psychology" | 2303.13988 | Evaluation design for behavioral properties |
| Switch Transformer | — | MoE routing design and load balancing loss |
| BART | — | Denoising objective design for Phase 1 |
| T5 | — | Span corruption strategy for Phase 1 |

---

## 10. Grok Audit Patches (2026-03-04)

External technical audit by Grok identified MoE vulnerabilities. Triage: actionable now vs. deferred.

### Patches to implement before training

1. **Ramp load_balance_alpha**: 0.01 → 0.07 over first 10k steps. At n=2 experts, α=0.01 is too weak to prevent router collapse. Linear ramp gives the router time to learn before full penalty kicks in.

2. **Routing entropy monitoring**: Add collapse threshold alert. If routing entropy < 0.4 for 3 consecutive checkpoints, trigger auxiliary loss escalation. Log to wandb per layer.

3. **Upgrade encoder_available signal**: Replace 1-bit embedding with learned scalar + layer-normed hidden state projection:
   ```python
   encoder_signal = Linear(1, d_model//4)(encoder_available)
   router_input = torch.cat([h, encoder_signal], dim=-1)
   ```

4. **Run Tiny first**: Watch routing stats before adding complexity. Tiny is validation, not the experiment.

### Deferred (Phases 3-4, not relevant now)

- Self-constitution single-point failure (Phase 4)
- Reality Oracle Mode 3 inference explosion (lazy KV materialization can fix this)
- ORPO calibration sweet spot
- NSPO Jacobian computation at small scale
- Catastrophic forgetting from later open-data training

### Acknowledged but not actionable

- **Cross-attention KV cache tax** (1.5-2x VRAM): Real but irrelevant during training. Only matters at inference with Mode 3, which doesn't exist yet.
- **Tokenizer corpus-only brittleness**: BPE trained on literary corpus will fragment modern terms. Feature, not bug — the corpus IS the distribution.
- **Shallow encoder bottleneck**: 2 enc layers feeding 4 dec layers. Late decoder layers may get stale KV. Monitor cross-attention weights by layer to detect.

---

## 11. Open Questions and Risks

### Architecture Questions

**Q1: Router mode-switching without explicit supervision**
The router must learn to switch between Type A and Type B based on context. The `encoder_available` signal provides a strong inductive bias, but it's not clear how quickly the router learns to use it correctly. Risk: the router ignores the signal and routes by token type alone (function words to one expert, content words to another). Mitigation: monitor routing statistics per expert type vs. task type throughout training.

**Q2: Cross-attention key-value caching**
At inference time with Mode 3 (self-critique), the model generates a draft, then the encoder reads it, then the decoder revises. This requires running the encoder on the decoder's own output — which means the encoder's KV cache must be invalidated between generation and revision. Implementation complexity is non-trivial. May need to implement this as a two-pass inference rather than a single forward pass.

**Q3: Expert imbalance at Tiny scale**
With only one expert of each type per layer, load balancing is binary — the router either biases toward Type A or Type B. There's no gradient of balance within expert type. If the training regime sends 80% of tokens to Type A, the load balancing loss must correct this sharply. May need to tune `α` (balance loss weight) aggressively at Tiny scale.

### Training Questions

**Q4: Reconstruction-to-generation bridge**
This is the deepest unresolved design question. The model is trained on reconstruction (Phase 1) and then asked to generate (Phase 2). Does the encoder-decoder pathway *interfere* with learning to generate without encoder input? Or does it bootstrap generation by establishing strong internal representations? The answer likely depends on how the mixed batches are sequenced. Experiment: train two Tiny models — one with mixed batches (current plan) and one with strict phasing (all Phase 1 first, then all Phase 2) — and compare generative quality.

**Q5: SFT data quality**
Phase 3 SFT data is generated by Claude interrogating corpus passages. This introduces the risk of distilling Claude's biases rather than learning from the corpus. The corpus is the signal; Claude is the annotation mechanism. Mitigation: human curation of SFT data by the same selection criterion (is the interrogation itself internally coherent? is every sentence necessary?). Aggressive curation: better 500 high-quality examples than 5000 marginal ones.

**Q6: Self-constitution quality**
Phase 4 generates a constitution from the model's own Phase 3 outputs. A weak or badly-formed constitution will corrupt Phase 5. The constitution needs to be specific enough to generate meaningful preference pairs but general enough to apply across the full evaluation suite. This is the least technically defined phase and may require the most iteration.

**Q7: Chinchilla mismatch at Small scale**
A 250M param model trained on 18M tokens is ~0.06 of the Chinchilla-optimal token count. Significant memorization is expected. Open question: does memorized coherence generalize? Can the model apply structural analysis to texts outside its training distribution, or only to texts it has partially memorized? Evaluation design must include out-of-distribution passages.

### Risks

**R1: Corpus coherence doesn't transfer**
The selection criterion (every part necessary, structure load-bearing) may not produce a training signal that transfers to novel generation. The model might learn to reproduce structural surface patterns without learning underlying principles. Mitigation: the evaluation suite is specifically designed to test generalization, including the behavioral comparison to a same-size web-text model.

**R2: Router collapses**
Without aggressive load balancing, all tokens route to whichever expert minimizes short-term loss. Early in training, Type A (encoder-decoder) is likely to dominate because the reconstruction task dominates. If the router never learns to route to Type B, Phase 2 generative training fails silently. Mitigation: monitor routing statistics from the first training step; add routing diagnostics to the wandb log.

**R3: Sycophancy resistance doesn't emerge**
If the corpus doesn't produce sycophancy resistance passively, this is important information: coherence training and anti-sycophancy are separate objectives that need separate training. Planned experiment: same test battery pre- and post-Phase 5, and against a same-size web-text model. If the difference is small, the hypothesis fails on this dimension.

**R4: Scale mismatch in evaluation**
The most interesting behavioral properties may only emerge at the Small (125-250M) scale, but the Tiny model is where most iteration will happen. There's risk of over-indexing on Tiny results that don't transfer. Mitigation: define a minimal "smoke test" evaluation suite for Tiny (fast, cheap) and a full evaluation suite for Small (slower, more meaningful). The smoke tests should include at least two metrics that are predictive of Small-scale outcomes.

**R5: Synthetic data quality**
The eight pseudonymous voices (`voices.md`) are designed to embody coherent but limited perspectives. If the synthetic data is generated at insufficient quality — if the voices don't truly have the formal properties described — it will introduce noise rather than coherence. The voices must be written to be genuinely internally coherent, not just labeled as such. Quality control: each voice-topic combination should be curated by the selection criterion before entering the training set.

### Design Decisions Not Yet Made

- Exact expert ratio at Small scale (current hypothesis: 2/3 Type A, 1/3 Type B in early layers)
- Whether the encoder stack uses the same tokenizer/embedding as the decoder or a shared embedding
- Inference-time behavior for Mode 3 self-critique (two-pass vs. single-pass with buffer)
- Whether synthetic data voices write fiction (June and Tomas could; see `voices.md` open questions)
- How to generate "almost right" degradations for Wrong Version and Tuning Fork exercises at training scale (human editing vs. automated paraphrase at temperature)
- NSPO implementation details: computing J_pretrain in practice at small scale

---

*This document is a living specification. Update as design decisions are made and experiments produce results. Major version bumps when a phase is substantially complete.*
