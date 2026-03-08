# vast.ai Deployment Playbook

Quick reference for deploying training runs on vast.ai.

## Prerequisites

- **vastai CLI**: Requires Python 3.10+. On this machine: `/opt/homebrew/bin/vastai` (installed via pip for python3.11)
- **API key**: Stored at `~/.vast_api_key`
- **Local files needed**:
  - Checkpoint: `checkpoints/step_XXXXXX.pt` (~147MB for tiny model)
  - Corpus: `cleaned/` directory (202 .txt files). Pre-tar for speed: `tar czf /tmp/cleaned_corpus.tar.gz cleaned/`
  - Code: Pulled from GitHub (`s0ph1stry/corpus-training`)

## 1. Find an Instance

```bash
# Search for offers — need 40GB+ VRAM to avoid OOM with batch_size=8
# RTX 3060 (12GB) OOMs on cross-attention in Type A experts
/opt/homebrew/bin/vastai search offers \
  'gpu_ram>=40 num_gpus=1 inet_down>=200 reliability>0.95 cuda_vers>=12.0' \
  --order 'dph' | head -15
```

**Known working**: Quadro RTX 8000 (48GB), A40 (48GB), RTX A6000 (48GB), L40S (48GB).
**Known OOM**: RTX 3060 (12GB) at batch_size=8 and batch_size=4.

## 2. Create Instance

```bash
/opt/homebrew/bin/vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 30 \
  --onstart-cmd 'pip install tokenizers && pip install mamba-ssm --no-build-isolation && cd /root && git clone https://github.com/s0ph1stry/corpus-training.git && mkdir -p /root/corpus-training/checkpoints'
```

**IMPORTANT**: Must use `devel` image (not `runtime`) — mamba-ssm needs nvcc to compile.

## 3. Wait for Readiness

Instance takes ~2-5 min to provision, then onstart runs (mamba-ssm compilation takes a few minutes).

```bash
# Check status
/opt/homebrew/bin/vastai show instances --raw | python3.11 -c "
import json,sys
for inst in json.load(sys.stdin):
    print(f\"ID {inst['id']}: {inst.get('actual_status','?')} ssh={inst.get('ssh_host','')}:{inst.get('ssh_port','')}\")
"

# Test readiness (replace host/port)
ssh -o StrictHostKeyChecking=no -p <PORT> root@<HOST> \
  'test -d /root/corpus-training && python3 -c "import mamba_ssm" && echo READY || echo WAITING'
```

## 4. Upload Data

```bash
# Pre-tar corpus if not already done
cd "/Users/sorenanderson/Documents/workspace/06 Projects/13 corpus-training"
tar czf /tmp/cleaned_corpus.tar.gz cleaned/

# Upload checkpoint and corpus (can run in parallel)
scp -o StrictHostKeyChecking=no -P <PORT> checkpoints/step_XXXXXX.pt root@<HOST>:/root/corpus-training/checkpoints/

scp -o StrictHostKeyChecking=no -P <PORT> /tmp/cleaned_corpus.tar.gz root@<HOST>:/root/corpus-training/
ssh -o StrictHostKeyChecking=no -p <PORT> root@<HOST> \
  'cd /root/corpus-training && tar xzf cleaned_corpus.tar.gz && rm cleaned_corpus.tar.gz'
```

## 5. Start Training

```bash
ssh -o StrictHostKeyChecking=no -p <PORT> root@<HOST> \
  'cd /root/corpus-training && nohup python3 -u -m training.train_phase1 \
    --preset tiny \
    --total-steps 100000 \
    --batch-size 8 \
    --checkpoint-every 2000 \
    --no-wandb \
    --resume checkpoints/step_XXXXXX.pt \
    --weights-only \
    --resurrect \
    > /root/training.log 2>&1 &'
```

Flags:
- `--weights-only`: Load model weights but reset optimizer/scheduler (fresh start from checkpoint weights)
- `--resurrect`: Detect and re-initialize dead experts (gate bias → +0.5)
- `--no-wandb`: Skip wandb logging (use for quick runs)

## 6. Monitor

```bash
# Check log
ssh -p <PORT> root@<HOST> 'tail -20 /root/training.log'

# Check GPU
ssh -p <PORT> root@<HOST> 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```

Log format: `step XXXXX | loss X.XXXX | aux X.XXXX | lr X.XXe-XX | grad X.XX | ent X.XXX | R:XX% S:XX% X:XX%`

First log line appears at step 100 (log_every=100). First batch is slow due to CUDA kernel compilation — expect 3-5 min before first output.

## 7. Download Checkpoints

```bash
scp -o StrictHostKeyChecking=no -P <PORT> root@<HOST>:/root/corpus-training/checkpoints/step_XXXXXX.pt checkpoints/
```

## 8. Destroy Instance

```bash
/opt/homebrew/bin/vastai destroy instance <INSTANCE_ID>
```

**Don't forget this!** Instances bill by the hour.

## Known Issues

- **Python 3.9 vastai CLI broken**: The `vast.py` module uses `dict[str, str] | None` syntax requiring 3.10+. Use `/opt/homebrew/bin/vastai` (python3.11) instead of system `vastai`.
- **China instances**: PyPI downloads timeout. Use non-China instances.
- **`runtime` vs `devel` image**: `runtime` image lacks nvcc → mamba-ssm build fails with `NameError: name 'bare_metal_version' is not defined`.
- **MPS Metal (local M4)**: `MPSNDArray buffer not large enough` — model tensor shapes exceed Metal limits. Use `FORCE_CPU=1` env var for local testing.
- **Instance provisioning failures**: vast.ai instances sometimes get stuck in "loading" or "created" state. If no progress after 5 min, destroy and try a different offer.

## Current Run (Run 5)

- **Instance**: 32469159 (Michigan, Quadro RTX 8000 48GB, $0.21/hr)
- **SSH**: `ssh -p 29158 root@ssh6.vast.ai`
- **Started**: 2026-03-06 ~23:15 UTC
- **Config**: tiny preset, batch_size=8, steps 50K→100K, v2.1 fixes (jitter, liveness, resurrection, annealed modes)
- **Checkpoint resume**: step_050000.pt (weights only)
- **Resurrected**: 7 dead experts across layers 0, 2, 3, 5
