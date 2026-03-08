"""Probe per-expert routing from a checkpoint without interrupting training.

Loads checkpoint, runs a forward pass on sample data, reports per-layer
per-expert activation fractions. Outputs CSV for sonification.
"""
import sys, os, json, glob, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.config import ModelConfig
from model.model import HeteroMoETransformer
from model.experts import MOEManager
from data.dataset import CorpusDataset

def main():
    # Find latest checkpoint
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'checkpoints', 'phase1')
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pt')))
    if not ckpts:
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'checkpoints')
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pt')))
    if not ckpts:
        print("No checkpoints found"); return

    ckpt_path = ckpts[-1]
    print(f"Loading: {ckpt_path}")

    # Load config and model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config_data = ckpt.get('config', {})
    if isinstance(config_data, dict):
        # Filter to only valid ModelConfig fields
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}
        config = ModelConfig(**filtered)
    else:
        config = config_data

    model = HeteroMoETransformer(config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load some data for the forward pass
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned')
    files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    if not files:
        print("No data files found"); return

    # Sample N batches for more stable estimates
    n_batches = 20
    batch_size = 4
    seq_len = getattr(config, 'max_seq_len', getattr(config, 'context_len', 512))

    # Simple tokenization
    from tokenizers import Tokenizer
    tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'tokenizer', 'tokenizer_with_authors.json')
    tokenizer = Tokenizer.from_file(tok_path)

    # Accumulate fracs across batches
    accum_fracs = None
    accum_count = 0

    with torch.no_grad():
        for batch_i in range(n_batches):
            # Pick random files and chunks
            texts = []
            for _ in range(batch_size):
                f = random.choice(files)
                with open(f) as fh:
                    text = fh.read()
                # Random chunk
                start = random.randint(0, max(0, len(text) - 500))
                texts.append(text[start:start+500])

            # Tokenize
            encodings = [tokenizer.encode(t) for t in texts]
            ids = []
            for enc in encodings:
                toks = enc.ids[:seq_len]
                toks = toks + [0] * (seq_len - len(toks))
                ids.append(toks)

            input_ids = torch.tensor(ids, device=device)
            # Create padding mask
            padding_mask = (input_ids != 0)

            # Forward pass — need to reset moe_manager first
            model.moe_manager.reset()
            _ = model(decoder_input_ids=input_ids,
                      decoder_padding_mask=padding_mask)

            # Collect per-expert fracs
            fracs = model.moe_manager.layer_expert_fracs  # list of lists
            if accum_fracs is None:
                accum_fracs = [list(layer_fracs) for layer_fracs in fracs]
            else:
                for li, layer_fracs in enumerate(fracs):
                    for ei, f in enumerate(layer_fracs):
                        accum_fracs[li][ei] += f
            accum_count += 1

    # Average
    for li in range(len(accum_fracs)):
        for ei in range(len(accum_fracs[li])):
            accum_fracs[li][ei] /= accum_count

    # Report
    expert_types = ['A', 'B']  # n_type_a=1, then n_type_b=1
    resurrected = {(0,0), (0,1), (2,0), (3,0), (3,1), (5,0), (5,1)}

    print("\n=== Per-Expert Routing Fractions ===")
    print("layer,expert,type,frac,resurrected")
    for li, layer_fracs in enumerate(accum_fracs):
        for ei, f in enumerate(layer_fracs):
            etype = expert_types[ei] if ei < len(expert_types) else '?'
            res = 'yes' if (li, ei) in resurrected else 'no'
            print(f"{li},{ei},{etype},{f:.4f},{res}")

    # Also print a human-readable summary
    print("\n=== Summary ===")
    for li, layer_fracs in enumerate(accum_fracs):
        parts = []
        for ei, f in enumerate(layer_fracs):
            etype = expert_types[ei] if ei < len(expert_types) else '?'
            marker = " *" if (li, ei) in resurrected else ""
            parts.append(f"  expert {ei} (Type {etype}): {f*100:.1f}%{marker}")
        print(f"Layer {li}:")
        for p in parts:
            print(p)

if __name__ == '__main__':
    main()
