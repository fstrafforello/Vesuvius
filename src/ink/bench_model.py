"""Measure training speed and peak GPU memory of a model at several batch/patch sizes.

Usage:  uv run python bench_model.py full3d
"""

import sys
import time

import torch

from models import build

kind = sys.argv[1] if len(sys.argv) > 1 else "full3d"
model = build(kind, 32).cuda()
print(f"{kind}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
opt = torch.optim.AdamW(model.parameters())

for bs, patch in ((4, 384), (6, 384), (8, 384), (8, 256), (12, 256)):
    try:
        x = torch.randn(bs, 32, patch, patch, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            if i == 3:
                torch.cuda.synchronize()
                t = time.time()
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x).float().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"bs {bs:2d} patch {patch}: {5 / dt:.2f} it/s | {5 * bs / dt:.1f} samples/s | "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"bs {bs:2d} patch {patch}: out of memory", flush=True)
    x = None
