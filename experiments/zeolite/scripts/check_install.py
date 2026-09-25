#!/usr/bin/env python3
mods = ["torch", "torch_geometric", "torch_scatter", "torch_sparse", "numpy", "scipy", "hydra", "omegaconf", "pymatgen", "spglib", "wandb"]
failed = []
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK {m:16s} {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        failed.append((m, str(e)))
        print(f"FAIL {m:14s} {e}")
if failed:
    raise SystemExit(1)
print("ENVIRONMENT CHECK: OK")
