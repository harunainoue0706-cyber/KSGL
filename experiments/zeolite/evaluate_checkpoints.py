from pathlib import Path
import math
import csv
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate

from symgraph.dataset import create_dataloaders
from symgraph.model.model_utils import langmuir_freundlich_2s


ROOT = Path("models")
OUT = Path("FINAL_CKPT_EVAL")
OUT.mkdir(exist_ok=True)

MODELS = ["symgnn", "ksgl"]
SEEDS = [2023, 2024, 2025]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, testloader, std, pres_points=100):
    model.eval()

    test_mse = 0.0
    test_mae = 0.0

    iso_mse_sum = 0.0
    iso_mae_sum = 0.0
    total_iso = 0

    p = torch.linspace(
        1, 7, pres_points, device=DEVICE
    ).unsqueeze(-1)

    dp = p[1] - p[0]

    p_in = torch.cat(
        [p, p[[-1]] + dp],
        dim=0
    ) - dp / 2

    for data in testloader:
        data = data.to(DEVICE)

        out, q_prime_hat = model(data, pres=p_in)

        out = out.squeeze(-1)

        # -------------------------
        # HOA
        # Same definition as train.py
        # -------------------------
        test_mse += F.mse_loss(
            out * std,
            data.y * std
        ).item()

        test_mae += F.l1_loss(
            out * std,
            data.y * std
        ).item()

        # -------------------------
        # Isotherm
        # -------------------------
        true_iso = langmuir_freundlich_2s(
            data.iso_params,
            10 ** p
        ).T

        pred_iso = torch.cumulative_trapezoid(
            q_prime_hat,
            p_in.squeeze(-1),
            dim=-1
        )

        iso_mask = data.iso == 1
        n_iso = iso_mask.sum().item()

        if n_iso > 0:
            true_iso = true_iso[iso_mask]
            pred_iso = pred_iso[iso_mask]

            iso_mse_sum += (
                F.mse_loss(pred_iso, true_iso).item()
                * n_iso
            )

            iso_mae_sum += (
                F.l1_loss(pred_iso, true_iso).item()
                * n_iso
            )

            total_iso += n_iso

    hoa_mse = test_mse / len(testloader)
    hoa_mae = test_mae / len(testloader)

    iso_mse = iso_mse_sum / total_iso
    iso_mae = iso_mae_sum / total_iso

    return hoa_mse, hoa_mae, iso_mse, iso_mae


rows = []

for model_name in MODELS:
    for seed in SEEDS:

        exp = ROOT / f"FINAL_{model_name}_seed{seed}"

        cfg_path = exp / "config.yaml"
        ckpt_path = exp / "final.pth"
        mu_path = exp / "mu.npy"
        std_path = exp / "std.npy"

        print()
        print("=" * 80)
        print(model_name, seed)
        print("DIR:", exp)
        print("=" * 80)

        missing = [
            p for p in
            [cfg_path, ckpt_path, mu_path, std_path]
            if not p.exists()
        ]

        if missing:
            print("MISSING:")
            for p in missing:
                print(" ", p)
            continue

        cfg = OmegaConf.load(cfg_path)

        # Re-create exactly the same dataset split/config
        trainloader, valloader, testloader, mu, std_loader = \
            create_dataloaders(**cfg.dataset)

        # Use the saved training normalization
        mu = float(np.load(mu_path))
        std = float(np.load(std_path))

        # Generic Hydra construction:
        # works for SymGNN and local KSGL/KSGL config
        model = instantiate(cfg.model).to(DEVICE)

        state = torch.load(
            ckpt_path,
            map_location=DEVICE
        )

        model.load_state_dict(state)

        nsteps = int(cfg.isotherm.n_steps)

        hoa_mse, hoa_mae, iso_mse, iso_mae = evaluate(
            model,
            testloader,
            std,
            pres_points=nsteps
        )

        print(f"FINAL HOA MSE      = {hoa_mse:.4f}")
        print(f"FINAL HOA MAE      = {hoa_mae:.4f}")
        print(f"FINAL Isotherm MSE = {iso_mse:.4f}")
        print(f"FINAL Isotherm MAE = {iso_mae:.4f}")

        rows.append({
            "model": model_name,
            "seed": seed,
            "hoa_mse": hoa_mse,
            "hoa_mae": hoa_mae,
            "iso_mse": iso_mse,
            "iso_mae": iso_mae,
        })


# -------------------------
# Per-seed CSV
# -------------------------
csv_path = OUT / "final_checkpoint_per_seed.csv"

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "seed",
            "hoa_mse",
            "hoa_mae",
            "iso_mse",
            "iso_mae",
        ]
    )
    w.writeheader()
    w.writerows(rows)


# -------------------------
# Summary
# -------------------------
print()
print("=" * 90)
print("FINAL CHECKPOINT 3-SEED SUMMARY")
print("mean +/- SD [SE]")
print("=" * 90)

metrics = [
    ("hoa_mse", "HOA MSE"),
    ("hoa_mae", "HOA MAE"),
    ("iso_mse", "Iso MSE"),
    ("iso_mae", "Iso MAE"),
]

summary_rows = []

for model_name in MODELS:

    rr = [
        r for r in rows
        if r["model"] == model_name
    ]

    if len(rr) != 3:
        print(
            f"{model_name}: "
            f"only {len(rr)} valid seeds"
        )
        continue

    print()
    print(model_name.upper())

    out = {"model": model_name}

    for key, label in metrics:

        vals = [r[key] for r in rr]

        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        se = sd / math.sqrt(len(vals))

        print(
            f"{label:<10}: "
            f"{mean:.4f} +/- {sd:.4f} "
            f"[SE={se:.4f}]"
        )

        out[key + "_mean"] = mean
        out[key + "_sd"] = sd
        out[key + "_se"] = se

    summary_rows.append(out)


summary_path = OUT / "final_checkpoint_summary.csv"

if summary_rows:
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys())
        )
        w.writeheader()
        w.writerows(summary_rows)


print()
print("=" * 90)
print("SAVED:")
print(csv_path)
print(summary_path)
print("=" * 90)
