#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs"
OUTDIR = ROOT / "results_summary"
OUTDIR.mkdir(exist_ok=True)

MODELS = ["gnn", "symgnn", "ksgl", "alignn", "matformer"]
TASKS = ["gen", "int"]
SEEDS = [2023, 2024, 2025]

PATTERNS = {
    "hoa_mse": re.compile(r"Test mse_loss:\s*([0-9.eE+-]+)"),
    "hoa_mae": re.compile(r"Test l1_loss:\s*([0-9.eE+-]+)"),
    "iso_mse": re.compile(r"Test mse_loss \(Iso\):\s*([0-9.eE+-]+)"),
    "iso_mae": re.compile(r"Test l1_loss \(Iso\):\s*([0-9.eE+-]+)"),
}
BEST = re.compile(r"BEST_RELOAD\s+epoch=(\d+)\s+val_loss=([0-9.eE+-]+)")

rows = []
for task in TASKS:
    for model in MODELS:
        for seed in SEEDS:
            path = LOGDIR / f"{task.upper()}_{model}_seed{seed}.log"
            if not path.exists():
                print("MISSING", path)
                continue
            text = path.read_text(errors="replace")
            bm = BEST.findall(text)
            if not bm:
                print("NO BEST_RELOAD", path)
                continue
            epoch, val = bm[-1]
            row = {
                "task": task,
                "model": model,
                "seed": seed,
                "best_epoch": int(epoch),
                "best_val_loss": float(val),
            }
            ok = True
            for key, pat in PATTERNS.items():
                vals = pat.findall(text)
                if not vals:
                    print("MISSING", key, path)
                    ok = False
                    break
                row[key] = float(vals[-1])
            if ok:
                rows.append(row)

per_seed = OUTDIR / "per_seed.csv"
if rows:
    with per_seed.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

summary = []
for task in TASKS:
    for model in MODELS:
        rr = [r for r in rows if r["task"] == task and r["model"] == model]
        if len(rr) != 3:
            continue
        out = {"task": task, "model": model, "n": 3}
        for metric in ["hoa_mae", "hoa_mse", "iso_mae", "iso_mse"]:
            vals = [r[metric] for r in rr]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            se = sd / math.sqrt(3)
            out[metric + "_mean"] = mean
            out[metric + "_sd"] = sd
            out[metric + "_se"] = se
        summary.append(out)

summary_csv = OUTDIR / "summary_mean_sd_se.csv"
if summary:
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

print("\n" + "=" * 110)
print("ZEOLITE 3-SEED SUMMARY: mean +/- sample SD [SE]")
print("=" * 110)
for r in summary:
    print(f"\n{r['task'].upper():3s}  {r['model'].upper()}")
    for key, label in [("hoa_mae","HOA MAE"),("hoa_mse","HOA MSE"),("iso_mae","Iso MAE"),("iso_mse","Iso MSE")]:
        print(f"  {label:<8}: {r[key+'_mean']:.4f} +/- {r[key+'_sd']:.4f} [SE={r[key+'_se']:.4f}]")
print("\nWROTE", per_seed)
print("WROTE", summary_csv)
