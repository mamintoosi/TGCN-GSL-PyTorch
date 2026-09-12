"""Light Stage 58 checks: recompute Stage 41 means from Stage 40 JSONs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
TRAIN = ROOT / "results/stage40_canonical/training"
S41 = ROOT / "gsl_stage41/stage41_summary.csv"
df = pd.read_csv(S41)

SEEDS = [42, 43, 44, 45, 46]
CHECKS = [
    ("losloop", 1, "no_spatial", "T-GCN-NoSpatial"),
    ("losloop", 1, "physical", "T-GCN"),
    ("losloop", 1, "multi_gsl_mix", "T-GCN-MultiGSL-Mix"),
    ("losloop", 1, "gcn_no_spatial", "GCN-NoSpatial"),
    ("shenzhen", 1, "multi_gsl_mix", "T-GCN-MultiGSL-Mix"),
    ("losloop", 4, "gcn_physical", "GCN"),
]

print("=== Recompute Stage 41 mean/std from Stage 40 JSONs (ddof=1) ===")
ok = True
for ds, ph, vid, method in CHECKS:
    rmses, maes = [], []
    for seed in SEEDS:
        p = TRAIN / f"{ds}_ph{ph}_seed{seed}_{vid}.json"
        with open(p) as f:
            r = json.load(f)
        rmses.append(r["rmse"])
        maes.append(r["mae"])
    rm = np.array(rmses, dtype=float)
    ma = np.array(maes, dtype=float)
    row = df[(df.dataset == ds) & (df.ph == ph) & (df.variant_id == vid)].iloc[0]
    d_mean = abs(rm.mean() - row.rmse_mean)
    d_std = abs(rm.std(ddof=1) - row.rmse_std)
    d_mae = abs(ma.mean() - row.mae_mean)
    flag = "OK" if d_mean < 5e-4 and d_std < 5e-4 and d_mae < 5e-4 else "MISMATCH"
    if flag != "OK":
        ok = False
    print(f"{flag:8s} {ds:8s} PH{ph} {method:22s} "
          f"json_mean={rm.mean():.6f} csv={row.rmse_mean:.6f} "
          f"json_std1={rm.std(ddof=1):.6f} csv={row.rmse_std:.6f} "
          f"mae {ma.mean():.6f} vs {row.mae_mean:.6f}")

print("\n=== ddof=0 vs ddof=1 on one cell (los PH1 no_spatial) ===")
rm = [json.load(open(TRAIN / f"losloop_ph1_seed{s}_no_spatial.json"))["rmse"] for s in SEEDS]
print("values", rm)
print("mean", np.mean(rm), "std_ddof0", np.std(rm), "std_ddof1", np.std(rm, ddof=1))
row = df[(df.dataset=="losloop")&(df.ph==1)&(df.variant_id=="no_spatial")].iloc[0]
print("stage41 csv std", row.rmse_std)

print("\n=== Stage 29 15-min std convention (if artifacts exist) ===")
p29 = list((ROOT / "results").rglob("*stage29*results*.json"))
print("found", p29[:5])

print("\nALL_MATCH" if ok else "SOME_MISMATCH")
