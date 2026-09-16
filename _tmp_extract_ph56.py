import json
from collections import defaultdict


def row(g):
    return (g["display"], g["rmse_mean"], g["rmse_std"], g.get("cell_provenance"), g.get("status"))


s59 = json.load(open("results/stage59_ph56_horizon/ph56_horizon_summary.json"))
supp = json.load(open("results/supplementary_ph56/supplementary_summary.json"))
lam = json.load(open("results/supplementary_ph56_lambda001/supplementary_summary.json"))

print("=== stage59 losloop PH5-6 ===")
for ph in ["5", "6"]:
    print("PH", ph)
    for k, v in s59["datasets"]["losloop"][ph].items():
        print(f"  {v['display']:28s} {v['rmse_mean']:.2f} ({v['rmse_std']:.2f}) edges={v['n_edges']}")

print("\n=== supplementary losloop PH5-6 (10 variants) ===")
for ph in [5, 6]:
    rows = [g for g in supp["groups"] if g["dataset"] == "losloop" and g["ph"] == ph]
    print("PH", ph)
    for g in sorted(rows, key=lambda x: x["display"]):
        print(f"  {g['display']:28s} {g['rmse_mean']:.2f} ({g['rmse_std']:.2f}) {g['cell_provenance']}")

print("\n=== supplementary shenzhen PH5-6 ===")
for ph in [5, 6]:
    rows = [g for g in supp["groups"] if g["dataset"] == "shenzhen" and g["ph"] == ph]
    print("PH", ph)
    for g in sorted(rows, key=lambda x: x["display"]):
        print(f"  {g['display']:28s} {g['rmse_mean']:.2f} ({g['rmse_std']:.2f})")

print("\n=== lambda001 losloop PH5-6 ===")
for ph in [5, 6]:
    rows = [g for g in lam["groups"] if g["dataset"] == "losloop" and g["ph"] == ph]
    print("PH", ph)
    for g in sorted(rows, key=lambda x: x["display"]):
        print(f"  {g['display']:28s} {g['rmse_mean']:.2f} ({g['rmse_std']:.2f})")

# Compare GSL edge counts lambda
print("\n=== GSL n_edges / fit times ===")
for path, name in [
    ("results/supplementary_ph56/supplementary_summary.json", "supp"),
    ("results/supplementary_ph56_lambda001/supplementary_summary.json", "lam001"),
]:
    d = json.load(open(path))
    for g in d["groups"]:
        if g["variant"] in ("gsl", "gcn_gsl") and g["dataset"] == "losloop":
            meta = g.get("graph_meta", {}).get("gsl", {})
            print(name, "los", "PH", g["ph"], g["variant"], "edges", meta.get("n_edges"), "fit_s", meta.get("fit_time_s"))

# Mix vs no_spatial percent
print("\n=== Mix vs graph-free (stage59 los) ===")
for ph in ["5", "6"]:
    d = s59["datasets"]["losloop"][ph]
    ns = d["no_spatial"]["rmse_mean"]
    mix = d["multi_gsl_mix"]["rmse_mean"]
    print(f"PH{ph}: mix={mix:.2f} ns={ns:.2f} rel={(ns-mix)/ns*100:.1f}%")

print("\n=== MultiGSL vs graph-free (supp los) ===")
for ph in [5, 6]:
    rows = {g["variant"]: g for g in supp["groups"] if g["dataset"] == "losloop" and g["ph"] == ph}
    ns = rows["no_spatial"]["rmse_mean"]
    mg = rows["multi_gsl"]["rmse_mean"]
    print(f"PH{ph}: multi={mg:.2f} ns={ns:.2f} rel={(ns-mg)/ns*100:.1f}%")

print("\n=== MultiGSL vs graph-free (supp sz) ===")
for ph in [5, 6]:
    rows = {g["variant"]: g for g in supp["groups"] if g["dataset"] == "shenzhen" and g["ph"] == ph}
    ns = rows["no_spatial"]["rmse_mean"]
    mg = rows["multi_gsl"]["rmse_mean"]
    print(f"PH{ph}: multi={mg:.4f} ns={ns:.4f} rel={(ns-mg)/ns*100:.2f}%")
