import pandas as pd

df = pd.read_csv("results/stage57_tgcn_gcn_audit/full.csv")
print("rows", len(df))
print(df.groupby(["arm", "adj_kind", "dataset"]).size())
print("epochs", df.max_epochs.unique(), "seeds", sorted(df.seed.unique()), "phs", sorted(df.ph.unique()))

for adj in ["identity", "physical"]:
    print("\n==========", adj, "==========")
    sub = df[df.adj_kind == adj]
    g = (
        sub.groupby(["dataset", "ph", "arm"])
        .agg(rmse=("RMSE", "mean"), rmse_std=("RMSE", "std"), mae=("MAE", "mean"), n=("RMSE", "count"))
        .reset_index()
    )
    for ds in ["losloop", "shenzhen"]:
        print(f"\n--- {ds} ---")
        for ph in [1, 2, 3, 4]:
            rows = g[(g.dataset == ds) & (g.ph == ph)].sort_values("rmse")
            print(f"PH{ph}:")
            for _, r in rows.iterrows():
                print(f"  {r.arm:14s}  {r.rmse:7.3f} ({r.rmse_std:.3f})  n={int(r.n)}")

print("\n========== KEY DELTAS (mean RMSE over seeds) ==========")
for ds in ["losloop", "shenzhen"]:
    for ph in [1, 2, 3, 4]:
        for adj in ["identity", "physical"]:
            m = df[(df.dataset == ds) & (df.ph == ph) & (df.adj_kind == adj)].groupby("arm")["RMSE"].mean()

            def get(a):
                return float(m[a]) if a in m.index else float("nan")

            A, C, B, D = get("A_gcn_mse"), get("C_tgcn_mse"), get("B_tgcn_reg"), get("D_gcn_reg")
            print(
                f"{ds:8s} PH{ph} {adj:8s}  A={A:6.3f} C={C:6.3f} B={B:6.3f} D={D:6.3f}"
                f"  |  A-C={A-C:+.3f}  C-B={C-B:+.3f}  A-D={A-D:+.3f}"
            )

# How often is GCN better than T-GCN under matched mse?
print("\n========== Win counts (mean over seeds): A < C ? ==========")
for ds in ["losloop", "shenzhen"]:
    for adj in ["identity", "physical"]:
        wins = 0
        for ph in [1, 2, 3, 4]:
            m = df[(df.dataset == ds) & (df.ph == ph) & (df.adj_kind == adj)].groupby("arm")["RMSE"].mean()
            if m["A_gcn_mse"] < m["C_tgcn_mse"]:
                wins += 1
        print(f"{ds:8s} {adj:8s}: GCN+mse better in {wins}/4 PHs")

print("\n========== Stage40 NoSpatial reference ==========")
print("Los: GCN 4.88 vs T-GCN 5.25 | SZ: GCN 4.11 vs T-GCN 4.12")
