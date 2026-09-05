#!/usr/bin/env python
"""
DAGMA Threshold Sensitivity Audit — PRACTICAL VERSION
Runs DAGMA with reduced iterations for feasibility, and with a small data subsample.
Also does comprehensive analysis of existing W_est files.
"""
import sys, os, numpy as np, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data.functions import load_features
from dagma.linear import DagmaLinear

OUTPUT_DIR = "results/dagma_threshold_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Analyze existing W_est files in exhaustive detail
# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("PART 1: COMPREHENSIVE ANALYSIS OF EXISTING W_est FILES")
print("="*70)

existing_results = {}
for dataset in ["shenzhen", "losloop"]:
    existing_results[dataset] = {}
    for ph in [1, 2, 3, 4]:
        fname = f"data/W_est_{dataset}_pre_len{ph}.npy"
        W = np.load(fname)
        print(f"\n--- {dataset} PH={ph}: shape={W.shape} ---")
        
        # Analyze per-slice
        for i in range(W.shape[2]):
            ws = W[:, :, i]
            nz = np.count_nonzero(ws)
            pos = np.sum(ws > 0)
            neg = np.sum(ws < 0)
            abs_nz = np.abs(ws[ws != 0])
            
            print(f"  Slice {i}: nonzero={nz}, pos={pos}, neg={neg}")
            if nz > 0:
                print(f"    abs(W) range: [{abs_nz.min():.6f}, {abs_nz.max():.6f}]")
                print(f"    abs(W) mean: {abs_nz.mean():.6f}, median: {np.median(abs_nz):.6f}")
                # Show all nonzero weights
                rows, cols = np.where(ws != 0)
                for r, c in zip(rows, cols):
                    print(f"    W[{r},{c}] = {ws[r,c]:.6f}")
            
            existing_results[dataset][f"ph{ph}_slice{i}"] = {
                "nonzero": int(nz),
                "positive": int(pos),
                "negative": int(neg),
                "abs_nz_min": float(abs_nz.min()) if nz > 0 else 0,
                "abs_nz_max": float(abs_nz.max()) if nz > 0 else 0,
            }

# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Run DAGMA with w_threshold=0 on reduced data
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: DAGMA WITH w_threshold=0 (reduced iterations)")
print("="*70)

def extract_input(feat_path, pre_len):
    """Replicate exact DAGMA input construction from the codebase."""
    feat = load_features(feat_path)
    max_val = np.max(feat)
    feat_norm = feat / max_val
    train_size = int(feat_norm.shape[0] * 0.8)
    train_data = feat_norm[:train_size]
    train_X = []
    for i in range(len(train_data) - 1 - pre_len):
        train_X.append(np.array(train_data[i:i+1]))
    train_X = np.array(train_X)
    data = np.array([x[0] for x in train_X])
    return data[0::pre_len]  # X = data[i::pre_len] for i=0

dagma_results = {}
for dataset_name, feat_path, lambda1 in [
    ("shenzhen", "data/sz_speed.csv", 0.01),
    ("losloop", "data/los_speed.csv", 0.02),
]:
    X_full = extract_input(feat_path, 1)
    print(f"\n--- {dataset_name}: full X shape={X_full.shape} ---")
    
    # Use a subsample for speed (first 200 rows)
    X_sub = X_full[:200]
    print(f"  Subsample X shape: {X_sub.shape}")
    
    # Run DAGMA with w_threshold=0 and reduced iterations
    # Original: T=5, warm_iter=30000, max_iter=60000
    # Reduced: T=3, warm_iter=3000, max_iter=6000
    model = DagmaLinear(loss_type='l2')
    t0 = time.time()
    W_raw = model.fit(
        X_sub,
        lambda1=lambda1,
        w_threshold=0.0,  # NO thresholding
        T=3,
        warm_iter=3000,
        max_iter=6000,
        checkpoint=3000,
    )
    elapsed = time.time() - t0
    print(f"  DAGMA completed in {elapsed:.1f}s")
    
    # Save raw matrix
    out_path = os.path.join(OUTPUT_DIR, f"W_raw_{dataset_name}_pre_len1_thresh0_subsample.npy")
    np.save(out_path, W_raw)
    
    # Comprehensive analysis
    abs_all = np.abs(W_raw)
    nz_mask = W_raw != 0
    abs_nz = abs_all[nz_mask]
    
    print(f"\n  Raw W analysis ({dataset_name}):")
    print(f"    Shape: {W_raw.shape}")
    print(f"    Total entries: {W_raw.size}")
    print(f"    Exact zeros: {np.sum(~nz_mask)}")
    print(f"    Nonzero: {np.sum(nz_mask)}")
    print(f"    Positive: {np.sum(W_raw > 0)}")
    print(f"    Negative: {np.sum(W_raw < 0)}")
    
    if len(abs_nz) > 0:
        print(f"    |W| nonzero range: [{abs_nz.min():.6f}, {abs_nz.max():.6f}]")
        print(f"    |W| nonzero mean: {abs_nz.mean():.6f}")
        print(f"    |W| nonzero median: {np.median(abs_nz):.6f}")
        
        print(f"\n    Threshold sensitivity (|W| >= t):")
        for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            count = int(np.sum(abs_all >= t))
            print(f"      |W| >= {t:.3f}: {count:>6} edges")
        
        print(f"\n    Positive threshold sensitivity (W > t):")
        for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
            count = int(np.sum(W_raw > t))
            print(f"      W > {t:.3f}: {count:>6} edges")
        
        # Top 30 entries by absolute magnitude
        flat_idx = np.argsort(np.abs(W_raw).ravel())[::-1]
        print(f"\n    Top 30 entries by |W|:")
        print(f"    {'rank':>4} {'i':>5} {'j':>5} {'W_ij':>10} {'|W|':>10} {'sign':>6}")
        for rank, idx in enumerate(flat_idx[:30]):
            i, j = divmod(idx, W_raw.shape[1])
            w = W_raw[i, j]
            if w == 0:
                break
            sign = "+" if w > 0 else "-"
            print(f"    {rank+1:>4} {i:>5} {j:>5} {w:>10.6f} {abs(w):>10.6f} {sign:>6}")
        
        # Quantiles
        print(f"\n    |W| quantiles (nonzero):")
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = float(np.percentile(abs_nz, q))
            print(f"      {q:>3}%: {val:.6f}")
    
    dagma_results[dataset_name] = {
        "shape": list(W_raw.shape),
        "nonzero": int(np.sum(nz_mask)),
        "positive": int(np.sum(W_raw > 0)),
        "negative": int(np.sum(W_raw < 0)),
        "elapsed": elapsed,
        "subsample_size": X_sub.shape[0],
    }

# Save all results
json_path = os.path.join(OUTPUT_DIR, "threshold_audit_results.json")
with open(json_path, "w") as f:
    json.dump({"existing": existing_results, "dagma_w0": dagma_results}, f, indent=2, default=str)
print(f"\nResults saved to {json_path}")
print("DONE.")
