#!/usr/bin/env python3
"""
Standalone DAGMA verification script.
Compares fresh DAGMA execution against the old cached W_est files.
Times each run to establish actual DAGMA runtime.

Usage:
    python gsl_clean/verify_dagma.py --dataset sz
    python gsl_clean/verify_dagma.py --dataset los
    python gsl_clean/verify_dagma.py --dataset both
    python gsl_clean/verify_dagma.py --dataset sz --max-iter 1000 --warm-iter 1000  # Quick test
"""
import argparse, json, os, sys, time, copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dagma.linear import DagmaLinear
from utils.data.functions import load_features, generate_dataset

# ── Datasets ──────────────────────────────────────────────────────────────
DATASETS = {
    "sz":  {"feat": "data/sz_speed.csv", "name": "shenzhen",  "lambda1": 0.01, "N": 156},
    "los": {"feat": "data/los_speed.csv", "name": "losloop",   "lambda1": 0.02, "N": 207},
}

SEED = 42


def set_seeds(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(feat_path, seq_len, pre_len, split_ratio=0.8):
    """Reproduce the exact preprocessing from the old code."""
    feat = load_features(feat_path, dtype=np.float32)
    train_X, _, _, _ = generate_dataset(feat, seq_len=seq_len, pre_len=pre_len,
                                         split_ratio=split_ratio, normalize=True)
    # Old code: data = np.array([x[0] for x in self.train_data])
    # self.train_data = np.array([x[0].numpy() for x in train_dataset])
    # So train_X[:, 0, :] is the same thing
    data = train_X[:, 0, :]  # (M, N)
    return data, feat.shape


def w_stats(W, label=""):
    """Print comprehensive W statistics."""
    absW = np.abs(W)
    N = W.shape[0]
    nz = np.count_nonzero(W)
    diag_nz = int(np.count_nonzero(np.diag(W)))
    off_diag_nz = nz - diag_nz
    pos = int(np.sum(W > 0))
    neg = int(np.sum(W < 0))
    
    stats = {
        "nonzero": nz, "off_diag_nonzero": off_diag_nz, "diag_nonzero": diag_nz,
        "positive": pos, "negative": neg,
        "density": nz / (N * N),
        "min": float(np.min(W)), "max": float(np.max(W)),
        "mean": float(np.mean(W)), "std": float(np.std(W)),
        "abs_gte_0001": int(np.sum(absW >= 0.001)),
        "abs_gte_001": int(np.sum(absW >= 0.01)),
        "abs_gte_01": int(np.sum(absW >= 0.1)),
        "abs_gte_03": int(np.sum(absW >= 0.3)),
    }
    
    print(f"\n  --- W Statistics: {label} ---")
    print(f"    shape:            {W.shape}")
    print(f"    nonzero:          {nz}  (off-diag: {off_diag_nz}, diag: {diag_nz})")
    print(f"    positive/negative: {pos} / {neg}")
    print(f"    density:          {stats['density']:.6f}")
    print(f"    range:            [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"    mean ± std:       {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"    |W| >= 0.001:    {stats['abs_gte_0001']}")
    print(f"    |W| >= 0.01:     {stats['abs_gte_001']}")
    print(f"    |W| >= 0.1:      {stats['abs_gte_01']}")
    print(f"    |W| >= 0.3:      {stats['abs_gte_03']}")
    return stats


def run_dagma_timed(X, lambda1, w_threshold=0.3, warm_iter=30000, max_iter=60000, T=5, label=""):
    """Run fresh DAGMA with explicit parameters and timing."""
    N = X.shape[1]
    M = X.shape[0]
    total_iters = (T - 1) * warm_iter + max_iter
    
    print(f"\n{'='*70}")
    print(f"FRESH DAGMA: {label}")
    print(f"{'='*70}")
    print(f"  Input X shape:    ({M}, {N})")
    print(f"  lambda1:          {lambda1}")
    print(f"  w_threshold:      {w_threshold}")
    print(f"  warm_iter:        {warm_iter}")
    print(f"  max_iter:         {max_iter}")
    print(f"  T:                {T}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Loss type:        l2")
    print(f"  CUDA available:   False (DAGMA uses numpy)")
    
    # Validate input
    assert np.isfinite(X).all(), "Input has non-finite values!"
    
    t0 = time.perf_counter()
    model = DagmaLinear(loss_type='l2')
    W = model.fit(
        X, lambda1=lambda1, w_threshold=w_threshold,
        warm_iter=warm_iter, max_iter=max_iter, T=T,
    )
    elapsed = time.perf_counter() - t0
    
    # Validate output
    assert W.shape == (N, N), f"Unexpected W shape: {W.shape}"
    assert np.isfinite(W).all(), "Output has non-finite values!"
    
    iters_per_sec = total_iters / elapsed if elapsed > 0 else 0
    print(f"\n  Runtime: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"  Speed:   {iters_per_sec:.1f} it/s")
    
    stats = w_stats(W, label)
    stats["runtime_seconds"] = round(elapsed, 2)
    stats["runtime_minutes"] = round(elapsed / 60, 2)
    stats["iters_per_sec"] = round(iters_per_sec, 1)
    return W, stats


def load_cached_west(dataset_key, pre_len):
    """Load old cached W_est file."""
    name = DATASETS[dataset_key]["name"]
    path = f"data/W_est_{name}_pre_len{pre_len}.npy"
    if os.path.exists(path):
        W_est_all = np.load(path)
        print(f"\n  Loaded cached: {path}  shape={W_est_all.shape}")
        return W_est_all
    print(f"\n  NOT FOUND: {path}")
    return None


def compare_west(W_new, W_cached, pre_len, dataset_key):
    """Compare fresh DAGMA result against old cached result."""
    if W_cached is None:
        print(f"  Cannot compare: cached W_est not found for PH={pre_len}")
        return
    
    N = DATASETS[dataset_key]["N"]
    
    # The old code stores W_est_all as (N, N, pre_len) and takes W_est_all[:,:,i]
    if W_cached.ndim == 3:
        W_old = W_cached[:, :, pre_len - 1]  # 0-indexed
        print(f"  W_cached[:, :, {pre_len-1}] shape: {W_old.shape}")
    else:
        W_old = W_cached
    
    # Before thresholding (W_old is already thresholded)
    # Compare the thresholded versions
    W_new_binary = (W_new > 0).astype(int)
    W_old_binary = (W_old > 0).astype(int)
    
    # Count overlaps
    both_nonzero = np.sum((W_new_binary > 0) & (W_old_binary > 0))
    only_new = np.sum((W_new_binary > 0) & (W_old_binary == 0))
    only_old = np.sum((W_new_binary == 0) & (W_old_binary > 0))
    
    print(f"\n  --- Comparison: Fresh vs Cached (PH={pre_len}) ---")
    print(f"    Fresh nonzero:  {np.count_nonzero(W_new)}")
    print(f"    Cached nonzero: {np.count_nonzero(W_old)}")
    print(f"    Both nonzero:   {both_nonzero}")
    print(f"    Only fresh:     {only_new}")
    print(f"    Only cached:    {only_old}")
    
    # Check if matrices are exactly equal
    if W_new.shape == W_old.shape:
        exact_match = np.allclose(W_new, W_old, atol=1e-10)
        print(f"    Exact match:    {exact_match}")
        
        # Even if not exact, check binary structure match
        structure_match = np.array_equal(W_new_binary, W_old_binary)
        print(f"    Structure match: {structure_match}")


def estimate_time_from_short_run(short_time, short_iters, target_iters):
    """Estimate time for full run based on short run."""
    return short_time * (target_iters / short_iters)


def main():
    parser = argparse.ArgumentParser(description="Standalone DAGMA verification")
    parser.add_argument("--dataset", type=str, default="sz", choices=["sz", "los", "both"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--w-threshold", type=float, default=0.3)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    
    # DAGMA iteration parameters (for quick testing)
    parser.add_argument("--warm-iter", type=float, default=30000)
    parser.add_argument("--max-iter", type=float, default=60000)
    parser.add_argument("--T", type=int, default=5)
    
    # Quick test mode
    parser.add_argument("--quick", action="store_true",
                        help="Use 1000 iterations for quick timing estimate")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare cached W_est files, don't run DAGMA")
    
    args = parser.parse_args()
    
    if args.quick:
        args.warm_iter = 1000
        args.max_iter = 2000
        args.T = 3
    
    set_seeds(SEED)
    
    print("=" * 70)
    print("STANDALONE DAGMA VERIFICATION")
    print("=" * 70)
    print(f"  Seed:         {SEED}")
    print(f"  DAGMA ver:    1.1.1 (installed)")
    print(f"  Note:         requirements.txt says 0.1.0 but that never existed on PyPI")
    print(f"  Iterations:   T={args.T}, warm={int(args.warm_iter)}, max={int(args.max_iter)}")
    print(f"  Total iters:  {(args.T-1)*int(args.warm_iter)+int(args.max_iter)}")
    
    datasets_to_run = []
    if args.dataset in ("sz", "both"):
        datasets_to_run.append(("sz", DATASETS["sz"]))
    if args.dataset in ("los", "both"):
        datasets_to_run.append(("los", DATASETS["los"]))
    
    all_results = []
    
    for ds_key, ds_cfg in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_cfg['name']} ({ds_key})")
        print(f"{'#'*70}")
        
        data, raw_shape = prepare_data(ds_cfg["feat"], args.seq_len, pre_len=1,
                                        split_ratio=args.split_ratio)
        print(f"  Raw shape:     {raw_shape}")
        print(f"  Train data:    {data.shape}  (M, N)")
        
        lambda1 = ds_cfg["lambda1"]
        
        for ph in args.horizons:
            X = data[0::ph].copy()
            print(f"\n  --- PH={ph}: X = data[0::{ph}], shape={X.shape} ---")
            
            if not args.compare_only:
                # Run fresh DAGMA
                label = f"{ds_key} PH={ph}"
                W_fresh, stats = run_dagma_timed(
                    X, lambda1=lambda1, w_threshold=args.w_threshold,
                    warm_iter=int(args.warm_iter), max_iter=int(args.max_iter),
                    T=args.T, label=label,
                )
            else:
                W_fresh = None
                stats = {}
            
            # Load and compare with cached
            W_cached = load_cached_west(ds_key, ph)
            if W_fresh is not None and W_cached is not None:
                compare_west(W_fresh, W_cached, ph, ds_key)
            
            # If quick mode, estimate full runtime
            if args.quick and stats.get("runtime_seconds", 0) > 0:
                full_iters = 4 * 30000 + 60000
                short_iters = (args.T-1)*int(args.warm_iter)+int(args.max_iter)
                est = estimate_time_from_short_run(stats["runtime_seconds"], short_iters, full_iters)
                print(f"\n  *** Estimated time for default config ({full_iters} iters): {est:.0f}s ({est/60:.1f} min) ***")
            
            result = {
                "dataset": ds_cfg["name"], "dataset_key": ds_key, "ph": ph,
                "lambda1": lambda1, "w_threshold": args.w_threshold,
                "warm_iter": int(args.warm_iter), "max_iter": int(args.max_iter),
                "T": args.T, "N": ds_cfg["N"], "M": X.shape[0],
                "fresh_stats": stats,
            }
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<10} {'PH':>3} {'N':>4} {'M':>6} {'NZ':>6} {'Runtime':>10} {'Speed':>10}")
    print("-" * 90)
    for r in all_results:
        s = r.get("fresh_stats", {})
        rt = s.get("runtime_seconds", "N/A")
        sp = s.get("iters_per_sec", "N/A")
        nz = s.get("nonzero", "N/A")
        rt_str = f"{rt:.1f}s" if isinstance(rt, (int, float)) else rt
        sp_str = f"{sp:.0f}/s" if isinstance(sp, (int, float)) else sp
        print(f"{r['dataset']:<10} {r['ph']:>3} {r['N']:>4} {r['M']:>6} {nz:>6} {rt_str:>10} {sp_str:>10}")
    
    # Save
    os.makedirs("results/dagma_fresh", exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [{k: v for k, v in r.items()} for r in all_results],
    }
    with open("results/dagma_fresh/verify_dagma_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to: results/dagma_fresh/verify_dagma_report.json")


if __name__ == "__main__":
    main()
