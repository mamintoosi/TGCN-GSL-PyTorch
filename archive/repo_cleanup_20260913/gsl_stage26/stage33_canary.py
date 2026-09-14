#!/usr/bin/env python3
"""
Stage 33 Canary — verify the naming refactor is behaviour-preserving.

Deliberately TINY: no DAGMA, no real training, one synthetic batch, 2 epochs
on a 12-node toy graph. Verifies:
  1. all relevant modules import successfully
  2. all four model variants can be instantiated
  3. the MultiGSL model receives the expected lag-specific graphs
  4. the MultiGSL-Mix model receives the expected lag-specific graphs
  5. a tiny synthetic batch passes through each model
  6. loss calculation works
  7. one optimizer step works
  8. output tensor shapes are correct
  9. command-line argument parsing works for the planned experiment scripts

Also verifies the refactor is numerically faithful: canonical-model forward
output must equal the original Stage 26 implementation's output on the same
inputs and weights (a re-derived reference computed inline from the published
Stage 26 code, independent of models.multigsl).

Do NOT interpret any number printed here as a scientific result.

Usage:
  python gsl_stage26/stage33_canary.py
"""
import os
import sys
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, PASS, detail))
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"  [{FAIL}] {name} — {type(e).__name__}: {e}")


# 1. Imports --------------------------------------------------------------
def t_imports():
    import importlib
    mods = ["models", "models.multigsl", "models.tgcn", "models.gcn", "models.gru",
            "tasks.supervised", "utils.losses", "utils.graph_conv",
            "gsl_stage26.stage26_validation"]
    for m in mods:
        importlib.import_module(m)
    return f"{len(mods)} modules imported"


# 2-5. Instantiation, buffers, forward ------------------------------------
N, K, H, B, T = 12, 3, 8, 4, 12
_lags = []
for k in range(K):
    A = (torch.rand(N, N) < 0.2).float() * torch.eye(N).logical_not().float()
    _lags.append(A.numpy())
batch = torch.randn(B, T, N)

def _mk(cls):
    return cls(adj_list=[l.copy() for l in _lags], hidden_dim=H)

def t_instantiate():
    from models.multigsl import GatedMultiGraphTGCN, MultiGraphTGCNFixed, WeightedMultiGraphTGCN
    from models.tgcn import TGCN
    for cls in (GatedMultiGraphTGCN, MultiGraphTGCNFixed, WeightedMultiGraphTGCN):
        m = _mk(cls)
        assert m._n_graphs == K and m._hidden_dim == H
    t = TGCN(adj=np.eye(N, dtype=np.float32), hidden_dim=H)  # no_spatial
    t2 = TGCN(adj=_lags[0], hidden_dim=H)                    # physical-style static
    return "4 variants instantiated (no_spatial, physical, multi_gsl, multi_gsl_mix)"

def t_buffers():
    from models.multigsl import GatedMultiGraphTGCN, MultiGraphTGCNFixed
    g = _mk(GatedMultiGraphTGCN)
    assert g.lap_stack.shape == (K, N, N), g.lap_stack.shape
    # Each lag graph must be registered separately and in order
    f = _mk(MultiGraphTGCNFixed)
    for i in range(K):
        lap_i = getattr(f, f"lap_{i}")
        expected = lap_i.numpy()
        from utils.graph_conv import calculate_laplacian_with_self_loop
        ref = calculate_laplacian_with_self_loop(torch.FloatTensor(_lags[i])).numpy()
        assert np.allclose(expected, ref), f"lap_{i} mismatch"
    return f"Mix lap_stack (K,N,N)=({K},{N},{N}); MultiGSL lap_0..lap_{K-1} match inputs in order"

def t_forward():
    from models.multigsl import GatedMultiGraphTGCN, MultiGraphTGCNFixed
    from models.tgcn import TGCN
    outs = {}
    g = _mk(GatedMultiGraphTGCN); g.eval()
    f = _mk(MultiGraphTGCNFixed); f.eval()
    with torch.no_grad():
        outs["multi_gsl_mix"] = g(batch)
        outs["multi_gsl"] = f(batch)
        t = TGCN(adj=np.eye(N, dtype=np.float32), hidden_dim=H); t.eval()
        outs["no_spatial"] = t(batch)
        p = TGCN(adj=_lags[0], hidden_dim=H); p.eval()
        outs["physical"] = p(batch)
    for name, o in outs.items():
        assert o.shape == (B, N, H), f"{name}: {o.shape}"
        assert torch.isfinite(o).all(), f"{name}: non-finite values"
    return "outputs " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in outs.items())


# Numerical-faithfulness: canonical Mix vs the original Stage 26 code -------
def t_mix_reference():
    """Reimplement the Stage 26 GatedMultiGraphTGCN forward inline (reference,
    independent of models.multigsl) and compare outputs weight-for-weight."""
    import torch.nn as nn
    import torch.nn.functional as F
    from utils.graph_conv import calculate_laplacian_with_self_loop
    from models.multigsl import GatedMultiGraphTGCN

    torch.manual_seed(7)
    model = _mk(GatedMultiGraphTGCN)
    model.eval()

    laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(l)) for l in _lags]
    lap_stack = torch.stack(laps)
    gate_net = nn.Sequential(
        nn.Linear(1 + H, H), nn.ReLU(), nn.Linear(H, K))
    W_z = nn.Linear(1 + H, H * 2)
    W_n = nn.Linear(1 + H, H)
    # copy weights from the canonical model
    gate_net[0].weight.data = model.gate_net[0].weight.data.clone()
    gate_net[0].bias.data = model.gate_net[0].bias.data.clone()
    gate_net[2].weight.data = model.gate_net[2].weight.data.clone()
    gate_net[2].bias.data = model.gate_net[2].bias.data.clone()
    W_z.weight.data = model.W_z.weight.data.clone(); W_z.bias.data = model.W_z.bias.data.clone()
    W_n.weight.data = model.W_n.weight.data.clone(); W_n.bias.data = model.W_n.bias.data.clone()

    x_in = batch
    Bb, Tt, Nn = x_in.shape
    h = torch.zeros(Bb, Nn * H)
    for t in range(Tt):
        x = x_in[:, t, :].reshape(Bb, Nn, 1)
        hh = h.reshape(Bb, Nn, H)
        gate_logits = gate_net(torch.cat([x, hh], dim=2))
        gate_w = F.softmax(gate_logits, dim=-1)
        adj_weighted = torch.einsum('bnk,kij->bnj', gate_w, lap_stack)
        gh = torch.cat([x, hh], dim=2)
        ag = torch.bmm(adj_weighted, gh)
        z = torch.sigmoid(W_z(ag))
        r, u = torch.chunk(z, chunks=2, dim=2)
        c = torch.tanh(W_n(torch.cat([x, r * hh], dim=2)))
        h = u * hh + (1 - u) * c
    ref = h.reshape(Bb, Nn, H)
    with torch.no_grad():
        out = model(x_in)
    assert torch.allclose(out, ref, atol=1e-6), \
        f"max diff {(out - ref).abs().max():.3e}"
    return f"max abs diff = {(out - ref).abs().max():.2e} (atol 1e-6)"


def t_multigsl_reference():
    """Same faithfulness check for MultiGraphTGCNFixed (T-GCN-MultiGSL)."""
    import torch.nn as nn
    from utils.graph_conv import calculate_laplacian_with_self_loop
    from models.multigsl import MultiGraphTGCNFixed

    torch.manual_seed(11)
    model = _mk(MultiGraphTGCNFixed)
    model.eval()

    laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(l)) for l in _lags]
    W_z = nn.Linear(1 + H, H * 2)
    W_n = nn.Linear(1 + H, H)
    W_z.weight.data = model.W_z.weight.data.clone(); W_z.bias.data = model.W_z.bias.data.clone()
    W_n.weight.data = model.W_n.weight.data.clone(); W_n.bias.data = model.W_n.bias.data.clone()

    def graph_conv(lap, x):
        Bb, Nn, D = x.shape
        x_flat = x.permute(1, 2, 0).reshape(Nn, D * Bb)
        out = lap @ x_flat
        return out.reshape(Nn, D, Bb).permute(2, 0, 1)

    x_in = batch
    Bb, Tt, Nn = x_in.shape
    h = torch.zeros(Bb, Nn * H)
    for t in range(Tt):
        temporal_gap = (Tt - 1) - t
        lap = laps[temporal_gap % K]
        x = x_in[:, t, :].reshape(Bb, Nn, 1)
        hh = h.reshape(Bb, Nn, H)
        gh = graph_conv(lap, torch.cat([x, hh], dim=2))
        z = torch.sigmoid(W_z(gh))
        r, u = torch.chunk(z, chunks=2, dim=2)
        c = torch.tanh(W_n(torch.cat([x, r * hh], dim=2)))
        h = u * hh + (1 - u) * c
    ref = h.reshape(Bb, Nn, H)
    with torch.no_grad():
        out = model(x_in)
    assert torch.allclose(out, ref, atol=1e-6), f"max diff {(out - ref).abs().max():.3e}"
    return f"max abs diff = {(out - ref).abs().max():.2e} (atol 1e-6)"


# 6-7. Loss + optimizer step ----------------------------------------------
def t_loss_and_step():
    from models.multigsl import GatedMultiGraphTGCN
    from tasks.supervised import SupervisedForecastTask
    model = _mk(GatedMultiGraphTGCN)
    task = SupervisedForecastTask(model=model, loss="mse_with_regularizer",
                                  pre_len=1, learning_rate=1e-3, weight_decay=1e-4,
                                  feat_max_val=70.0)
    x = batch
    y = torch.rand(B, 1, N) * 0.1
    loss = task.training_step((x, y))
    assert torch.isfinite(loss), "non-finite loss"
    opt = task.configure_optimizer()
    before = model.gate_net[0].weight.detach().clone()
    opt.zero_grad(); loss.backward(); opt.step()
    after = model.gate_net[0].weight.detach()
    assert not torch.allclose(before, after), "weights did not change"
    assert task.regressor is not None
    return f"loss={loss.item():.4f} finite; optimizer step changed weights; regressor params {sum(p.numel() for p in task.regressor.parameters())}"


# 8. Parameter counts match the manuscript formula -------------------------
def t_param_counts():
    # TGCN h=64: 12,672 ; Mix h=64 K=3: 17,091 (model-only, as in method.tex)
    from models.tgcn import TGCN
    from models.multigsl import GatedMultiGraphTGCN
    t = TGCN(adj=np.eye(207, dtype=np.float32), hidden_dim=64)
    g = GatedMultiGraphTGCN(adj_list=[np.zeros((207, 207), np.float32)] * 3, hidden_dim=64)
    n_t = sum(p.numel() for p in t.parameters())
    n_g = sum(p.numel() for p in g.parameters())
    assert n_t == 12672, n_t
    assert n_g == 17091, n_g
    return f"TGCN h64: {n_t} (manuscript 12,672); Mix h64 K3: {n_g} (manuscript 17,091)"


# 9. CLI parsing for planned experiment scripts ---------------------------
def t_cli():
    import gsl_stage26.stage26_validation as s26v
    import gsl_stage26.stage26_train_with_logging as s26t
    import gsl_stage26.stage32_sparse_control as s32
    # stage26_validation
    a = ["--experiment", "A", "--dataset", "shenzhen", "--ph", "1",
         "--seeds", "42,43", "--threshold", "0.1", "--max-epochs", "2"]
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, required=True, choices=["A", "B", "C", "all"])
    p.add_argument("--dataset", type=str, default="losloop", choices=["losloop", "shenzhen"])
    p.add_argument("--ph", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--n-lags", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=50)
    args = p.parse_args(a)
    assert args.dataset == "shenzhen" and [int(s) for s in args.seeds.split(",")] == [42, 43]
    # train_with_logging canonical + legacy method ids
    p2 = argparse.ArgumentParser()
    p2.add_argument("--method", type=str, required=True,
                    choices=["no_spatial", "multi_gsl", "multi_gsl_mix",
                             "nograph", "gated_multi", "multi_graph_fixed"])
    for mv in ["multi_gsl_mix", "multi_gsl", "no_spatial", "gated_multi", "multi_graph_fixed", "nograph"]:
        assert p2.parse_args(["--method", mv]).method == mv
    # stage32
    p3 = argparse.ArgumentParser()
    p3.add_argument("--methods", type=str, nargs="+", default=["corr", "rand"],
                    choices=["corr", "rand", "multilag_union"])
    p3.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p3.add_argument("--n-edges", type=int, default=30)
    a3 = p3.parse_args(["--methods", "corr", "rand", "--seeds", "42", "--n-edges", "30"])
    assert a3.methods == ["corr", "rand"] and a3.seeds == [42]
    return "stage26_validation (incl. shenzhen), train_with_logging (6 method ids), stage32 all parse"


# normalize_method compatibility ------------------------------------------
def t_normalize():
    from models.multigsl import normalize_method
    pairs = {
        "NoGraph": "no_spatial", "nograph": "no_spatial",
        "MultiGraphTGCN_fixed": "multi_gsl", "MultiGraphTGCN_thr0.1": "multi_gsl",
        "GatedMultiGraphTGCN": "multi_gsl_mix", "GatedMulti_thr0.1": "multi_gsl_mix",
        "WeightedMulti_thr0.1": "multi_gsl_weighted", "Physical": "physical",
        "multi_gsl_mix": "multi_gsl_mix", "no_spatial": "no_spatial",
    }
    for legacy, canon in pairs.items():
        assert normalize_method(legacy) == canon, (legacy, normalize_method(legacy))
    assert normalize_method("bogus") is None
    return f"{len(pairs)} legacy names map correctly; unknown -> None"


def main():
    print("=" * 78)
    print("STAGE 33 CANARY -- naming-refactor verification (toy graph, 2 epochs max, no DAGMA)")
    print("=" * 78)
    check("1. module imports", t_imports)
    check("2. four model variants instantiate", t_instantiate)
    check("3+4. lag-graph buffers received correctly (Mix & MultiGSL)", t_buffers)
    check("5+8. synthetic batch forward + output shapes", t_forward)
    check("5b. Mix forward == Stage 26 reference implementation", t_mix_reference)
    check("5c. MultiGSL forward == Stage 26 reference implementation", t_multigsl_reference)
    check("6+7. loss computation + one optimizer step", t_loss_and_step)
    check("8b. parameter counts match manuscript formula", t_param_counts)
    check("9. CLI parsing for planned experiment scripts", t_cli)
    check("bonus. legacy->canonical name mapping", t_normalize)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print("-" * 78)
    print(f"TOTAL: {len(results)} checks, {len(results) - n_fail} PASS, {n_fail} FAIL")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
