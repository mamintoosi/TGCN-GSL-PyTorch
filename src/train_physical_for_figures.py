#!/usr/bin/env python3
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN


def seqs(d, T=12, PH=1):
    X, Y = [], []
    for i in range(len(d) - T - PH):
        X.append(d[i : i + T])
        Y.append(d[i + T : i + T + PH])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train_physical(ds, feat_path, adj_path, out_name):
    print("load", ds, flush=True)
    feat = np.array(pd.read_csv(ROOT / feat_path), dtype=np.float32)
    adj = np.array(pd.read_csv(ROOT / adj_path, header=None), dtype=np.float32)
    split = int(0.8 * len(feat))
    mx = float(feat[:split].max())
    train = feat[:split] / mx
    test = feat[split:] / mx
    trX, trY = seqs(train)
    teX, teY = seqs(test)
    print("shapes", trX.shape, teX.shape, flush=True)
    torch.manual_seed(42)
    model = TGCN(adj=adj, hidden_dim=64)
    task = SupervisedForecastTask(
        model=model,
        loss="mse_with_regularizer",
        pre_len=1,
        learning_rate=0.001,
        weight_decay=0.0001,
        feat_max_val=mx,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(trX), torch.tensor(trY)),
        batch_size=128,
        shuffle=True,
    )
    opt = task.configure_optimizer()
    t0 = time.time()
    train_losses = []
    for e in range(50):
        task.model.train()
        if task.regressor is not None:
            task.regressor.train()
        ep_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = task.training_step((xb, yb))
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(ep_losses)))
        if e % 10 == 0:
            print("epoch", e, "t", round(time.time() - t0, 1), flush=True)
    task.model.eval()
    if task.regressor is not None:
        task.regressor.eval()
    yp = []
    with torch.no_grad():
        for xb, yb in torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(teX), torch.tensor(teY)),
            batch_size=len(teX),
            shuffle=False,
        ):
            pred = task.forward(xb).transpose(1, 2)
            yp.append(pred.cpu().numpy())
    yp = np.concatenate(yp, 0)
    out = ROOT / "results" / "stage26_checkpoint" / out_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_true.npy", teY)
    np.save(out / "y_pred.npy", yp)
    # checkpoint-style histories for training-loss figure
    hist = {
        "train_losses": train_losses,
        "best_epoch": int(np.argmin(train_losses)),
        "best_loss": float(min(train_losses)),
        "train_time_s": round(time.time() - t0, 1),
        "max_epochs": 50,
        "seed": 42,
    }
    import json

    (out / "train_loss_history.json").write_text(json.dumps(hist), encoding="utf-8")
    print("saved", out, teY.shape, yp.shape, "losses", len(train_losses), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["losloop", "shenzhen", "both"], default="losloop")
    a = p.parse_args()
    if a.dataset in ("losloop", "both"):
        train_physical("losloop", "data/los_speed.csv", "data/los_adj.csv", "los_ph1_seed42_physical")
    if a.dataset in ("shenzhen", "both"):
        train_physical("shenzhen", "data/sz_speed.csv", "data/sz_adj.csv", "sz_ph1_seed42_physical")
