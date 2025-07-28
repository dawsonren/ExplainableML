#!/usr/bin/env python3
"""ale_mnist_cnn.py

Compute first‑order Accumulated Local Effect (ALE) curves for a trained MNIST
CNN **without Alibi**, using **PyALE** (conda‑forge).

Quick install:
    conda install -c conda-forge pyale tqdm torchvision pytorch pandas

Example (explain digit‑0 probability for 16 highest‑variance channels):
    python ale_mnist_cnn.py --checkpoint mnist_cnn.pt --num-features 16 \
                            --class-idx 0 --output-dir ale_out
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PyALE import ale  # conda‑forge::pyale

# -----------------------------------------------------------------------------
# Network definition (must match checkpoint)
# -----------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Three‑layer CNN for MNIST (1×28×28 → 10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)  # 28→14

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x  # logits


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------


class LastConvFlatten(nn.Module):
    """Return flattened conv3 activations: shape (B, 32*14*14 = 6272)."""

    def __init__(self, cnn: SmallCNN):
        super().__init__()
        self.cnn = cnn
        self._buf: torch.Tensor | None = None
        self.cnn.conv3.register_forward_hook(self._hook)

    def _hook(self, _m, _i, output):
        self._buf = output.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, 28, 28)
        _ = self.cnn(x)  # run full net to trigger hook
        activ = self._buf  # (B, 32, 14, 14)
        b, c, h, w = activ.shape
        return activ.reshape(b, c * h * w)


class FCHead(nn.Module):
    """Fully‑connected part of SmallCNN, exposed as separate module."""

    def __init__(self, cnn: SmallCNN):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = cnn.fc1
        self.fc2 = cnn.fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class SklearnWrapper:
    """Adapter so PyALE can call `.predict()` on a Torch head.

    PyALE may supply a *DataFrame* or *ndarray*; both are accepted.
    """

    def __init__(self, head: nn.Module, device: torch.device, class_idx: int, softmax: bool = True):
        self.head = head.to(device).eval()
        self.device = device
        self.cls = class_idx
        self.softmax = softmax

    @torch.no_grad()
    def predict(self, X):  # X: DataFrame or ndarray of shape (N, 6272)
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32, order="C")
        logits = self.head(torch.from_numpy(X).to(self.device))
        if self.softmax:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs[:, self.cls]
        return logits.cpu().numpy()[:, self.cls]


# -----------------------------------------------------------------------------
# Data‑frame caching
# -----------------------------------------------------------------------------

def cache_flat_features(model: SmallCNN, device: torch.device, sample_frac: float) -> pd.DataFrame:
    """Forward MNIST test set through model and capture flattened conv features."""

    test_ds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())

    if 0 < sample_frac < 1.0:
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(test_ds), generator=rng)[: int(len(test_ds) * sample_frac)]
        test_ds = torch.utils.data.Subset(test_ds, idx.tolist())

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    extractor = LastConvFlatten(model).to(device).eval()
    feats: list[torch.Tensor] = []

    for imgs, _ in tqdm(loader, desc="Caching features"):
        feats.append(extractor(imgs.to(device)).cpu())

    X = torch.cat(feats).numpy()
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


# -----------------------------------------------------------------------------
# CLI & main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ALE plots for SmallCNN via PyALE")
    p.add_argument("--checkpoint", default="mnist_cnn.pt")
    p.add_argument("--output-dir", default="ale_out")
    p.add_argument("--class-idx", type=int, default=0, help="Digit class to explain (0–9)")
    p.add_argument("--grid-size", type=int, default=20, help="Quantile bins for ALE")
    p.add_argument("--num-features", type=int, default=16, help="How many top‑variance features to plot")
    p.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of test set to use")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # 1. Load model
    cnn = SmallCNN().to(device)
    cnn.load_state_dict(torch.load(args.checkpoint, map_location=device))
    cnn.eval()

    # 2. Cache feature matrix (N, 6272)
    X_df = cache_flat_features(cnn, device, args.sample_frac)

    # 3. Rank features by variance
    variances = X_df.var().sort_values(ascending=False)
    top_feats: List[str] = variances.head(args.num_features).index.tolist()

    # 4. Wrap fully‑connected head
    fc_head = FCHead(cnn)
    model_wrap = SklearnWrapper(fc_head, device, class_idx=args.class_idx)

    # 5. Compute ALE curves
    rankings: list[dict] = []
    for feat in top_feats:
        print(f"Computing ALE for {feat} …")
        eff_df = ale(
            X=X_df,
            model=model_wrap,
            feature=[feat],
            grid_size=args.grid_size,
            include_CI=False,
            plot=True,
        )
        eff_df.to_csv(out_dir / f"ale_{feat}.csv", index=False)
        rankings.append({"feature": feat, "var": variances[feat]})

    pd.DataFrame(rankings).to_csv(out_dir / "feature_ranking.csv", index=False)
    print(f"All outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
