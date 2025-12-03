#!/usr/bin/env python3
"""
Attach p_hat to a meta dataset using a frozen model and the training feature pipeline.
"""

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# Allow running directly from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snap_harvester.config import load_config
from snap_harvester.modeling import build_feature_matrix


def add_predictions(meta_path: str, model_path: str, config_path: str, out_path: str) -> None:
    cfg = load_config(config_path)
    meta = pd.read_csv(meta_path)

    model = joblib.load(model_path)
    train_cols = list(model.feature_names_in_)

    X = build_feature_matrix(meta, cfg)
    for col in train_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[train_cols]

    p_hat = model.predict_proba(X)[:, 1]
    meta = meta.copy()
    meta["p_hat"] = p_hat.astype(np.float64)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    print(f"[Save] meta with p_hat -> {out} (n={len(meta)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach p_hat to meta dataset.")
    ap.add_argument("--meta", required=True, help="Input meta CSV")
    ap.add_argument("--model", required=True, help="Frozen model .joblib")
    ap.add_argument("--config", required=True, help="Config YAML (for feature pipeline)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    add_predictions(
        meta_path=args.meta,
        model_path=args.model,
        config_path=args.config,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
