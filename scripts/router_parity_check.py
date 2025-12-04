from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure local package import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snap_harvester.config import load_config  # type: ignore
from snap_harvester.live.router import LiveRouter  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Router parity check vs research routed tape.")
    parser.add_argument("--config", default="configs/snap_harvester_live_btc.yaml")
    parser.add_argument(
        "--meta",
        default="results/meta/snap_meta_with_p_hat_2025_BTC_agg.csv",
        help="Path to meta dataset or events with full feature columns",
    )
    parser.add_argument("--routed", default="results/meta/snap_routed_tape_2025_BTC_agg.csv")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta_path = Path(args.meta)
    routed_path = Path(args.routed)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta dataset not found: {meta_path}")
    if not routed_path.exists():
        raise FileNotFoundError(f"Routed tape not found: {routed_path}")

    meta_df = pd.read_csv(meta_path)
    time_col = cfg["data"].get("events_time_col", "timestamp")
    sym_col = cfg["data"].get("events_symbol_col", "symbol")
    if "event_id" not in meta_df.columns:
        meta_df[time_col] = pd.to_datetime(meta_df[time_col], utc=True, errors="coerce")
        meta_df = meta_df.sort_values(time_col).reset_index(drop=True)
        meta_df["event_id"] = [
            f"{row.get(sym_col,'BTCUSDT')}-{pd.to_datetime(row[time_col]).isoformat()}-{i}"
            for i, row in meta_df.iterrows()
        ]
    routed_df = pd.read_csv(routed_path)
    merged = meta_df.set_index("event_id").join(
        routed_df.set_index("event_id"), how="inner", lsuffix="_meta", rsuffix="_routed"
    )
    if merged.empty:
        # Fallback join on timestamp + symbol if event_id schemas differ
        ts_col = time_col
        meta_df[ts_col] = pd.to_datetime(meta_df[ts_col], utc=True, errors="coerce")
        routed_df[ts_col] = pd.to_datetime(routed_df["timestamp"], utc=True, errors="coerce")
        merged = (
            meta_df.set_index([sym_col, ts_col])
            .join(routed_df.set_index(["symbol", ts_col]), how="inner", lsuffix="_meta", rsuffix="_routed")
            .reset_index()
            .set_index("event_id_routed")
        )
        merged.rename(columns={"event_id_meta": "event_id"}, inplace=True)

    router = LiveRouter(cfg, model_path=cfg["live"]["model_path"])
    gate = float(cfg.get("live", {}).get("min_p_hat", 0.0))
    use_ml = bool(cfg.get("live", {}).get("use_ml_routing", False))

    # Compute p_hat_live for each row
    p_live = []
    for _, row in merged.iterrows():
        should_route, p_hat = router.score_and_route(row.to_dict())
        p_live.append(p_hat)
    merged["p_hat_live"] = p_live

    if "p_hat" in merged.columns:
        p_ref = merged["p_hat"]
    elif "p_hat_routed" in merged.columns:
        p_ref = merged["p_hat_routed"]
    elif "p_hat_meta" in merged.columns:
        p_ref = merged["p_hat_meta"]
    else:
        raise KeyError("No p_hat column found in merged dataframe.")

    merged["p_diff"] = (merged["p_hat_live"] - p_ref).abs()
    max_diff = merged["p_diff"].max()
    mismatches = (merged["p_diff"] > args.epsilon).sum()
    if use_ml:
        route_diff = (merged["p_hat_live"] >= gate) != (p_ref >= gate)
    else:
        route_diff = pd.Series(False, index=merged.index)

    print(f"Rows compared: {len(merged)}")
    print(f"Max |p_hat_live - p_hat|: {max_diff}")
    print(f"p_hat diffs > {args.epsilon}: {mismatches}")
    print(f"Routing decision mismatches (>= {gate} gate): {route_diff.sum()}")

    return 0 if mismatches == 0 and route_diff.sum() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
