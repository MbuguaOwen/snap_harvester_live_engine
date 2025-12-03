#!/usr/bin/env python3
"""
End-to-end helper to rebuild a clean November 2025 OOS slice:
- Clean ticks (skip malformed rows, enforce timestamp bounds)
- Build November-only minutes
- Run ShockFlip detection with proven Nov params (z=1.2, jump=1.5, persistence=1, z_window=90, donchian=40, min_bars=60)
- Build meta, attach p_hat (frozen 2024 model), build base + routed tapes, summarize PnL.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild November 2025 OOS pipeline.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tick_glob", required=True, help="Glob for November tick CSVs, e.g. data/ticks/BTCUSDT/*2025-11*.csv")
    ap.add_argument("--timestamp_unit", default="ms", help="Timestamp unit for ticks (default: ms)")
    ap.add_argument("--min_timestamp", default="2020-01-01", help="Drop ticks before this UTC time (default: 2020-01-01)")
    ap.add_argument("--max_timestamp", default="2026-01-01", help="Drop ticks after this UTC time (default: 2026-01-01)")
    ap.add_argument("--minutes_out", default=None, help="Output Parquet for November minutes (default: data/minutes/{symbol}_1min_2025-11_clean.parquet)")
    ap.add_argument("--events_dir", default=None, help="Output directory for Diamond Hunter events (default: results/diamond_hunter_btc_2025_11/)")
    ap.add_argument("--config", default="configs/snap_harvester_2025_btc_nov.yaml", help="YAML config for meta build")
    ap.add_argument("--model", default="results/models/hgb_snap_2024_BTC.joblib", help="Frozen model path for p_hat")
    ap.add_argument("--risk_profile", default="strategy", help="Risk profile label for routed tape")
    args = ap.parse_args()

    tick_glob = str(Path(args.tick_glob).resolve())
    symbol = args.symbol
    symbol_lower = symbol.lower()

    minutes_out = Path(args.minutes_out or f"data/minutes/{symbol}_1min_2025-11_clean.parquet")
    events_dir = Path(args.events_dir or f"results/diamond_hunter_{symbol_lower}_2025_11")
    events_dir.mkdir(parents=True, exist_ok=True)
    events_raw = events_dir / "events_annotated.csv"
    events_final = events_dir / "events_annotated_2025_11_BTC.csv"

    # Step 1: clean + build November minutes
    run(
        [
            "python",
            "scripts/build_minutes_from_ticks.py",
            "--symbol",
            symbol,
            "--tick_glob",
            tick_glob,
            "--out",
            str(minutes_out),
            "--timestamp_unit",
            args.timestamp_unit,
            "--min_timestamp",
            args.min_timestamp,
            "--max_timestamp",
            args.max_timestamp,
        ]
    )

    # Step 2: run ShockFlip detection with proven Nov params
    run(
        [
            "python",
            "scripts/diamond_hunter.py",
            "--tick_dir",
            str(Path(tick_glob).parent),
            "--out",
            str(events_dir),
            "--z_band",
            "1.2",
            "--jump_band",
            "1.5",
            "--persistence",
            "1",
            "--z_window",
            "90",
            "--donchian_window",
            "40",
            "--min_bars",
            "60",
        ]
    )

    # Normalize events filename
    if events_raw.exists():
        shutil.move(events_raw, events_final)

    # Step 3: build meta dataset
    run(
        [
            "python",
            "scripts/01_build_meta_dataset.py",
            "--config",
            args.config,
            "--risk-profile",
            args.risk_profile,
        ]
    )

    # Derive paths from config naming convention
    meta_out = Path("results/meta/snap_meta_dataset_2025_BTC_nov.csv")
    meta_with_preds = Path("results/meta/snap_meta_with_p_hat_2025_BTC_nov.csv")
    base_trades = Path("results/trades/snap_base_trades_2025_BTC_nov.csv")
    routed_trades = Path("results/trades/snap_routed_tape_2025_BTC_nov.csv")
    pnl_summary = Path("results/trades/snap_routed_pnl_summary_2025_BTC_nov.csv")

    # Step 4: add p_hat using frozen 2024 model
    run(
        [
            "python",
            "scripts/add_preds_to_meta.py",
            "--meta",
            str(meta_out),
            "--model",
            args.model,
            "--config",
            args.config,
            "--out",
            str(meta_with_preds),
        ]
    )

    # Step 5: build base trade tape (Snap barrier 0.5R/3R/H30)
    run(
        [
            "python",
            "scripts/build_base_trade_tape_from_meta.py",
            "--meta",
            str(meta_out),
            "--out",
            str(base_trades),
            "--barrier_col",
            "barrier_y_H30_R3p0_sl0p5",
            "--tp_r",
            "3.0",
            "--sl_r",
            "0.5",
        ]
    )

    # Step 6: attach p_hat to trades (routed tape)
    run(
        [
            "python",
            "scripts/build_routed_trade_tape.py",
            "--trades",
            str(base_trades),
            "--preds",
            str(meta_with_preds),
            "--out",
            str(routed_trades),
            "--risk_profile",
            args.risk_profile,
        ]
    )

    # Step 7: summarize PnL
    run(
        [
            "python",
            "scripts/summarize_routed_pnl.py",
            "--routed",
            str(routed_trades),
            "--out_csv",
            str(pnl_summary),
        ]
    )

    print("\n[Done] November OOS pipeline completed.")
    print(f"  Minutes:   {minutes_out}")
    print(f"  Events:    {events_final}")
    print(f"  Meta:      {meta_out}")
    print(f"  Meta+pred: {meta_with_preds}")
    print(f"  Trades:    {routed_trades}")
    print(f"  Summary:   {pnl_summary}")


if __name__ == "__main__":
    main()
