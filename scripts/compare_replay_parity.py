from __future__ import annotations

import argparse
import sys
from typing import Tuple

import pandas as pd


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "event_id" not in df.columns:
        raise ValueError(f"{path} is missing required column event_id")
    df["event_id"] = df["event_id"].astype(str)
    return df


def compare(live_path: str, research_path: str, epsilon: float) -> Tuple[int, str]:
    live = _load(live_path).set_index("event_id")
    ref = _load(research_path).set_index("event_id")

    missing_live = sorted(set(ref.index) - set(live.index))
    missing_ref = sorted(set(live.index) - set(ref.index))

    merged = live.join(ref, lsuffix="_live", rsuffix="_ref", how="outer")

    flag_cols = ["hit_tp", "hit_sl", "hit_be"]
    flag_mismatches = {}
    for col in flag_cols:
        col_l = f"{col}_live"
        col_r = f"{col}_ref"
        mask = merged[col_l] != merged[col_r]
        flag_mismatches[col] = merged.index[mask].tolist()

    merged["r_diff"] = merged["r_final_live"] - merged["r_final_ref"]
    r_bad = merged.index[merged["r_diff"].abs() > epsilon].tolist()

    lines = [
        f"Live rows: {len(live)}, Research rows: {len(ref)}",
        f"Missing in live: {len(missing_live)}",
        f"Missing in research: {len(missing_ref)}",
    ]
    for col, ids in flag_mismatches.items():
        lines.append(f"{col} mismatches: {len(ids)}")
    lines.append(f"r_final differences > {epsilon}: {len(r_bad)}")

    errors = sum(
        [
            len(missing_live),
            len(missing_ref),
            *(len(ids) for ids in flag_mismatches.values()),
            len(r_bad),
        ]
    )
    detail = "\n".join(lines)
    if errors:
        return 1, detail
    return 0, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare routed tape parity vs research snapshot.")
    parser.add_argument("--live", default="results/live/replay_routed_tape_2025_BTC.csv")
    parser.add_argument("--research", default="results/meta/snap_routed_tape_2025_BTC_agg.csv")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    code, message = compare(args.live, args.research, args.epsilon)
    print(message)
    sys.exit(code)


if __name__ == "__main__":
    main()
