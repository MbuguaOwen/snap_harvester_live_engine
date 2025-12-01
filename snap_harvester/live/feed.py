from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


@dataclass
class Bar:
    """Minimal 1m bar representation used by the live engine."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


Event = Dict[str, object]


class ReplayBarFeed:
    """
    Simple iterator over a Parquet / CSV minutes file.

    This is used for:
    - Offline parity / paper runs (replay mode)
    - Backfilling state before switching to live

    Bars are yielded in strict timestamp order.
    """

    def __init__(
        self,
        path: str | Path,
        time_col: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        p = Path(path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        if time_col not in df.columns:
            raise ValueError(f"Bars file missing time column {time_col!r}")
        self.time_col = time_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.df = df.sort_values(time_col).reset_index(drop=True)

    def __iter__(self) -> Iterator[Bar]:
        for _, row in self.df.iterrows():
            ts = pd.to_datetime(row[self.time_col], utc=True, errors="coerce")
            yield Bar(
                timestamp=ts,
                open=float(row[self.open_col]),
                high=float(row[self.high_col]),
                low=float(row[self.low_col]),
                close=float(row[self.close_col]),
                volume=float(row.get(self.volume_col, 0.0)),
            )


class ReplayEventFeed:
    """
    Iterator over a meta-events CSV (ShockFlip events).

    Expects columns at least:
    - timestamp
    - side
    - symbol
    - all feature columns used by the 2024 BTC meta-model
    """

    def __init__(self, path: str | Path, time_col: str = "timestamp") -> None:
        p = Path(path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        if time_col not in df.columns:
            raise ValueError(f"Events file missing time column {time_col!r}")

        # Normalize timestamp
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        self.time_col = time_col
        self.df = df.sort_values(time_col).reset_index(drop=True)

    def __iter__(self) -> Iterator[Event]:
        for _, row in self.df.iterrows():
            ev = row.to_dict()
            # Provide a canonical event_bar_time used everywhere
            if "event_bar_time" not in ev or pd.isna(ev["event_bar_time"]):
                ev["event_bar_time"] = ev[self.time_col]
            yield ev
