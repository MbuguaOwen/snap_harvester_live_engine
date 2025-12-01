from __future__ import annotations

from typing import Dict, Any

import pandas as pd


class LiveMetaBuilder:
    """
    Build a single meta row for a ShockFlip event.

    Design:
    - For replay, we typically receive events that already contain all engineered
      features used in the 2024 BTC meta-model (same schema as the meta_events CSV).
    - For live, the upstream ShockFlip engine should emit the same schema.
    - This class just normalizes a raw event dict into a canonical meta row.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        data_cfg = cfg.get("data", {})
        self.time_col = data_cfg.get("events_time_col", "timestamp")

    def build_meta_row(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a raw event into a meta row.

        Requirements:
        - Preserve all numeric feature columns as-is.
        - Ensure 'event_bar_time' exists and is a UTC timestamp.
        - Ensure 'event_bar_offset_min' exists (0 for aligned events).
        """
        row = dict(event)

        # Ensure event_bar_time
        time_val = row.get("event_bar_time", row.get(self.time_col))
        ts = pd.to_datetime(time_val, utc=True, errors="coerce")
        row["event_bar_time"] = ts

        # Event-bar offset in minutes (0 for live / already aligned events)
        if "event_bar_offset_min" not in row or pd.isna(row["event_bar_offset_min"]):
            row["event_bar_offset_min"] = 0

        return row
