from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from snap_harvester.modeling import build_feature_matrix


@dataclass
class RouterConfig:
    min_p_hat: float = 0.0
    use_ml_routing: bool = False


class LiveRouter:
    """
    Apply the frozen 2024 BTC meta-model on a single meta row.

    - Uses the same build_feature_matrix() preprocessing as training.
    - Aligns columns to model.feature_names_in_ and fills missing with 0.0.
    """

    def __init__(self, cfg: dict, model_path: str | Path) -> None:
        self.cfg = cfg
        live_cfg = cfg.get("live", {})
        self.router_cfg = RouterConfig(
            min_p_hat=float(live_cfg.get("min_p_hat", 0.0)),
            use_ml_routing=bool(live_cfg.get("use_ml_routing", False)),
        )
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.train_cols = list(self.model.feature_names_in_)

    def _build_X(self, meta_row: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([meta_row])
        X = build_feature_matrix(df, self.cfg)
        # Align to training columns
        for col in self.train_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.train_cols]
        return X

    def score_and_route(self, meta_row: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Returns (should_route, p_hat).

        - If use_ml_routing=False, always routes (baseline "strategy" profile).
        - If True, routes only when p_hat >= min_p_hat.
        """
        X = self._build_X(meta_row)
        p_vec = self.model.predict_proba(X)[:, 1]
        p_hat = float(p_vec[0])

        if self.router_cfg.use_ml_routing:
            should_route = p_hat >= self.router_cfg.min_p_hat
        else:
            should_route = True

        return should_route, p_hat
