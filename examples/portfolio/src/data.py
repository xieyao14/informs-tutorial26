"""Load asset returns and prepare chronological splits and training-period scales."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "IWM", "EFA", "EEM", "AGG", "GLD"]


@dataclass
class SplitData:
    """Raw returns, standardized flow inputs, and training statistics."""

    R_train: pd.DataFrame
    R_test: pd.DataFrame
    X_train: pd.DataFrame
    center: pd.Series
    flow_scale: pd.Series
    scale: float


def load_returns(
    tickers: list[str],
    data_dir: str | Path,
) -> pd.DataFrame:
    """Compute daily simple returns on trading dates shared by all assets."""
    price_frames = []

    for ticker in tickers:
        price_data = pd.read_csv(
            Path(data_dir) / f"{ticker}_daily_adjusted.csv",
            usecols=["timestamp", "adjusted_close"],
            parse_dates=["timestamp"],
            dtype={"adjusted_close": float},
        ).dropna().sort_values("timestamp")
        price_frames.append(
            price_data.rename(columns={"adjusted_close": ticker}).set_index("timestamp")
        )

    adjusted_prices = (
        pd.concat(price_frames, axis=1, join="inner").sort_index().dropna()
    )
    return adjusted_prices.pct_change().dropna()


def split_and_scale(
    R: pd.DataFrame,
    start_date: str,
    end_date: str,
    train_end_date: str,
) -> SplitData:
    """Split returns chronologically and estimate scaling from the training period."""
    R = R.loc[start_date:end_date].dropna()

    R_train = R.loc[:train_end_date].copy()
    test_start = pd.Timestamp(train_end_date) + pd.Timedelta(days=1)
    R_test = R.loc[test_start:].copy()

    center = R_train.mean()
    # The flow uses assetwise standard deviations; DRO uses one global scale.
    flow_scale = R_train.std().replace(0.0, 1.0)
    global_scale = float(np.mean(np.abs(R_train.to_numpy(dtype=float))))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    X_train = (R_train - center) / flow_scale
    return SplitData(
        R_train=R_train,
        R_test=R_test,
        X_train=X_train,
        center=center,
        flow_scale=flow_scale,
        scale=global_scale,
    )
