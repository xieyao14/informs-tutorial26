from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "IWM", "EFA", "EEM", "AGG", "GLD"]


@dataclass
class SplitData:
    R: pd.DataFrame
    R_train: pd.DataFrame
    R_test: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    center: pd.DataFrame
    scale: pd.DataFrame


def load_prices(symbol: str, data_dir: str | Path) -> pd.DataFrame:
    file_path = Path(data_dir) / f"{symbol}_daily_adjusted.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No price file found for {symbol}: {file_path}")

    price_data = pd.read_csv(file_path)
    required_columns = {"timestamp", "adjusted_close"}
    missing_columns = required_columns - set(price_data.columns)
    if missing_columns:
        raise ValueError(
            f"Price file for {symbol} is missing required columns "
            f"{sorted(missing_columns)}: {file_path}"
        )
    if price_data.empty:
        raise ValueError(f"Price file for {symbol} is empty: {file_path}")
    return price_data


def build_return_matrix(
    tickers: list[str],
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames = []

    for ticker in tickers:
        price_data = load_prices(symbol=ticker, data_dir=data_dir)
        price_data = price_data[["timestamp", "adjusted_close"]].copy()
        price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
        price_data["adjusted_close"] = pd.to_numeric(
            price_data["adjusted_close"], errors="coerce"
        )
        price_data = price_data.dropna().sort_values("timestamp")
        price_frames.append(
            price_data.rename(columns={"adjusted_close": ticker}).set_index("timestamp")
        )

    adjusted_prices = (
        pd.concat(price_frames, axis=1, join="inner").sort_index().dropna()
    )
    daily_returns = adjusted_prices.pct_change().dropna()
    return adjusted_prices, daily_returns


def split_and_scale(
    R: pd.DataFrame,
    start_date: str,
    end_date: str,
    train_end_date: str,
) -> SplitData:
    R = R.copy()

    if start_date:
        R = R.loc[start_date:]
    if end_date:
        R = R.loc[:end_date]

    R = R.dropna()

    R_train = R.loc[:train_end_date].copy()
    test_start = pd.Timestamp(train_end_date) + pd.Timedelta(days=1)
    R_test = R.loc[test_start:].copy()

    if len(R_train) < 50 or len(R_test) < 20:
        raise ValueError(
            f"Training/test split too small. train={len(R_train)}, test={len(R_test)}. "
            f"Check start={start_date}, train_end={train_end_date}, end={end_date}."
        )

    center = R_train.mean()
    global_scale = float(np.mean(np.abs(R_train.to_numpy(dtype=float))))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    scale = pd.Series(global_scale, index=R_train.columns)

    X_train = (R_train - center) / scale
    X_test = (R_test - center) / scale

    return SplitData(
        R=R,
        R_train=R_train,
        R_test=R_test,
        X_train=X_train,
        X_test=X_test,
        center=center.to_frame("center"),
        scale=scale.to_frame("scale"),
    )
