from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COUNTIES = [
    ("13121", "Fulton County"),
    ("13089", "DeKalb County"),
    ("13067", "Cobb County"),
    ("13135", "Gwinnett County"),
    ("13063", "Clayton County"),
    ("13097", "Douglas County"),
    ("13057", "Cherokee County"),
    ("13117", "Forsyth County"),
    ("13151", "Henry County"),
    ("13247", "Rockdale County"),
]


def load_outage_splits(
    data_root: str | Path,
    county_fips: list[str],
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> dict:
    frame = pd.read_csv(
        Path(data_root) / "outage_counts.csv",
        dtype={fips: np.uint32 for fips in county_fips},
        parse_dates=["timestamp"],
    )
    missing = [fips for fips in county_fips if fips not in frame.columns]
    if missing:
        raise KeyError(f"missing counties in outage_counts.csv: {missing}")

    raw_values = frame[county_fips].to_numpy(dtype=np.uint32, copy=True)
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    county_name_lookup = dict(DEFAULT_COUNTIES)
    county_names = [county_name_lookup[fips] for fips in county_fips]
    train_mask = (timestamps >= pd.Timestamp(train_start)) & (
        timestamps <= pd.Timestamp(train_end)
    )
    val_mask = (timestamps >= pd.Timestamp(val_start)) & (
        timestamps <= pd.Timestamp(val_end)
    )
    if not train_mask.any():
        raise ValueError("empty training split")
    if not val_mask.any():
        raise ValueError("empty validation split")

    splits = {
        "train_values": raw_values[train_mask],
        "val_values": raw_values[val_mask],
        "train_timestamps": timestamps[train_mask].to_numpy(),
        "val_timestamps": timestamps[val_mask].to_numpy(),
    }
    splits["train_log1p"] = np.log1p(splits["train_values"]).astype(np.float32)
    splits["val_log1p"] = np.log1p(splits["val_values"]).astype(np.float32)
    train_mean = splits["train_log1p"].mean(axis=0, keepdims=True).astype(np.float32)
    train_std = splits["train_log1p"].std(axis=0, keepdims=True).astype(np.float32)
    train_std = np.where(train_std < 1e-6, 1.0, train_std).astype(np.float32)
    splits["train_mean"] = train_mean
    splits["train_std"] = train_std
    splits["train_standardized"] = ((splits["train_log1p"] - train_mean) / train_std).astype(
        np.float32
    )
    splits["val_standardized"] = ((splits["val_log1p"] - train_mean) / train_std).astype(
        np.float32
    )
    splits["county_names"] = county_names
    splits["county_fips"] = county_fips
    return splits
