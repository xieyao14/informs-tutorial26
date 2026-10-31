"""Download the selected EAGLE-I counties to data/outage_counts.csv."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from src.data import DEFAULT_COUNTIES

DATASET_URL = "https://api.figshare.com/v2/articles/24237376/versions/4"
YEARS = range(2018, 2025)


def read_year(url: str, year: int, county_fips: list[str]) -> pd.DataFrame:
    """Read one annual EAGLE-I file and retain the selected counties."""
    # The 2023 release names the outage-count column "sum".
    count_column = "sum" if year == 2023 else "customers_out"
    selected = []
    with urlopen(url, timeout=120) as response:
        # Filter counties while streaming the national file to limit memory use.
        chunks = pd.read_csv(
            response,
            usecols=["fips_code", count_column, "run_start_time"],
            dtype={"fips_code": str, count_column: float, "run_start_time": str},
            chunksize=250000,
        )
        for chunk in chunks:
            selected.append(chunk.loc[chunk["fips_code"].isin(county_fips)])
    frame = pd.concat(selected, ignore_index=True)
    frame["run_start_time"] = pd.to_datetime(frame["run_start_time"])
    counts = frame.pivot(
        index="run_start_time", columns="fips_code", values=count_column
    )
    # Put every year on the same 15-minute grid and county-column order.
    timestamps = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01",
        freq="15min", inclusive="left", name="timestamp",
    )
    return counts.reindex(index=timestamps, columns=county_fips).fillna(0).astype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EAGLE-I county outage counts.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "outage_counts.csv"
    if path.exists() and not args.overwrite:
        print(f"Keeping {path}")
        return

    with urlopen(DATASET_URL, timeout=120) as response:
        files = {file["name"]: file for file in json.load(response)["files"]}
    county_fips = [fips for fips, _ in DEFAULT_COUNTIES]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for year in YEARS:
            file = files[f"eaglei_outages_{year}.csv"]
            print(f"{year}: reading {file['size'] / 1e9:.2f} GB", flush=True)
            futures.append(pool.submit(
                read_year, file["download_url"] + "?download=1", year, county_fips
            ))
        yearly_counts = []
        for year, future in zip(YEARS, futures):
            counts = future.result()
            yearly_counts.append(counts)
            print(f"{year}: retained {len(counts)} timestamps", flush=True)
    pd.concat(yearly_counts).to_csv(path, index_label="timestamp")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
