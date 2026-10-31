"""Read saved outage results and create figures using src/viz.py."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.data import load_outage_splits
from src.viz import save_figures

DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot outage figures from saved results."
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Result directory to regenerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # === Data preparation ===
    # Reload real observations with the settings saved by train.py.
    metadata = json.loads((output_dir / "config.json").read_text())
    splits = load_outage_splits(
        data_root=DATA_DIR,
        county_fips=metadata["county_fips"],
        train_start=metadata["train_start"],
        train_end=metadata["train_end"],
        val_start=metadata["val_start"],
        val_end=metadata["val_end"],
    )
    county_names = splits["county_names"]
    # === Saved results ===
    # Load the saved training history and generated counts.
    flow_history = pd.read_csv(output_dir / "flow_loss.csv")
    real_counts = splits["val_values"].astype(float)
    generated_counts = pd.read_csv(output_dir / "nominal.csv").to_numpy(
        dtype=float
    )

    # === Figure output (src/viz.py) ===
    save_figures(
        output_dir=figures_dir,
        train_history=flow_history["loss"].to_numpy(dtype=float),
        real_counts=real_counts,
        synth_counts=generated_counts,
        real_values=np.log1p(real_counts),
        synth_values=np.log1p(generated_counts),
        county_names=county_names,
    )


if __name__ == "__main__":
    main()
