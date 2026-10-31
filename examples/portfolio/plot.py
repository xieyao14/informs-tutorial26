"""Read saved portfolio results and create figures using src/viz.py."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from src.data import load_returns, split_and_scale
from src.viz import save_figures

DATA_DIR = Path(__file__).resolve().parent / "data" / "yfinance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot portfolio figures from saved results."
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

    # === Data preparation ===
    # Reload real returns with the settings saved by train.py.
    with (output_dir / "config.json").open() as f:
        config = json.load(f)

    returns = load_returns(
        tickers=config["tickers"],
        data_dir=DATA_DIR,
    )
    data = split_and_scale(
        R=returns,
        start_date=config["start"],
        end_date=config["end"],
        train_end_date=config["train_end"],
    )
    R_test = data.R_test
    stress_delta = float(config["stress_delta"])
    R_stress = R_test.to_numpy(dtype=float) - stress_delta

    # === Saved results ===
    # Load the saved portfolio weights and optimization histories.
    model_weights = {}
    weights = pd.read_csv(output_dir / "weights.csv")
    for _, row in weights.iterrows():
        key = str(row["model_key"])
        weight_values = row.drop(labels=["model_key", "label"])
        model_weights[key] = weight_values.to_numpy(dtype=float)

    allowed_robust_names = {
        model_name
        for model_name in model_weights
        if model_name.startswith("robust_gamma_")
    }

    dro_loss = pd.read_csv(output_dir / "dro_loss.csv")
    nominal_history = dro_loss[dro_loss["model_key"] == "nominal"].drop(
        columns=["model_key"]
    ).dropna(axis=1, how="all")
    nominal_model = SimpleNamespace(name="nominal", history=nominal_history)

    robust_models = []
    for model_name in sorted(
        allowed_robust_names,
        key=lambda name: float(name.removeprefix("robust_gamma_")),
    ):
        history = dro_loss[dro_loss["model_key"] == model_name].drop(
            columns=["model_key"]
        ).dropna(axis=1, how="all")
        robust_models.append(SimpleNamespace(name=model_name, history=history))

    all_models = [nominal_model] + robust_models

    # === Scenario distributions ===
    # Assemble real, nominal, and worst-case scenarios for distribution comparisons.
    generated_returns = pd.read_csv(output_dir / "nominal.csv")
    training_datasets = {
        "empirical": data.R_train,
        "nominal": generated_returns,
    }
    worst_case_returns = pd.read_csv(output_dir / "worst_case.csv")
    for scenario_name, scenario_frame in worst_case_returns.groupby("scenario", sort=False):
        training_datasets[scenario_name] = scenario_frame.drop(columns=["scenario", "sample"])

    generated_scaled = (
        generated_returns - data.center.to_numpy(dtype=float)[None, :]
    ) / data.flow_scale.to_numpy(dtype=float)[None, :]

    # === Figure output (src/viz.py) ===
    save_figures(
        output_dir=output_dir,
        data=data,
        stress_returns=R_stress,
        weights=model_weights,
        models=all_models,
        datasets=training_datasets,
        generated_scaled=generated_scaled,
    )
    print(f"Rebuilt figures for {output_dir}")


if __name__ == "__main__":
    main()
