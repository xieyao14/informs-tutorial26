from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data import DEFAULT_TICKERS, build_return_matrix, split_and_scale
from src.viz import (
    plot_convergence_histories,
    plot_cumulative_returns,
    plot_flow_outputs,
    plot_left_tail_histogram,
    plot_nominal_history,
    plot_distributions,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "yfinance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate portfolio figures from saved outputs without retraining."
    )
    parser.add_argument(
        "--output-dir",
        default="results/default",
        help="Result directory to regenerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    dpi = 160
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open() as f:
        config = json.load(f)

    _, returns = build_return_matrix(
        tickers=config.get("tickers", DEFAULT_TICKERS),
        data_dir=DATA_DIR,
    )
    data = split_and_scale(
        R=returns,
        start_date=config["start"],
        end_date=config["end"],
        train_end_date=config["train_end"],
    )
    R_test = data.R_test

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
        if model_name in allowed_robust_names:
            history = dro_loss[dro_loss["model_key"] == model_name].drop(
                columns=["model_key"]
            ).dropna(axis=1, how="all")
            robust_models.append(
                SimpleNamespace(name=model_name, history=history)
            )

    all_models = [nominal_model] + robust_models
    if "equal_weight" in model_weights:
        all_models = [
            SimpleNamespace(name="equal_weight", history=pd.DataFrame())
        ] + all_models

    stress_delta = float(config["stress_delta"])
    R_stress = R_test.to_numpy(dtype=float) - stress_delta

    plot_cumulative_returns(
        R_test=R_test,
        R_stress=R_stress,
        model_weights=model_weights,
        output_path=str(figures_dir / "test_cumulative_wealth.png"),
        dpi=dpi,
    )
    plot_left_tail_histogram(
        R_test=R_test,
        model_weights=model_weights,
        output_path=str(figures_dir / "test_left_tail_histogram.png"),
        dpi=dpi,
    )
    plot_convergence_histories(
        models=all_models,
        output_path=str(figures_dir / "robust_training_convergence.png"),
        dpi=dpi,
    )
    plot_nominal_history(
        nominal_model=nominal_model,
        output_path=str(figures_dir / "nominal_training_convergence.png"),
        dpi=dpi,
    )

    generated_returns = pd.read_csv(output_dir / "nominal.csv")
    training_datasets = {
        "empirical": data.R_train,
        "nominal": generated_returns,
    }
    worst_case_path = output_dir / "worst_case.csv"
    if worst_case_path.exists():
        allowed_distribution_names = {
            f"worst_case_{model.name.removeprefix('robust_')}"
            for model in robust_models
        }
        worst_case_returns = pd.read_csv(worst_case_path)
        for scenario_name, scenario_frame in worst_case_returns.groupby(
            "scenario", sort=False
        ):
            if scenario_name in allowed_distribution_names:
                training_datasets[scenario_name] = scenario_frame.drop(
                    columns=["scenario", "sample"]
                )

    if training_datasets:
        color_map = (
            {"empirical": "gray", "nominal": "black"}
            if "nominal" in training_datasets
            else None
        )
        plot_distributions(
            datasets=training_datasets,
            output_path=str(figures_dir / "distribution_comparison.png"),
            dpi=dpi,
            color_map=color_map,
            label_map={"nominal": "Nominal"},
        )

    generated_scaled = (
        generated_returns - data.center["center"].to_numpy(dtype=float)[None, :]
    ) / data.scale["scale"].to_numpy(dtype=float)[None, :]
    plot_flow_outputs(
        output_dir=str(output_dir),
        dpi=dpi,
        real_scaled=data.X_train,
        generated_scaled=generated_scaled,
    )
    print(f"Rebuilt figures for {output_dir}")


if __name__ == "__main__":
    main()
