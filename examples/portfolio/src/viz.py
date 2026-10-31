"""Plot flow results, portfolio optimization histories, and test performance."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.dro import display_label


# === Figure styling ===
def _base_fontsize() -> float:
    return float(plt.rcParams.get("font.size", 10))


def model_color_map(model_names: Iterable[str]) -> dict[str, str]:
    model_names = list(model_names)
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "equal_weight": "#dd8452",
        "nominal": "black",
        "empirical": "black",
    }
    gamma_names = []
    for name in model_names:
        if name.startswith("robust_gamma_"):
            gamma_names.append(("robust_gamma_", name.removeprefix("robust_gamma_")))
        elif name.startswith("worst_case_gamma_"):
            gamma_names.append(
                ("worst_case_gamma_", name.removeprefix("worst_case_gamma_"))
            )

    gamma_values = sorted({gamma_value for _, gamma_value in gamma_names}, key=float)
    gamma_colors = {
        gamma_value: default_cycle[index % len(default_cycle)]
        for index, gamma_value in enumerate(gamma_values)
    }
    for name in model_names:
        if name.startswith("robust_gamma_"):
            gamma_value = name.removeprefix("robust_gamma_")
            color_map[name] = gamma_colors[gamma_value]
        elif name.startswith("worst_case_gamma_"):
            gamma_value = name.removeprefix("worst_case_gamma_")
            color_map[name] = gamma_colors[gamma_value]
    return color_map


# === Flow training and sample quality ===
def plot_flow_outputs(
    output_dir: str | Path,
    dpi: int,
    real_scaled: pd.DataFrame,
    generated_scaled: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "flow_loss.csv"
    metrics_path = output_dir / "metrics.csv"

    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(history["step"], history["loss"], linewidth=1.4)
    ax.set_title("Flow Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "flow_convergence.png", dpi=dpi)
    plt.close(fig)

    metrics = pd.read_csv(metrics_path)
    mmd2 = metrics[
        (metrics["section"] == "flow")
        & (metrics["metric"] == "mmd2")
        & (metrics["series"] != "joint")
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(mmd2["series"], mmd2["value"])
    ax.set_title("Per-Asset MMD2")
    ax.set_ylabel("MMD2")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "flow_metrics.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, column_name in zip(axes.flatten(), real_scaled.columns):
        axis.hist(
            real_scaled[column_name],
            bins=30,
            alpha=0.5,
            density=True,
            label="real",
        )
        axis.hist(
            generated_scaled[column_name],
            bins=30,
            alpha=0.5,
            density=True,
            label="synthetic",
        )
        axis.set_title(column_name)
        axis.grid(alpha=0.2)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Standardized return marginals: real vs synthetic")
    fig.tight_layout()
    fig.savefig(figures_dir / "flow_marginals.png", dpi=dpi)
    plt.close(fig)


# === Portfolio optimization ===
def plot_nominal_convergence(
    nominal_model: object,
    output_path: str | Path,
    dpi: int,
) -> None:
    history = nominal_model.history
    fontsize = _base_fontsize()
    color_map = model_color_map([nominal_model.name])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_loss, ax_theta_grad = axes

    ax_loss.plot(
        history["step"],
        history["loss"],
        label=display_label(nominal_model.name),
        color=color_map.get(nominal_model.name),
    )
    ax_loss.set_title("Nominal loss", fontsize=fontsize)
    ax_loss.set_xlabel("Step", fontsize=fontsize)
    ax_loss.set_ylabel("Loss", fontsize=fontsize)

    ax_theta_grad.plot(
        history["step"],
        history["grad_theta_norm"],
        label=display_label(nominal_model.name),
        color=color_map.get(nominal_model.name),
    )
    ax_theta_grad.set_title(r"Nominal theta gradient norm", fontsize=fontsize)
    ax_theta_grad.set_xlabel("Step", fontsize=fontsize)
    ax_theta_grad.set_ylabel(r"$\|\nabla_\theta\|$", fontsize=fontsize)
    ax_theta_grad.set_yscale("log")

    for ax in axes:
        ax.legend(fontsize=fontsize)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=fontsize)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_robust_convergence(
    models: Iterable[object],
    output_path: str | Path,
    dpi: int,
) -> None:
    robust_models = [model for model in models if "objective" in model.history.columns]
    fontsize = _base_fontsize()
    color_map = model_color_map(model.name for model in robust_models)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_robust_objective, ax_theta_grad, ax_z_grad = axes

    for model in robust_models:
        history = model.history
        ax_robust_objective.plot(
            history["step"],
            history["objective"],
            label=display_label(model.name),
            color=color_map.get(model.name),
            linewidth=1.2,
        )
        ax_theta_grad.plot(
            history["step"],
            history["grad_theta_norm"],
            label=display_label(model.name),
            color=color_map.get(model.name),
            linewidth=1.2,
        )
        ax_z_grad.plot(
            history["step"],
            history["grad_z_norm"],
            label=display_label(model.name),
            color=color_map.get(model.name),
            linewidth=1.2,
        )

    ax_robust_objective.set_title("Robust objective", fontsize=fontsize)
    ax_robust_objective.set_xlabel("Step", fontsize=fontsize)
    ax_robust_objective.set_ylabel("Objective", fontsize=fontsize)

    ax_theta_grad.set_title(r"Robust theta gradient norm", fontsize=fontsize)
    ax_theta_grad.set_xlabel("Step", fontsize=fontsize)
    ax_theta_grad.set_ylabel(r"$\|\nabla_\theta\|$", fontsize=fontsize)
    ax_theta_grad.set_yscale("log")

    ax_z_grad.set_title(r"Particle gradient norm", fontsize=fontsize)
    ax_z_grad.set_xlabel("Step", fontsize=fontsize)
    ax_z_grad.set_ylabel(r"$\|\nabla_z\|$", fontsize=fontsize)
    ax_z_grad.set_yscale("log")

    for ax in axes:
        ax.legend(fontsize=fontsize)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=fontsize)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# === Scenario distributions ===
def plot_distributions(
    datasets: dict[str, pd.DataFrame],
    output_path: str | Path,
    dpi: int,
    color_map: dict[str, str] | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    merged_color_map = model_color_map(datasets.keys())
    if color_map is not None:
        merged_color_map.update(color_map)
    line_style_map = {
        "empirical": {"linestyle": "--", "linewidth": 1.6},
        "nominal": {"linestyle": ":", "linewidth": 2.2},
    }
    fontsize = _base_fontsize()

    columns = next(iter(datasets.values())).columns
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.8))
    for axis, column_name in zip(axes.flatten(), columns):
        column_values = [
            frame[column_name].to_numpy(dtype=float) for frame in datasets.values()
        ]
        x_min = min(values.min() for values in column_values)
        x_max = max(values.max() for values in column_values)
        x_grid = np.linspace(x_min, x_max, 400)

        for label, frame in datasets.items():
            values = frame[column_name].to_numpy(dtype=float)
            if np.allclose(values.std(ddof=0), 0.0):
                continue
            kde = gaussian_kde(values)
            label_text = (
                label_map.get(label)
                if label_map is not None and label in label_map
                else display_label(label)
            )
            axis.plot(
                x_grid,
                kde(x_grid),
                **line_style_map.get(label, {"linestyle": "-", "linewidth": 1.2}),
                label=label_text,
                color=merged_color_map.get(label),
            )
        axis.set_title(column_name, fontsize=fontsize)
        if axis in (axes[0, 0], axes[1, 0]):
            axis.set_ylabel("Density", fontsize=fontsize)
        axis.grid(alpha=0.2)
        axis.tick_params(axis="both", labelsize=fontsize)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.99),
        fontsize=fontsize,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# === Portfolio evaluation ===
def plot_wealth(
    R_test: pd.DataFrame,
    R_stress: np.ndarray,
    model_weights: dict[str, np.ndarray],
    output_path: str | Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), sharey=True)
    ax_test, ax_stress = axes
    fontsize = _base_fontsize()
    color_map = model_color_map(model_weights.keys())
    line_style_map = {
        "equal_weight": {"linestyle": "--", "linewidth": 1.2, "zorder": 4},
        "nominal": {"linestyle": ":", "linewidth": 2.2, "zorder": 4},
    }
    for model_name, w in model_weights.items():
        test_wealth = np.cumprod(1.0 + np.asarray(R_test.values @ w))
        stress_wealth = np.cumprod(1.0 + np.asarray(R_stress @ w))
        line_kwargs = line_style_map.get(
            model_name,
            {"linestyle": "-", "linewidth": 1.2, "zorder": 3},
        )
        ax_test.plot(
            R_test.index,
            test_wealth,
            label=display_label(model_name),
            color=color_map.get(model_name),
            **line_kwargs,
        )
        ax_stress.plot(
            R_test.index,
            stress_wealth,
            label=display_label(model_name),
            color=color_map.get(model_name),
            **line_kwargs,
        )

    ax_test.set_title("Non-Stress Test", fontsize=fontsize)
    ax_test.set_ylabel("Wealth", fontsize=fontsize)
    ax_stress.set_title("Stress Test", fontsize=fontsize)
    x_start = R_test.index.min()
    x_end = R_test.index.max()
    x_pad = pd.Timedelta(days=5)
    ax_test.set_xlim(x_start - x_pad, x_end + x_pad)
    ax_stress.set_xlim(x_start - x_pad, x_end + x_pad)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="both", labelsize=fontsize)

    ax_test.legend(fontsize=fontsize)
    ax_stress.legend(fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_left_tail_histogram(
    R_test: pd.DataFrame,
    model_weights: dict[str, np.ndarray],
    output_path: str | Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(6, 3.2))
    fontsize = _base_fontsize()
    color_map = model_color_map(model_weights.keys())
    for model_name, w in model_weights.items():
        r = R_test.values @ w
        q = np.quantile(r, 0.15)
        r_left = r[r <= q]
        plt.hist(
            r_left,
            bins=25,
            alpha=0.45,
            density=True,
            label=display_label(model_name),
            color=color_map.get(model_name),
        )

    plt.title("Left-tail portfolio return distribution on test set", fontsize=fontsize)
    plt.xlabel("Portfolio return", fontsize=fontsize)
    plt.ylabel("Density", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.gcf().savefig(output_path, dpi=dpi)
    plt.close()


# === Figure output (plot.py) ===
def save_figures(
    output_dir, data, stress_returns, weights, models, datasets, generated_scaled
):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = 160
    plot_flow_outputs(output_dir, dpi, data.X_train, generated_scaled)
    plot_nominal_convergence(
        next(model for model in models if model.name == "nominal"),
        figures_dir / "nominal_training_convergence.png",
        dpi,
    )
    plot_robust_convergence(
        models, figures_dir / "robust_training_convergence.png", dpi
    )
    plot_distributions(
        datasets,
        figures_dir / "distribution_comparison.png",
        dpi,
        color_map={"empirical": "gray", "nominal": "black"},
        label_map={"nominal": "Nominal"},
    )
    plot_wealth(
        data.R_test, stress_returns, weights, figures_dir / "test_cumulative_wealth.png", dpi
    )
    plot_left_tail_histogram(
        data.R_test, weights, figures_dir / "test_left_tail_histogram.png", dpi
    )
