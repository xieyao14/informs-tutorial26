from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import numpy as np


PLOT_COUNTIES = {"Fulton", "DeKalb", "Clayton", "Rockdale"}

REAL_COLOR = "#355C7D"
SYNTH_COLOR = "#E07A5F"
CORRELATION_CMAP = "RdBu_r"


FIGURE_TITLE_FONTSIZE = 11
FIGURE_LABEL_FONTSIZE = 10
FIGURE_TICK_FONTSIZE = 10
FIGURE_LEGEND_FONTSIZE = 10


def display_county_name(name: str) -> str:
    return name.removesuffix(" County")


def _style_axis_fonts(ax: plt.Axes) -> None:
    ax.title.set_fontsize(FIGURE_TITLE_FONTSIZE)
    ax.xaxis.label.set_fontsize(FIGURE_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(FIGURE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=FIGURE_TICK_FONTSIZE)


def _style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("0.8")
    frame.set_alpha(0.95)
    for text in legend.get_texts():
        text.set_fontsize(FIGURE_LEGEND_FONTSIZE)


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=160)


def save_loss_plot(history: list[float], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, linewidth=1.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    _style_axis_fonts(ax)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def save_histograms(
    real_counts: np.ndarray,
    synth_counts: np.ndarray,
    county_names: list[str],
    path: str | Path,
) -> None:
    n_counties = len(county_names)
    n_cols = n_counties
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.25))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    ymax = 0.0
    panels = []
    for idx, name in enumerate(county_names):
        real = np.log10(1.0 + np.asarray(real_counts[:, idx], dtype=np.float64))
        synth = np.log10(1.0 + np.asarray(synth_counts[:, idx], dtype=np.float64))
        hi = max(float(np.percentile(real, 99.5)), float(np.percentile(synth, 99.5)), 0.5)
        bins = np.linspace(0.0, hi, 14)
        real_weights = np.ones_like(real) / max(len(real), 1)
        synth_weights = np.ones_like(synth) / max(len(synth), 1)
        real_hist, _ = np.histogram(real, bins=bins, weights=real_weights)
        synth_hist, _ = np.histogram(synth, bins=bins, weights=synth_weights)
        ymax = max(ymax, float(real_hist.max(initial=0.0)), float(synth_hist.max(initial=0.0)))
        panels.append((name, real, synth, bins))
    ymax *= 1.08
    for idx, (name, real, synth, bins) in enumerate(panels):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.hist(
            real,
            bins=bins,
            weights=np.ones_like(real) / max(len(real), 1),
            alpha=0.55,
            color=REAL_COLOR,
            label="Real",
        )
        ax.hist(
            synth,
            bins=bins,
            weights=np.ones_like(synth) / max(len(synth), 1),
            alpha=0.55,
            color=SYNTH_COLOR,
            label="Synthetic",
        )
        ax.set_title(display_county_name(name))
        ax.set_xlabel(r"$\log_{10}(1 + \text{\# Outages})$")
        ax.set_ylabel("Probability")
        ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.25)
        _style_axis_fonts(ax)
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            handlelength=1.8,
            borderpad=0.3,
            labelspacing=0.25,
        )
        _style_legend(legend)
    for idx in range(n_counties, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def save_marginal_distributions(
    real_counts: np.ndarray,
    synth_counts: np.ndarray,
    county_names: list[str],
    path: str | Path,
) -> None:
    n_counties = len(county_names)
    n_cols = n_counties
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.25))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    panels = []
    x_max = 0.0
    for idx, name in enumerate(county_names):
        real = np.sort(np.asarray(real_counts[:, idx], dtype=np.float64))
        synth = np.sort(np.asarray(synth_counts[:, idx], dtype=np.float64))
        x_max = max(x_max, float(real.max(initial=0.0)), float(synth.max(initial=0.0)))
        panels.append((name, real, synth))
    x_max_data = max(x_max, 1.0)
    symlog = mscale.SymmetricalLogScale(None, base=10, linthresh=1.0, linscale=1.0).get_transform()
    symlog_inv = symlog.inverted()
    left_pad = 0.08
    right_curve_pad = 0.03
    right_axis_pad = 0.08
    x_min = float(symlog_inv.transform_non_affine(np.array([-left_pad], dtype=np.float64))[0])
    x_plot_max = float(symlog_inv.transform_non_affine(
        symlog.transform_non_affine(np.array([x_max_data], dtype=np.float64)) + right_curve_pad
    )[0])
    x_max = float(symlog_inv.transform_non_affine(
        symlog.transform_non_affine(np.array([x_max_data], dtype=np.float64)) + right_axis_pad
    )[0])
    y_min = -0.02
    y_max = 1.02
    for idx, (name, real, synth) in enumerate(panels):
        ax = axes[idx // n_cols, idx % n_cols]
        y_real = np.linspace(0.0, 1.0, len(real), endpoint=False)
        y_synth = np.linspace(0.0, 1.0, len(synth), endpoint=False)
        x_real = np.concatenate([real, [x_plot_max]])
        x_synth = np.concatenate([synth, [x_plot_max]])
        y_real = np.concatenate([y_real, [1.0]])
        y_synth = np.concatenate([y_synth, [1.0]])
        ax.plot(x_real, y_real, linewidth=1.9, linestyle="-", color=REAL_COLOR, label="Real")
        ax.plot(
            x_synth,
            y_synth,
            linewidth=2.8,
            linestyle=(0, (2.5, 1.5)),
            dash_capstyle="butt",
            color=SYNTH_COLOR,
            label="Synthetic",
        )
        ax.set_title(display_county_name(name))
        ax.set_xlabel(r"\# Outages")
        if idx == 0:
            ax.set_ylabel("Empirical CDF")
        else:
            ax.set_ylabel("")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        _style_axis_fonts(ax)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            handlelength=1.8,
            borderpad=0.3,
            labelspacing=0.25,
        )
        _style_legend(legend)
    for idx in range(n_counties, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def save_correlation_heatmaps(
    real_values: np.ndarray,
    synth_values: np.ndarray,
    county_names: list[str],
    path: str | Path,
) -> None:
    real_corr = np.corrcoef(real_values, rowvar=False)
    synth_corr = np.corrcoef(synth_values, rowvar=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.45))
    display_names = [display_county_name(name) for name in county_names]
    titles = ["Real", "Synthetic"]
    for ax, mat, title in zip(axes, [real_corr, synth_corr], titles):
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap=CORRELATION_CMAP)
        ax.set_xticks(range(len(county_names)))
        ax.set_yticks(range(len(county_names)))
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.set_yticklabels(display_names)
        ax.set_title(title)
        _style_axis_fonts(ax)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.23, top=0.88, wspace=0.22)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=FIGURE_TICK_FONTSIZE)
    _save_figure(fig, path)
    plt.close(fig)


def save_figures(
    output_dir: str | Path,
    train_history: list[float] | np.ndarray,
    real_counts: np.ndarray,
    synth_counts: np.ndarray,
    real_values: np.ndarray,
    synth_values: np.ndarray,
    county_names: list[str],
) -> None:
    output_dir = Path(output_dir)
    history = np.asarray(train_history, dtype=np.float64).tolist()
    indices = [
        idx
        for idx, name in enumerate(county_names)
        if display_county_name(name) in PLOT_COUNTIES
    ]
    if not indices:
        indices = list(range(len(county_names)))
    subset_county_names = [county_names[i] for i in indices]
    subset_real_counts = real_counts[:, indices]
    subset_synth_counts = synth_counts[:, indices]

    save_loss_plot(history, output_dir / "flow_loss.png")
    save_histograms(
        subset_real_counts,
        subset_synth_counts,
        subset_county_names,
        output_dir / "histograms.png",
    )
    save_marginal_distributions(
        subset_real_counts,
        subset_synth_counts,
        subset_county_names,
        output_dir / "marginal_distributions.png",
    )
    save_correlation_heatmaps(
        real_values,
        synth_values,
        county_names,
        output_dir / "correlation_heatmaps.png",
    )
