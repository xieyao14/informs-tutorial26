import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data import (
    DEFAULT_TICKERS,
    build_return_matrix,
    split_and_scale,
)
from src.flow import FlowMLP, compute_rbf_mmd2, sample_euler
from src.dro import (
    PortfolioModel,
    build_dro_data,
    display_label,
    train_portfolio_models,
)
from src.viz import (
    plot_flow_outputs,
    plot_convergence_histories,
    plot_cumulative_returns,
    plot_left_tail_histogram,
    plot_nominal_history,
    plot_distributions,
)

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
TRAIN_END_DATE = "2024-12-31"
Q = 0.02
BETA = 4.0
GAMMAS = [0.01, 0.1, 1.0, 10.0]
NOMINAL_STEPS = 100000
ROBUST_STEPS = 100000
TAU_NOMINAL = 1.0
TAU_ROBUST = 1.0
ETA = 0.01
BATCH_SIZE = None
SEED = 123
STRESS_DELTA = 0.001
FLOW_WIDTH = 256
FLOW_DEPTH = 4
FLOW_TIME_DIM = 64
FLOW_EPOCHS = 20000
FLOW_BATCH_SIZE = 1024
FLOW_LR = 1e-3
FLOW_WEIGHT_DECAY = 1e-5
FLOW_GRAD_CLIP = 1.0
FLOW_STEPS = 100
FLOW_LOG_INTERVAL = 500
DATA_DIR = Path(__file__).resolve().parent / "data" / "yfinance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio experiment with a flow-matching nominal distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    return parser.parse_args()


def apply_settings(args: argparse.Namespace) -> None:
    args.data_dir = str(DATA_DIR.relative_to(Path(__file__).resolve().parent))
    args.output_dir = args.output_dir or "results/default"
    args.tickers = DEFAULT_TICKERS
    args.start = START_DATE
    args.end = END_DATE
    args.train_end = TRAIN_END_DATE
    args.q = Q
    args.beta = BETA
    args.gammas = list(GAMMAS)
    args.nominal_steps = NOMINAL_STEPS
    args.robust_steps = ROBUST_STEPS
    args.tau_nominal = TAU_NOMINAL
    args.tau_robust = TAU_ROBUST
    args.eta = ETA
    args.batch_size = BATCH_SIZE
    args.seed = SEED
    args.stress_delta = STRESS_DELTA
    args.flow_width = FLOW_WIDTH
    args.flow_depth = FLOW_DEPTH
    args.flow_time_dim = FLOW_TIME_DIM
    args.flow_epochs = FLOW_EPOCHS
    args.flow_batch_size = FLOW_BATCH_SIZE
    args.flow_lr = FLOW_LR
    args.flow_weight_decay = FLOW_WEIGHT_DECAY
    args.flow_grad_clip = FLOW_GRAD_CLIP
    args.flow_steps = FLOW_STEPS
    args.flow_log_interval = FLOW_LOG_INTERVAL
    args.scaling = "global_mean_abs_return"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    apply_settings(args)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    for directory in [output_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading data...")
    _, returns = build_return_matrix(
        tickers=args.tickers,
        data_dir=DATA_DIR,
    )

    print("[2/8] Splitting and scaling...")
    data = split_and_scale(
        R=returns,
        start_date=args.start,
        end_date=args.end,
        train_end_date=args.train_end,
    )
    args.global_scale = float(data.scale["scale"].iloc[0])
    pd.Series(vars(args).copy()).to_json(output_dir / "config.json", indent=2)

    print("[3/8] Training flow model...")
    train_tensor = torch.from_numpy(
        data.X_train.to_numpy(dtype=np.float32, copy=True)
    )
    train_loader = DataLoader(
        train_tensor,
        batch_size=args.flow_batch_size,
        shuffle=True,
        drop_last=False,
    )
    dim = train_tensor.shape[1]
    flow_model = FlowMLP(
        dim=dim,
        hidden_width=args.flow_width,
        hidden_depth=args.flow_depth,
        time_embed_dim=args.flow_time_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=args.flow_lr,
        weight_decay=args.flow_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.flow_epochs * len(train_loader))
    )
    flow_history = []
    global_step = 0
    for epoch in range(args.flow_epochs):
        flow_model.train()
        for x0 in train_loader:
            x0 = x0.to(device)
            x1 = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=device)
            xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            target = x1 - x0

            pred = flow_model(xt, t)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(flow_model.parameters(), args.flow_grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            flow_history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if global_step % args.flow_log_interval == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": round(float(loss.item()), 6),
                            "device": str(device),
                        }
                    ),
                    flush=True,
                )
    print("[4/8] Sampling synthetic training set...")
    samples = sample_euler(
        flow_model,
        n_samples=len(data.X_train),
        dim=dim,
        device=device,
        n_steps=args.flow_steps,
        seed=args.seed + 17,
    ).cpu().numpy()
    synth_train = pd.DataFrame(samples, columns=data.X_train.columns)
    synth_returns = pd.DataFrame(
        data.center["center"].to_numpy(dtype=float)[None, :]
        + samples * data.scale["scale"].to_numpy(dtype=float)[None, :],
        columns=data.X_train.columns,
    )
    flow_history_frame = pd.DataFrame(flow_history)
    flow_history_frame.to_csv(output_dir / "flow_loss.csv", index=False)
    synth_returns.to_csv(output_dir / "nominal.csv", index=False)
    real_tensor = torch.from_numpy(data.X_train.to_numpy(dtype=np.float32, copy=True))
    synth_tensor = torch.from_numpy(samples.astype(np.float32, copy=True))
    mmd2_names = []
    mmd2_values = []
    for column_index, column_name in enumerate(data.X_train.columns):
        mmd2_names.append(column_name)
        mmd2_values.append(
            compute_rbf_mmd2(
                real_tensor[:, column_index : column_index + 1],
                synth_tensor[:, column_index : column_index + 1],
                seed=column_index,
            )
        )
    joint_mmd2 = compute_rbf_mmd2(
        real_tensor,
        synth_tensor,
        seed=args.seed,
    )
    mmd2_names.append("joint")
    mmd2_values.append(float(joint_mmd2))

    dro_data = build_dro_data(
        data=data,
        training_returns=synth_returns.values,
        stress_delta=args.stress_delta,
    )
    mean_abs_return = float(np.mean(np.abs(data.R_train.to_numpy(dtype=float))))
    loss_scale = (
        1.0 / mean_abs_return
        if np.isfinite(mean_abs_return) and mean_abs_return > 0.0
        else 1.0
    )

    print("[5/8] Training nominal and robust portfolios...")
    nominal_model, robust_models, portfolio_training_times = train_portfolio_models(
        args=args,
        X_train=dro_data.X_train,
        loss_scale=loss_scale,
        center=dro_data.center,
        scale=dro_data.scale,
    )
    print(
        f"    nominal training time: {portfolio_training_times['nominal_seconds']:.2f}s"
    )
    print(
        f"    robust training time: {portfolio_training_times['robust_seconds']:.2f}s"
    )

    print("[6/8] Evaluating portfolios...")
    print("[7/8] Saving figures and outputs...")
    training_datasets = {
        "empirical": data.R_train,
        "nominal": synth_returns,
    }
    worst_case_rows = []
    for model in robust_models:
        if model.V is not None:
            gamma_label = model.name.replace("robust_", "")
            worst_case_returns = (
                dro_data.center[None, :]
                + model.V * dro_data.scale[None, :]
            )
            training_datasets[f"worst_case_{gamma_label}"] = pd.DataFrame(
                worst_case_returns,
                columns=synth_train.columns,
            )
            row_frame = training_datasets[f"worst_case_{gamma_label}"].copy()
            row_frame.insert(0, "sample", np.arange(len(row_frame)))
            row_frame.insert(0, "scenario", f"worst_case_{gamma_label}")
            worst_case_rows.append(row_frame)
    if worst_case_rows:
        pd.concat(worst_case_rows, ignore_index=True).to_csv(
            output_dir / "worst_case.csv", index=False
        )

    equal_weight_model = PortfolioModel(
        q=args.q,
        beta=args.beta,
        name="equal_weight",
        loss_scale=loss_scale,
        center=dro_data.center,
        scale=dro_data.scale,
    )
    equal_weight_model.theta = np.zeros(len(args.tickers), dtype=float)
    all_models = [equal_weight_model, nominal_model] + robust_models
    dro_loss_rows = []
    for model in all_models:
        if not model.history.empty:
            dro_loss_rows.append(model.history.assign(model_key=model.name))
    if dro_loss_rows:
        dro_loss = pd.concat(dro_loss_rows, ignore_index=True)
        dro_loss = dro_loss[
            ["model_key", *[c for c in dro_loss.columns if c != "model_key"]]
        ]
        dro_loss.to_csv(output_dir / "dro_loss.csv", index=False)

    weight_rows = []
    for model in all_models:
        row = {"model_key": model.name, "label": display_label(model.name)}
        row.update(
            {ticker: float(weight) for ticker, weight in zip(args.tickers, model.w)}
        )
        weight_rows.append(row)
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(output_dir / "weights.csv", index=False)

    portfolio_metrics = pd.concat(
        [
            model.evaluate(
                R_test=dro_data.R_test,
                R_stress=dro_data.R_stress,
            )
            for model in all_models
        ],
        ignore_index=True,
    )
    metric_rows = [
        {
            "section": "flow",
            "model_key": "flow",
            "label": "Flow",
            "regime": "train",
            "series": series_name,
            "metric": "mmd2",
            "value": value,
        }
        for series_name, value in zip(mmd2_names, mmd2_values)
    ]
    portfolio_metric_columns = [
        column
        for column in portfolio_metrics.columns
        if column not in {"model_key", "label", "regime"}
    ]
    for _, row in portfolio_metrics.iterrows():
        for metric_name in portfolio_metric_columns:
            metric_rows.append(
                {
                    "section": "portfolio",
                    "model_key": row["model_key"],
                    "label": row["label"],
                    "regime": row["regime"],
                    "series": "",
                    "metric": metric_name,
                    "value": row[metric_name],
                }
            )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)

    plot_flow_outputs(
        output_dir=str(output_dir),
        dpi=160,
        real_scaled=data.X_train,
        generated_scaled=synth_train,
    )

    model_weights = {model.name: model.w for model in all_models}
    plot_cumulative_returns(
        R_test=data.R_test,
        R_stress=dro_data.R_stress,
        model_weights=model_weights,
        output_path=figures_dir / "test_cumulative_wealth.png",
        dpi=160,
    )
    plot_left_tail_histogram(
        R_test=data.R_test,
        model_weights=model_weights,
        output_path=figures_dir / "test_left_tail_histogram.png",
        dpi=160,
    )
    plot_convergence_histories(
        models=all_models,
        output_path=figures_dir / "robust_training_convergence.png",
        dpi=160,
    )
    plot_nominal_history(
        nominal_model=nominal_model,
        output_path=figures_dir / "nominal_training_convergence.png",
        dpi=160,
    )
    plot_distributions(
        datasets=training_datasets,
        output_path=figures_dir / "distribution_comparison.png",
        dpi=160,
        color_map={"empirical": "gray", "nominal": "black"},
        label_map={"nominal": "Nominal"},
    )

    print("[8/8] Done.")
    print(
        "Saved weights to:",
        output_dir / "weights.csv",
    )
    print(
        "Saved metrics to:",
        output_dir / "metrics.csv",
    )
    print("Saved figures to:", figures_dir)
    print("\nPortfolio weights:")
    display_weights = weights.drop(columns=["model_key"]).rename(
        columns={"label": "model"}
    )
    print(display_weights.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nEvaluation metrics:")
    display_metrics = portfolio_metrics.drop(columns=["model_key"]).rename(
        columns={"label": "model"}
    )
    print(
        display_metrics.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )


if __name__ == "__main__":
    main()
