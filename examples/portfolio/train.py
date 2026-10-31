"""Train the flow model and optimize portfolios; save results for plot.py."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.data import (
    DEFAULT_TICKERS,
    load_returns,
    split_and_scale,
)
from src.dro import (
    PortfolioModel,
    display_label,
    optimize_portfolios,
)
from src.flow import FlowMLP, compute_rbf_mmd2, sample_euler
from torch.nn.utils import clip_grad_norm_

# === Experiment settings ===
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
        default="results",
        help="Directory for outputs",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def experiment_settings() -> dict:
    return {
        "data_dir": str(DATA_DIR.relative_to(Path(__file__).resolve().parent)),
        "tickers": DEFAULT_TICKERS,
        "start": START_DATE,
        "end": END_DATE,
        "train_end": TRAIN_END_DATE,
        "q": Q,
        "beta": BETA,
        "gammas": list(GAMMAS),
        "nominal_steps": NOMINAL_STEPS,
        "robust_steps": ROBUST_STEPS,
        "tau_nominal": TAU_NOMINAL,
        "tau_robust": TAU_ROBUST,
        "eta": ETA,
        "seed": SEED,
        "stress_delta": STRESS_DELTA,
        "flow_width": FLOW_WIDTH,
        "flow_depth": FLOW_DEPTH,
        "flow_time_dim": FLOW_TIME_DIM,
        "flow_epochs": FLOW_EPOCHS,
        "flow_batch_size": FLOW_BATCH_SIZE,
        "flow_lr": FLOW_LR,
        "flow_weight_decay": FLOW_WEIGHT_DECAY,
        "flow_grad_clip": FLOW_GRAD_CLIP,
        "flow_steps": FLOW_STEPS,
        "flow_log_interval": FLOW_LOG_INTERVAL,
        "flow_scaling": "assetwise_standard_deviation",
        "dro_scaling": "global_mean_abs_return",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = argparse.Namespace(**{**experiment_settings(), **vars(parse_args())})
    device = torch.device(args.device)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Data preparation ===
    # Load aligned returns and estimate training-period scales in src/data.py.
    print("[1/8] Loading data...")
    returns = load_returns(
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
    args.global_scale = data.scale
    args.flow_scale = data.flow_scale.to_dict()

    # === Flow model ===
    # Build the velocity field (src/flow.py) and its optimizer.
    print("[3/8] Training flow model...")
    train_tensor = torch.from_numpy(
        data.X_train.to_numpy(dtype=np.float32, copy=True)
    ).to(device)
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
    batch_size = min(args.flow_batch_size, len(train_tensor))
    steps_per_epoch = (len(train_tensor) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.flow_epochs * steps_per_epoch
    )
    rng = torch.Generator(device=device).manual_seed(args.seed)
    # === Flow training ===
    # Fit the velocity field to x1 - x0 along straight interpolation paths.
    flow_history = []
    global_step = 0
    for epoch in range(args.flow_epochs):
        flow_model.train()
        order = torch.randperm(len(train_tensor), generator=rng, device=device)
        for start in range(0, len(train_tensor), batch_size):
            x0 = train_tensor[order[start : start + batch_size]]
            x1 = torch.randn(x0.shape, generator=rng, device=device)
            t = torch.rand(x0.shape[0], generator=rng, device=device)
            # Interpolate from data at t=0 to Gaussian noise at t=1.
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

    # === Scenario generation ===
    # Generate scenarios and undo assetwise standardization to recover raw returns.
    print("[4/8] Sampling synthetic training set...")
    samples = sample_euler(
        flow_model,
        n_samples=len(data.X_train),
        dim=dim,
        device=device,
        n_steps=args.flow_steps,
        seed=args.seed + 17,
    ).cpu().numpy()
    synth_returns = pd.DataFrame(
        data.center.to_numpy(dtype=float)[None, :]
        + samples.astype(float) * data.flow_scale.to_numpy(dtype=float)[None, :],
        columns=data.X_train.columns,
    )

    # === Distribution comparison ===
    # Compare generated and real returns in the flow's standardized coordinates.
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

    # === Portfolio optimization ===
    # Optimize all portfolios on the same nominal scenarios in src/dro.py.
    center = data.center.to_numpy(dtype=float)
    loss_scale = 1.0 / data.scale
    # Convert returns to the globally scaled coordinates used by the transport penalty.
    nominal_scaled = (synth_returns.to_numpy(dtype=float) - center[None, :]) / data.scale

    print("[5/8] Training nominal and robust portfolios...")
    nominal_model, robust_models, portfolio_training_times = optimize_portfolios(
        args=args,
        X_train=nominal_scaled,
        loss_scale=loss_scale,
        center=center,
        scale=data.scale,
    )
    print(
        f"    nominal training time: {portfolio_training_times['nominal_seconds']:.2f}s"
    )
    print(
        f"    robust training time: {portfolio_training_times['robust_seconds']:.2f}s"
    )

    # === Portfolio evaluation ===
    # Compare fitted portfolios and the equal-weight baseline on both test regimes.
    print("[6/8] Evaluating portfolios...")
    R_test = data.R_test.to_numpy(dtype=float)
    R_stress = R_test - args.stress_delta
    equal_weight_model = PortfolioModel(
        q=args.q,
        beta=args.beta,
        name="equal_weight",
        loss_scale=loss_scale,
        center=center,
        scale=data.scale,
    )
    all_models = [equal_weight_model, nominal_model] + robust_models
    portfolio_metrics = pd.concat(
        [
            model.evaluate(
                R_test=R_test,
                R_stress=R_stress,
            )
            for model in all_models
        ],
        ignore_index=True,
    )

    # === Save results ===
    # Save samples, training histories, settings, weights, and metrics for plot.py.
    print("[7/8] Saving outputs...")
    pd.Series(vars(args).copy()).to_json(output_dir / "config.json", indent=2)
    pd.DataFrame(flow_history).to_csv(output_dir / "flow_loss.csv", index=False)
    synth_returns.to_csv(output_dir / "nominal.csv", index=False)
    worst_case_rows = []
    for model in robust_models:
        gamma_label = model.name.replace("robust_", "")
        worst_case_returns = center[None, :] + model.z * data.scale
        row_frame = pd.DataFrame(worst_case_returns, columns=data.X_train.columns)
        row_frame.insert(0, "sample", np.arange(len(row_frame)))
        row_frame.insert(0, "scenario", f"worst_case_{gamma_label}")
        worst_case_rows.append(row_frame)
    pd.concat(worst_case_rows, ignore_index=True).to_csv(
        output_dir / "worst_case.csv", index=False
    )

    dro_loss = pd.concat(
        [model.history.assign(model_key=model.name) for model in [nominal_model] + robust_models],
        ignore_index=True,
    )
    dro_loss = dro_loss[["model_key", *[c for c in dro_loss.columns if c != "model_key"]]]
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

    print("[8/8] Done.")
    print(
        "Saved weights to:",
        output_dir / "weights.csv",
    )
    print(
        "Saved metrics to:",
        output_dir / "metrics.csv",
    )
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
