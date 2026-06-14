import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data import DEFAULT_COUNTIES, load_outage_splits
from src.flow import FlowMLP, compute_rbf_mmd2, sample_euler
from src.viz import save_figures

COUNTY_FIPS = [fips for fips, _ in DEFAULT_COUNTIES]
TRAIN_START = "2018-01-01T00:00:00"
TRAIN_END = "2024-12-31T23:45:00"
VAL_START = "2018-01-01T00:00:00"
VAL_END = "2024-12-31T23:45:00"
FLOW_WIDTH = 256
FLOW_DEPTH = 4
FLOW_TIME_DIM = 64
SEED = 42
FLOW_LR = 1e-3
FLOW_WEIGHT_DECAY = 1e-5
FLOW_GRAD_CLIP = 1.0
FLOW_BATCH_SIZE = 1024
FLOW_EPOCHS = 100
FLOW_LOG_INTERVAL = 50
FLOW_NUM_SAMPLES = 4096
FLOW_STEPS = 100
DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outage flow-matching reproduction",
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
    args.county_fips = COUNTY_FIPS
    args.train_start = TRAIN_START
    args.train_end = TRAIN_END
    args.val_start = VAL_START
    args.val_end = VAL_END
    args.seed = SEED
    args.flow_width = FLOW_WIDTH
    args.flow_depth = FLOW_DEPTH
    args.flow_time_dim = FLOW_TIME_DIM
    args.flow_epochs = FLOW_EPOCHS
    args.flow_batch_size = FLOW_BATCH_SIZE
    args.flow_lr = FLOW_LR
    args.flow_weight_decay = FLOW_WEIGHT_DECAY
    args.flow_grad_clip = FLOW_GRAD_CLIP
    args.flow_steps = FLOW_STEPS
    args.flow_num_samples = FLOW_NUM_SAMPLES
    args.flow_log_interval = FLOW_LOG_INTERVAL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val_flow_loss(
    flow_model: FlowMLP,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    flow_model.eval()
    losses = []
    with torch.no_grad():
        for x0 in val_loader:
            x0 = x0.to(device)
            x1 = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=device)
            xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            target = x1 - x0
            pred = flow_model(xt, t)
            losses.append(torch.mean((pred - target) ** 2).item())
    return float(np.mean(losses))


def main() -> None:
    t0 = time.perf_counter()
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

    splits = load_outage_splits(
        data_root=DATA_DIR,
        county_fips=args.county_fips,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
    )

    train_tensor = torch.from_numpy(
        splits["train_standardized"].astype(np.float32, copy=False)
    )
    val_tensor = torch.from_numpy(
        splits["val_standardized"].astype(np.float32, copy=False)
    )
    train_loader = DataLoader(
        train_tensor,
        batch_size=args.flow_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_tensor,
        batch_size=args.flow_batch_size,
        shuffle=False,
        drop_last=False,
    )

    dim = train_tensor.shape[1]
    flow_model = FlowMLP(
        dim=dim,
        hidden_width=args.flow_width,
        hidden_depth=args.flow_depth,
        time_embed_dim=args.flow_time_dim,
    )
    flow_model = flow_model.to(device)

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

    flow_history_frame = pd.DataFrame(flow_history)
    loss_values = flow_history_frame["loss"].to_numpy(dtype=float)
    val_loss = val_flow_loss(flow_model, val_loader, device)
    synth_log = sample_euler(
        flow_model,
        n_samples=args.flow_num_samples,
        dim=dim,
        device=device,
        n_steps=args.flow_steps,
        seed=args.seed + 17,
    ).cpu()
    real_log = val_tensor.cpu()
    mean = torch.from_numpy(splits["train_mean"]).float()
    std = torch.from_numpy(splits["train_std"]).float()
    synth_log = synth_log * std + mean
    real_log = real_log * std + mean
    synth_counts = np.rint(torch.expm1(synth_log).clamp_min(0.0).numpy()).clip(min=0.0)
    real_counts = torch.expm1(real_log).clamp_min(0.0).numpy()
    synth_log_eval = torch.log1p(torch.from_numpy(synth_counts).float())
    mmd2 = compute_rbf_mmd2(
        real_log.to(device),
        synth_log_eval.to(device),
        seed=args.seed,
    )
    resolved_config = vars(args).copy()
    resolved_config["device"] = str(device)

    county_names = splits["county_names"]
    pd.DataFrame(synth_counts, columns=county_names).to_csv(
        output_dir / "nominal.csv", index=False
    )
    flow_history_frame.to_csv(output_dir / "flow_loss.csv", index=False)

    save_figures(
        output_dir=figures_dir,
        train_history=loss_values,
        real_counts=real_counts,
        synth_counts=synth_counts,
        real_values=real_log.numpy(),
        synth_values=synth_log_eval.numpy(),
        county_names=county_names,
    )

    metadata = {
        "county_names": county_names,
        "train_size": int(len(train_tensor)),
        "val_size": int(len(val_tensor)),
        "standardize": True,
        "n_real_eval_samples": int(real_log.shape[0]),
        "n_synthetic_eval_samples": int(args.flow_num_samples),
        "synthetic_counts_rounded": True,
    }
    metrics = {
        "final_train_loss": float(loss_values[-1]),
        "mean_train_loss": float(np.mean(loss_values)),
        "val_flow_loss": float(val_loss),
        "mmd2_log1p": float(mmd2),
        "n_synthetic_samples": int(args.flow_num_samples),
        "device": str(device),
        "runtime_seconds": float(time.perf_counter() - t0),
    }

    metrics_frame = pd.DataFrame(
        [
            {
                "section": "flow",
                "model_key": "flow",
                "label": "Flow",
                "regime": "validation",
                "series": "",
                "metric": name,
                "value": value,
            }
            for name, value in metrics.items()
            if name != "device"
        ]
    )
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config | metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
