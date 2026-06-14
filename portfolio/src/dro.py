from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DroData:
    X_train: np.ndarray
    R_test: np.ndarray
    R_stress: np.ndarray
    center: np.ndarray
    scale: np.ndarray


def display_label(name: str) -> str:
    if name == "nominal":
        return "Nominal"
    if name == "equal_weight":
        return "Equal Weight"
    if name == "empirical":
        return "Empirical"
    if name.startswith("robust_gamma_"):
        gamma_value = name.removeprefix("robust_gamma_")
        return f"$\\gamma={gamma_value}$"
    if name.startswith("worst_case_gamma_"):
        gamma_value = name.removeprefix("worst_case_gamma_")
        return f"$\\gamma={gamma_value}$"
    return name.replace("_", " ")


class PortfolioModel:
    def __init__(
        self,
        q: float,
        beta: float,
        name: str = "",
        loss_scale: float = 1.0,
        center: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.q = q
        self.beta = beta
        self.loss_scale = loss_scale
        self.theta: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.history = pd.DataFrame()
        self.center = center
        self.scale = scale

    def loss_returns(self, X: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            return X
        return self.center[None, :] + X * self.scale[None, :]

    @property
    def w(self) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model parameters are not initialized.")
        return self.weights_from_theta(self.theta)

    @staticmethod
    def weights_from_theta(theta: np.ndarray) -> np.ndarray:
        shifted_logits = theta - np.max(theta)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    def shortfall_terms(
        self, returns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        margins = self.q - returns
        scaled_margins = self.beta * self.loss_scale * margins
        loss_terms = np.logaddexp(0.0, scaled_margins) / self.beta
        positive_mask = scaled_margins >= 0
        slope_terms = np.empty_like(scaled_margins, dtype=float)
        slope_terms[positive_mask] = self.loss_scale / (
            1.0 + np.exp(-scaled_margins[positive_mask])
        )
        exp_values = np.exp(scaled_margins[~positive_mask])
        slope_terms[~positive_mask] = self.loss_scale * exp_values / (1.0 + exp_values)
        return margins, loss_terms, slope_terms

    def nominal_batch_metrics_and_grads(
        self,
        X: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        w = self.w
        returns = self.loss_returns(X)
        _, loss_terms, slope_terms = self.shortfall_terms(returns @ w)
        g_w = -np.mean(slope_terms[:, None] * returns, axis=0)
        jacobian = np.diag(w) - np.outer(w, w)
        g_theta = jacobian @ g_w
        return float(np.mean(loss_terms)), g_theta

    def robust_batch_metrics_and_grads(
        self,
        theta_value: np.ndarray,
        v_penalty: np.ndarray,
        x_penalty: np.ndarray,
        gamma: float,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        w = self.weights_from_theta(theta_value)
        returns = self.loss_returns(v_penalty)
        _, loss_terms, slope_terms = self.shortfall_terms(returns @ w)
        penalty_deltas = v_penalty - x_penalty
        penalty_terms = 0.5 / gamma * np.sum(penalty_deltas**2, axis=1)
        g_w = -np.mean(slope_terms[:, None] * returns, axis=0)
        jacobian = np.diag(w) - np.outer(w, w)
        g_theta = jacobian @ g_w
        loss_jacobian = (
            w
            if self.scale is None
            else self.scale * w
        )
        g_v = -slope_terms[:, None] * loss_jacobian[None, :] - penalty_deltas / gamma
        return (
            float(np.mean(loss_terms)),
            float(np.mean(loss_terms - penalty_terms)),
            float(np.mean(np.linalg.norm(penalty_deltas, axis=1))),
            g_theta,
            g_v,
        )

    def compute_metrics(self, returns: np.ndarray) -> dict[str, float]:
        returns = np.asarray(returns)
        margins, loss_terms, _ = self.shortfall_terms(returns)
        return {
            "loss": float(np.mean(loss_terms)),
            "shortfall": float(np.mean(np.maximum(margins, 0.0))),
            "cvar_5": left_tail_average(returns, tail_fraction=0.05),
            "cvar_1": left_tail_average(returns, tail_fraction=0.01),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns, ddof=1)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
        }

    def evaluate(
        self,
        R_test: np.ndarray,
        R_stress: np.ndarray,
    ) -> pd.DataFrame:
        w = self.w
        test_returns = R_test @ w
        stress_returns = R_stress @ w

        rows = []
        for regime, metrics in [
            ("test", self.compute_metrics(test_returns)),
            ("stress_test", self.compute_metrics(stress_returns)),
        ]:
            row = {
                "model_key": self.name,
                "label": display_label(self.name),
                "regime": regime,
            }
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)


def left_tail_average(values: np.ndarray, tail_fraction: float) -> float:
    sorted_values = np.sort(np.asarray(values))
    tail_count = max(1, int(np.ceil(tail_fraction * len(sorted_values))))
    return float(sorted_values[:tail_count].mean())


def build_dro_data(
    data,
    training_returns: np.ndarray,
    stress_delta: float,
) -> DroData:
    center = data.center["center"].to_numpy(dtype=float)
    scale = data.scale["scale"].to_numpy(dtype=float)

    R_test = data.R_test.values
    training_returns = np.asarray(training_returns, dtype=float)
    X_train = (training_returns - center[None, :]) / scale[None, :]

    return DroData(
        X_train=X_train,
        R_test=R_test,
        R_stress=R_test - stress_delta * np.ones_like(R_test),
        center=center,
        scale=scale,
    )


def train_model(
    name: str,
    q: float,
    beta: float,
    loss_scale: float,
    X: np.ndarray,
    steps: int,
    tau: float,
    m: int | None,
    seed: int,
    theta_init: np.ndarray | None = None,
    gamma: float | None = None,
    eta: float | None = None,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> PortfolioModel:
    model = PortfolioModel(
        q=q,
        beta=beta,
        name=name,
        loss_scale=loss_scale,
        center=center,
        scale=scale,
    )
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if theta_init is None:
        model.theta = np.zeros(d, dtype=float)
    else:
        if theta_init.shape != (d,):
            raise ValueError(
                f"Initial theta has shape {theta_init.shape}, expected {(d,)}."
            )
        model.theta = theta_init.copy()
    V = X.copy() if gamma is not None else None
    history_rows = []
    rng = np.random.default_rng(seed)

    for k in range(1, steps + 1):
        if m is None or m >= n:
            idx = None
            x = X
        else:
            idx = rng.choice(n, size=m, replace=False)
            x = X[idx]

        if V is not None:
            v_penalty = V if idx is None else V[idx]
            loss, objective, mean_v_shift, g_theta, g_v = model.robust_batch_metrics_and_grads(
                theta_value=model.theta,
                v_penalty=v_penalty,
                x_penalty=x,
                gamma=gamma,
            )
            model.theta -= tau * g_theta
            v_penalty = v_penalty + eta * g_v
            if idx is None:
                V = v_penalty
            else:
                V[idx] = v_penalty
        else:
            loss, g_theta = model.nominal_batch_metrics_and_grads(x)
            model.theta -= tau * g_theta
        row = {
            "step": k,
            "loss": loss,
            "grad_theta_norm": float(np.linalg.norm(g_theta)),
        }
        if V is None:
            history_rows.append(row)
        else:
            row |= {
                "objective": objective,
                "grad_v_norm": float(np.sqrt(np.mean(np.sum(g_v**2, axis=1)))),
                "mean_v_shift_l2": mean_v_shift,
            }
            history_rows.append(row)

    model.history = pd.DataFrame(history_rows)
    model.V = V
    return model


def train_portfolio_models(
    args,
    X_train: np.ndarray,
    loss_scale: float,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[PortfolioModel, list[PortfolioModel], dict[str, float | bool | str | None]]:
    nominal_start = time.perf_counter()
    nominal_model = train_model(
        name="nominal",
        q=args.q,
        beta=args.beta,
        loss_scale=loss_scale,
        X=X_train,
        steps=args.nominal_steps,
        tau=args.tau_nominal,
        m=args.batch_size,
        seed=args.seed,
        center=center,
        scale=scale,
    )
    nominal_elapsed = time.perf_counter() - nominal_start

    robust_start = time.perf_counter()
    robust_models = []
    theta_init = np.zeros(X_train.shape[1], dtype=float)
    for gamma_index, gamma_value in enumerate(args.gammas):
        print(f"    gamma={gamma_value}, tau={args.tau_robust:.3e}, eta={args.eta:.3e}")
        robust_models.append(
            train_model(
                name=f"robust_gamma_{gamma_value:g}",
                q=args.q,
                beta=args.beta,
                loss_scale=loss_scale,
                X=X_train,
                steps=args.robust_steps,
                tau=args.tau_robust,
                m=args.batch_size,
                seed=args.seed + 1000 + gamma_index,
                theta_init=theta_init,
                gamma=gamma_value,
                eta=args.eta,
                center=center,
                scale=scale,
            )
        )
    robust_elapsed = time.perf_counter() - robust_start
    return (
        nominal_model,
        robust_models,
        {
            "nominal_seconds": float(nominal_elapsed),
            "robust_seconds": float(robust_elapsed),
            "portfolio_total_seconds": float(nominal_elapsed + robust_elapsed),
        },
    )
