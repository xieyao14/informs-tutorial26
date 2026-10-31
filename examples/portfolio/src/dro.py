"""Compute portfolio losses and gradients, optimize portfolios, and evaluate results."""

import time

import numpy as np
import pandas as pd


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
    """Long-only portfolio with softmax weights and softplus shortfall loss."""

    def __init__(
        self,
        q: float,
        beta: float,
        loss_scale: float,
        center: np.ndarray,
        scale: float,
        name: str = "",
    ) -> None:
        self.name = name
        self.q = q
        self.beta = beta
        self.loss_scale = loss_scale
        self.theta = np.zeros(len(center), dtype=float)
        self.z: np.ndarray | None = None
        self.history = pd.DataFrame()
        self.center = center
        self.scale = scale

    # === Scenario coordinates and portfolio weights ===
    def to_returns(self, X: np.ndarray) -> np.ndarray:
        """Map globally scaled scenarios back to raw asset returns."""
        return self.center[None, :] + X * self.scale

    @property
    def w(self) -> np.ndarray:
        return self.weights_from_theta(self.theta)

    @staticmethod
    def weights_from_theta(theta: np.ndarray) -> np.ndarray:
        """Use softmax to obtain nonnegative weights that sum to one."""
        shifted_logits = theta - np.max(theta)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    # === Nominal and robust losses and gradients ===
    def shortfall_terms(
        self, returns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return shortfall margins, smooth losses, and derivatives with respect to margins."""
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

    def nominal_loss_and_grad(
        self,
        X: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return mean softplus loss and its gradient with respect to theta."""
        w = self.w
        returns = self.to_returns(X)
        _, loss_terms, slope_terms = self.shortfall_terms(returns @ w)
        g_w = -np.mean(slope_terms[:, None] * returns, axis=0)
        jacobian = np.diag(w) - np.outer(w, w)
        g_theta = jacobian @ g_w
        return float(np.mean(loss_terms)), g_theta

    def robust_loss_and_grad(
        self,
        theta: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        gamma: float,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Return mean loss, penalized objective, mean displacement, and both gradients."""
        w = self.weights_from_theta(theta)
        returns = self.to_returns(z)
        _, loss_terms, slope_terms = self.shortfall_terms(returns @ w)
        penalty_deltas = z - x
        penalty_terms = 0.5 / gamma * np.sum(penalty_deltas**2, axis=1)
        g_w = -np.mean(slope_terms[:, None] * returns, axis=0)
        jacobian = np.diag(w) - np.outer(w, w)
        g_theta = jacobian @ g_w
        loss_jacobian = self.scale * w
        # Each particle maximizes its own penalized loss; theta minimizes the average.
        g_z = -slope_terms[:, None] * loss_jacobian[None, :] - penalty_deltas / gamma
        return (
            float(np.mean(loss_terms)),
            float(np.mean(loss_terms - penalty_terms)),
            float(np.mean(np.linalg.norm(penalty_deltas, axis=1))),
            g_theta,
            g_z,
        )

    # === Portfolio evaluation ===
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
    tail_count = int(np.ceil(tail_fraction * len(sorted_values)))
    return float(sorted_values[:tail_count].mean())


# === Portfolio optimization ===
def optimize_portfolio(
    name: str,
    q: float,
    beta: float,
    loss_scale: float,
    X: np.ndarray,
    steps: int,
    tau: float,
    center: np.ndarray,
    scale: float,
    gamma: float | None = None,
    eta: float | None = None,
) -> PortfolioModel:
    """Use gradient descent for nominal optimization and simultaneous descent-ascent for DRO."""
    model = PortfolioModel(
        q=q,
        beta=beta,
        name=name,
        loss_scale=loss_scale,
        center=center,
        scale=scale,
    )
    X = np.asarray(X, dtype=float)
    z = X.copy() if gamma is not None else None
    history_rows = []
    for k in range(1, steps + 1):
        if z is not None:
            loss, objective, mean_z_shift, g_theta, g_z = model.robust_loss_and_grad(
                theta=model.theta,
                z=z,
                x=X,
                gamma=gamma,
            )
            # Both gradients are evaluated before either variable is updated.
            model.theta -= tau * g_theta
            z = z + eta * g_z
        else:
            loss, g_theta = model.nominal_loss_and_grad(X)
            model.theta -= tau * g_theta
        row = {
            "step": k,
            "loss": loss,
            "grad_theta_norm": float(np.linalg.norm(g_theta)),
        }
        if z is None:
            history_rows.append(row)
        else:
            row |= {
                "objective": objective,
                "grad_z_norm": float(np.sqrt(np.mean(np.sum(g_z**2, axis=1)))),
                "mean_z_shift_l2": mean_z_shift,
            }
            history_rows.append(row)

    model.history = pd.DataFrame(history_rows)
    model.z = z
    return model


def optimize_portfolios(
    args,
    X_train: np.ndarray,
    loss_scale: float,
    center: np.ndarray,
    scale: float,
) -> tuple[PortfolioModel, list[PortfolioModel], dict[str, float]]:
    """Optimize the nominal portfolio, then one robust portfolio for each gamma."""
    nominal_start = time.perf_counter()
    nominal_model = optimize_portfolio(
        name="nominal",
        q=args.q,
        beta=args.beta,
        loss_scale=loss_scale,
        X=X_train,
        steps=args.nominal_steps,
        tau=args.tau_nominal,
        center=center,
        scale=scale,
    )
    nominal_elapsed = time.perf_counter() - nominal_start

    robust_start = time.perf_counter()
    robust_models = []
    for gamma_value in args.gammas:
        print(f"    gamma={gamma_value}, tau={args.tau_robust:.3e}, eta={args.eta:.3e}")
        robust_models.append(
            optimize_portfolio(
                name=f"robust_gamma_{gamma_value:g}",
                q=args.q,
                beta=args.beta,
                loss_scale=loss_scale,
                X=X_train,
                steps=args.robust_steps,
                tau=args.tau_robust,
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
