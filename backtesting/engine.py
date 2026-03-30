"""
backtesting/engine.py — Historical simulation engine for the Portfolio Optimizer.

Replays a price-feature array, calls the trained RL agent at each rebalance
step, and computes standard portfolio performance statistics.

This backtester is a simplified educational simulation.  It does not account
for market impact, partial fills, dividends, or corporate actions.

For educational and demonstration purposes only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from config import TICKERS, WINDOW_SIZE

if TYPE_CHECKING:
    from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for backtest performance statistics.

    All return/ratio values are scalars.  ``portfolio_values`` and
    ``cumulative_returns`` are time series aligned to the rebalance dates.
    """

    portfolio_values: list[float] = field(default_factory=list)
    weights_history: list[list[float]] = field(default_factory=list)
    period_returns: list[float] = field(default_factory=list)
    cumulative_returns: list[float] = field(default_factory=list)
    benchmark_cumulative_returns: list[float] = field(default_factory=list)

    # Scalar statistics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict (compatible with Pydantic / JSON)."""
        return {
            "portfolio_values": self.portfolio_values,
            "weights_history": self.weights_history,
            "period_returns": self.period_returns,
            "cumulative_returns": self.cumulative_returns,
            "benchmark_cumulative_returns": self.benchmark_cumulative_returns,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "calmar_ratio": self.calmar_ratio,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulates portfolio rebalancing over a historical feature dataset.

    Parameters
    ----------
    price_data:
        Feature array of shape ``(T, num_assets, num_features)`` as produced
        by ``data_pipeline.ingestion.yfinance_fetcher.compute_features``.
    tickers:
        Asset names for reporting (must match ``price_data`` asset dimension).
    rebalance_freq:
        How often to rebalance.  ``"daily"`` steps every period; ``"weekly"``
        and ``"monthly"`` step every 5 and 21 periods respectively.
    transaction_cost:
        Proportional cost on absolute weight changes per rebalance.
    initial_capital:
        Starting portfolio value.
    """

    _FREQ_MAP = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
    }

    def __init__(
        self,
        price_data: np.ndarray,
        scaled_data: np.ndarray | None = None,
        tickers: list[str] = TICKERS,
        rebalance_freq: str = "daily",
        transaction_cost: float = 0.001,
        initial_capital: float = 10_000.0,
    ) -> None:
        if price_data.ndim != 3:
            raise ValueError(
                f"price_data must be 3-D (T, num_assets, num_features), "
                f"got {price_data.shape}."
            )

        self.price_data = price_data.astype(np.float32)
        self.scaled_data = scaled_data.astype(np.float32) if scaled_data is not None else self.price_data
        self.T, self.num_assets, self.num_feats = price_data.shape
        self.rebalance_step = self._FREQ_MAP.get(rebalance_freq, 1)
        self.transaction_cost = transaction_cost
        self.initial_capital = float(initial_capital)

        if len(tickers) != self.num_assets:
            raise ValueError(
                f"len(tickers)={len(tickers)} != price_data.num_assets={self.num_assets}."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        agent: "PPO | None" = None,
        predict_fn: Callable | None = None,
    ) -> BacktestResult:
        """Simulate portfolio rebalancing and return performance statistics.

        Exactly one of *agent* or *predict_fn* must be provided.

        Parameters
        ----------
        agent:
            Trained SB3 PPO agent.  The engine will call
            ``agent.predict(obs, deterministic=True)`` at each rebalance step.
        predict_fn:
            Callable ``(obs: np.ndarray) -> np.ndarray`` returning raw logits.
            Use for custom strategies or equal-weight benchmarks.

        Returns
        -------
        BacktestResult
            Populated with time series and scalar statistics.
        """
        if agent is None and predict_fn is None:
            raise ValueError("Provide either 'agent' or 'predict_fn'.")

        result = BacktestResult()
        portfolio_value = self.initial_capital
        weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets

        result.portfolio_values.append(portfolio_value)
        result.weights_history.append(weights.tolist())

        # Step through data starting after the first valid window
        for t in range(WINDOW_SIZE, self.T):
            if (t - WINDOW_SIZE) % self.rebalance_step != 0:
                continue

            # --- Observation ---
            obs = self._build_observation(t)

            # --- Predict weights ---
            if agent is not None:
                action, _ = agent.predict(obs[np.newaxis, ...], deterministic=True)
                new_weights = self._softmax(action.flatten())
            else:
                raw = predict_fn(obs)
                new_weights = self._softmax(raw)

            # --- Transaction cost ---
            tc = self.transaction_cost * np.abs(new_weights - weights).sum()

            # --- Period return ---
            period_return = self._simulate_period_return(new_weights, t) - tc

            # --- Update state ---
            portfolio_value *= (1.0 + period_return)
            weights = new_weights

            result.portfolio_values.append(portfolio_value)
            result.weights_history.append(weights.tolist())
            result.period_returns.append(float(period_return))

        # --- Compute statistics ---
        stats = self._compute_statistics(result.portfolio_values, result.period_returns)
        for key, value in stats.items():
            setattr(result, key, value)

        # --- Cumulative returns ---
        pv = np.array(result.portfolio_values)
        result.cumulative_returns = (pv / pv[0] - 1.0).tolist()

        # --- Benchmark ---
        result.benchmark_cumulative_returns = self.compute_benchmark()

        logger.info(
            "Backtest complete. Steps=%d | Sharpe=%.3f | MaxDD=%.3f | TotalReturn=%.3f",
            len(result.period_returns),
            result.sharpe_ratio,
            result.max_drawdown,
            result.total_return,
        )
        return result

    def compute_benchmark(self) -> list[float]:
        """Compute equal-weight buy-and-hold cumulative returns.

        Returns
        -------
        list[float]
            Cumulative return series (starting at 0.0) for the equal-weight portfolio.
        """
        equal_weights = np.ones(self.num_assets) / self.num_assets
        portfolio = 1.0
        values = [portfolio]

        for t in range(WINDOW_SIZE, self.T, self.rebalance_step):
            period_return = self._simulate_period_return(equal_weights, t)
            portfolio *= (1.0 + period_return)
            values.append(portfolio)

        pv = np.array(values)
        return (pv / pv[0] - 1.0).tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self, t: int) -> np.ndarray:
        """Extract the rolling window ending at step *t*.

        Returns
        -------
        np.ndarray
            Shape ``(window_size, num_assets * num_features)``.
        """
        window = self.scaled_data[t - WINDOW_SIZE:t]   # (window, assets, feats)
        return window.reshape(WINDOW_SIZE, -1).astype(np.float32)

    def _simulate_period_return(
        self,
        weights: np.ndarray,
        t: int,
    ) -> float:
        """Compute the portfolio return for step *t* given *weights*.

        Uses feature index 0 (log-returns) as a proxy for asset returns.

        Parameters
        ----------
        weights:
            Portfolio weights, shape ``(num_assets,)``.
        t:
            Current time index into ``self.price_data``.

        Returns
        -------
        float
            Scalar portfolio return for the period.
        """
        asset_returns = self.price_data[t, :, 0]  # log-returns at time t
        return float(np.dot(weights, asset_returns))

    def _compute_statistics(
        self,
        portfolio_values: list[float],
        period_returns: list[float],
    ) -> dict[str, float]:
        """Compute standard portfolio performance statistics.

        Parameters
        ----------
        portfolio_values:
            Time series of portfolio values.
        period_returns:
            Time series of per-period returns.

        Returns
        -------
        dict
            Keys: ``total_return``, ``annualized_return``, ``annualized_volatility``,
            ``sharpe_ratio``, ``max_drawdown``, ``calmar_ratio``.
        """
        returns = np.array(period_returns, dtype=np.float64)
        pv = np.array(portfolio_values, dtype=np.float64)
        n = len(returns)

        if n == 0:
            return {k: 0.0 for k in (
                "total_return", "annualized_return", "annualized_volatility",
                "sharpe_ratio", "max_drawdown", "calmar_ratio",
                "sortino_ratio", "var_95",
            )}

        total_return = float((pv[-1] / pv[0]) - 1.0)
        annualized_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1)
        annualized_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = float(annualized_return / (annualized_vol + 1e-8))

        # Max drawdown
        peak = np.maximum.accumulate(pv)
        drawdowns = (peak - pv) / (peak + 1e-8)
        max_dd = float(drawdowns.max())

        calmar = float(annualized_return / (max_dd + 1e-8))

        # Sortino ratio — downside deviation only
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            sortino = float(annualized_return / (downside_dev + 1e-8))
        else:
            sortino = 0.0

        # VaR 95% — loss at 5th percentile, expressed as a positive number
        var_95 = float(-np.percentile(returns, 5))

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "var_95": var_95,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax → valid portfolio weights."""
        e = np.exp(x - x.max())
        return e / e.sum()
