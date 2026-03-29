"""
portfolio_env.py — Gymnasium RL environment for portfolio weight optimization.

The agent observes a rolling window of normalized financial features and
outputs a continuous allocation vector over *n* assets.  Weights are
constrained to be non-negative and sum to 1 (long-only, fully invested)
by applying a softmax transform inside ``step()``.

Observation space : Box(shape=(window_size, num_assets * num_features))
Action space      : Box(shape=(num_assets,), low=0, high=1)  — raw logits
Reward            : risk-adjusted return (configurable via ``reward_mode``)

For educational purposes only.  Not a production trading system.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import NUM_FEATURES, REWARD_MODE, WINDOW_SIZE

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """Gymnasium environment simulating sequential portfolio rebalancing.

    Parameters
    ----------
    price_data:
        Feature array of shape ``(T, num_assets, num_features)`` as produced
        by ``data_pipeline.ingestion.yfinance_fetcher.compute_features``.
    window_size:
        Number of past time steps visible to the agent per observation.
    reward_mode:
        How to compute the step reward:
        - ``"sharpe"``: Sharpe-like ratio (mean / std of rolling returns).
        - ``"return"``: Raw period portfolio return.
        - ``"penalized"``: Return minus a drawdown penalty term.
    transaction_cost:
        Proportional cost applied to the absolute weight change at each step.
    initial_capital:
        Starting portfolio value (normalised; 1.0 is conventional).
    render_mode:
        ``"human"`` prints step info to stdout; ``None`` disables rendering.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: np.ndarray,
        window_size: int = WINDOW_SIZE,
        reward_mode: str = REWARD_MODE,
        transaction_cost: float = 0.001,
        initial_capital: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if price_data.ndim != 3:
            raise ValueError(
                f"price_data must be 3-D (T, num_assets, num_features), "
                f"got shape {price_data.shape}."
            )

        self.price_data = price_data.astype(np.float32)
        self.T, self.num_assets, self.num_feats = price_data.shape
        self.window_size = window_size
        self.reward_mode = reward_mode
        self.transaction_cost = transaction_cost
        self.initial_capital = float(initial_capital)
        self.render_mode = render_mode

        if self.T <= window_size:
            raise ValueError(
                f"price_data length ({self.T}) must exceed window_size ({window_size})."
            )

        # --- Spaces ---
        obs_dim = self.num_assets * self.num_feats
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, obs_dim),
            dtype=np.float32,
        )
        # Raw logits; softmax applied in step() → valid probability simplex
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

        # Internal state (reset in reset())
        self._step: int = 0
        self._portfolio_value: float = self.initial_capital
        self._weights: np.ndarray = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        self._portfolio_history: list[float] = []
        self._return_history: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the beginning of the price series.

        Returns
        -------
        observation:
            Initial window of features.
        info:
            Empty dict (Gymnasium convention).
        """
        super().reset(seed=seed)

        self._step = self.window_size  # first valid index after full window
        self._portfolio_value = self.initial_capital
        self._weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        self._portfolio_history = [self.initial_capital]
        self._return_history = []

        return self._get_observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one time step.

        The raw action (logits) is converted to valid portfolio weights via
        softmax.  Transaction costs proportional to the L1 weight change are
        deducted.  The portfolio value is updated and the reward is computed.

        Parameters
        ----------
        action:
            Raw allocation logits, shape ``(num_assets,)``.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        # --- Convert action to valid weights ---
        weights = self._softmax(action)

        # --- Compute transaction cost ---
        weight_change = np.abs(weights - self._weights).sum()
        tc = self.transaction_cost * weight_change

        # --- Simulate one-period return ---
        # Use log-returns (feature index 0) as a proxy for asset returns
        period_returns = self.price_data[self._step, :, 0]  # shape (num_assets,)
        portfolio_return = float(np.dot(weights, period_returns)) - tc

        # --- Update portfolio ---
        self._portfolio_value *= (1.0 + portfolio_return)
        self._weights = weights
        self._portfolio_history.append(self._portfolio_value)
        self._return_history.append(portfolio_return)

        # --- Reward ---
        reward = self._compute_reward(weights, np.array(self._return_history))

        # --- Advance step ---
        self._step += 1
        terminated = self._step >= self.T
        truncated = False

        info = {
            "weights": weights.tolist(),
            "portfolio_value": self._portfolio_value,
            "period_return": portfolio_return,
            "step": self._step,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print current step summary (human render mode)."""
        print(
            f"Step {self._step:4d} | "
            f"Portfolio: {self._portfolio_value:.4f} | "
            f"Weights: {[f'{w:.3f}' for w in self._weights]}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Extract the current rolling window and flatten the asset dimension.

        Returns
        -------
        np.ndarray
            Shape ``(window_size, num_assets * num_features)``.
        """
        start = self._step - self.window_size
        end = self._step
        window = self.price_data[start:end]              # (window, assets, feats)
        return window.reshape(self.window_size, -1).astype(np.float32)

    def _compute_reward(
        self,
        weights: np.ndarray,
        return_history: np.ndarray,
    ) -> float:
        """Dispatch reward calculation based on ``self.reward_mode``.

        Parameters
        ----------
        weights:
            Current portfolio weights (post-softmax).
        return_history:
            Array of all portfolio returns so far in the episode.

        Returns
        -------
        float
            Scalar reward signal.
        """
        if len(return_history) == 0:
            return 0.0

        if self.reward_mode == "sharpe":
            return self._sharpe_reward(return_history)
        elif self.reward_mode == "return":
            return float(return_history[-1])
        elif self.reward_mode == "penalized":
            return self._penalized_reward(return_history)
        else:
            raise ValueError(
                f"Unknown reward_mode '{self.reward_mode}'. "
                "Choose from: 'sharpe', 'return', 'penalized'."
            )

    @staticmethod
    def _sharpe_reward(return_history: np.ndarray, eps: float = 1e-8) -> float:
        """Sharpe-like reward: mean(returns) / (std(returns) + eps)."""
        if len(return_history) < 2:
            return float(return_history[-1])
        return float(return_history.mean() / (return_history.std() + eps))

    def _penalized_reward(self, return_history: np.ndarray) -> float:
        """Last-period return penalised by max drawdown of the episode."""
        peak = np.maximum.accumulate(self._portfolio_history)
        drawdown = (peak - np.array(self._portfolio_history)) / (peak + 1e-8)
        max_dd = float(drawdown.max())
        return float(return_history[-1]) - 0.5 * max_dd

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax for converting logits to weights."""
        e = np.exp(x - x.max())
        return e / e.sum()
