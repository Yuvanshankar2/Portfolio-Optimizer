"""
predict_allocation.py — Inference module for the Portfolio Optimizer.

Loads a trained PPO agent from the MLflow Model Registry, fetches the most
recent market data window, applies the training-time scaler, and returns
normalized portfolio weights.

For educational and demonstration purposes only.  Output must not be used
for actual investment or trading decisions.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from config import MODEL_NAME, MODELS_DIR, TICKERS, WINDOW_SIZE

if TYPE_CHECKING:
    from stable_baselines3 import PPO

    from ml.environments.portfolio_env import PortfolioEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent + scaler loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_agent(
    model_name: str = MODEL_NAME,
    version: str | int = "latest",
) -> "PPO":
    """Load and cache a trained PPO agent from the MLflow Model Registry.

    Results are cached by ``(model_name, version)`` so repeated API calls
    do not re-download the artifact on every request.

    Parameters
    ----------
    model_name:
        Registered model name in the MLflow registry.
    version:
        Registry version or ``"latest"``.

    Returns
    -------
    stable_baselines3.PPO
        The loaded agent in evaluation mode.
    """
    from mlflow_client import PortfolioMLflowClient

    logger.info("Loading agent from MLflow registry: %s@%s", model_name, version)
    client = PortfolioMLflowClient()
    agent = client.load_registered_model(model_name=model_name, version=version)
    logger.info("Agent loaded successfully.")
    return agent


def load_scaler() -> dict:
    """Load the feature scaler saved alongside the latest trained model.

    The scaler is written to ``models/latest.scaler.pkl`` at the end of each
    training run so inference always has a stable, version-agnostic path.

    Raises
    ------
    FileNotFoundError
        If no model has been trained yet.
    """
    import joblib

    path = MODELS_DIR / "latest.scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. Train a model first:\n"
            "  python ml/training/train_rl_agent.py"
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------

def get_latest_observation(
    tickers: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Fetch the most recent *window_size* rows of scaled feature data.

    In synthetic mode (``USE_SYNTHETIC_DATA=True``) this generates fresh
    synthetic data so no network call is made.

    Parameters
    ----------
    tickers:
        Ticker override.  Defaults to ``config.TICKERS``.
    window_size:
        Number of time steps in the observation window.

    Returns
    -------
    np.ndarray
        Shape ``(1, window_size, num_assets * num_features)`` — batch dim
        prepended so it can be passed directly to ``agent.predict()``.
    """
    from data_pipeline.ingestion.yfinance_fetcher import apply_scaler, compute_features, load_data

    tickers = tickers or TICKERS
    df = load_data(tickers=tickers)
    features = compute_features(df)  # (T, assets, feats) — raw, unscaled

    # Apply the training-time scaler so the observation matches the distribution
    # the agent was trained on.  Without this, the agent receives out-of-
    # distribution inputs and outputs near-uniform logits.
    try:
        scaler = load_scaler()
        features = apply_scaler(features, scaler)
        logger.debug(
            "Scaler applied. Obs stats: mean=%.4f std=%.4f range=[%.4f, %.4f]",
            features.mean(), features.std(), features.min(), features.max(),
        )
    except FileNotFoundError as exc:
        logger.warning("Scaler not found — using unscaled features. (%s)", exc)

    if len(features) < window_size:
        raise ValueError(
            f"Not enough data to form a window. Have {len(features)} rows, need {window_size}."
        )

    window = features[-window_size:]                     # (window, assets, feats)
    obs = window.reshape(window_size, -1).astype(np.float32)  # (window, assets*feats)
    return obs[np.newaxis, ...]                          # (1, window, input_dim)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_weights(
    observation: np.ndarray | None = None,
    model_name: str = MODEL_NAME,
    version: str | int = "latest",
    tickers: list[str] | None = None,
) -> dict[str, float]:
    """Predict portfolio weights for the current market state.

    Parameters
    ----------
    observation:
        Pre-computed observation array of shape
        ``(1, window_size, num_assets * num_features)``.
        If ``None``, ``get_latest_observation()`` is called automatically.
        The observation must already be scaled with the training-time scaler.
    model_name:
        Registry model name.
    version:
        Registry version or ``"latest"``.
    tickers:
        Ticker symbols to label the output weights.

    Returns
    -------
    dict
        ``{ticker: weight}`` mapping where weights sum to 1.0.
    """
    tickers = tickers or TICKERS

    if observation is None:
        observation = get_latest_observation(tickers=tickers)

    agent = load_agent(model_name=model_name, version=str(version))

    # VecNormalize is NOT applied here: training used norm_obs=False so
    # observations were never normalized by VecNormalize — only by the
    # feature scaler (applied above in get_latest_observation).
    action, _ = agent.predict(observation, deterministic=True)

    raw = action.flatten()
    logger.info(
        "Raw action: min=%.4f max=%.4f mean=%.4f std=%.6f",
        raw.min(), raw.max(), raw.mean(), raw.std(),
    )

    weights = _softmax(raw)

    if len(weights) != len(tickers):
        raise ValueError(
            f"Weight vector length ({len(weights)}) does not match "
            f"number of tickers ({len(tickers)})."
        )

    allocation = {ticker: float(w) for ticker, w in zip(tickers, weights)}
    logger.info(
        "Final weights: %s",
        {t: f"{w:.4f}" for t, w in allocation.items()},
    )
    return allocation


def predict_weights_from_env(
    env: "PortfolioEnv",
    agent: "PPO",
) -> tuple[np.ndarray, dict]:
    """Run a single prediction step using an existing environment.

    Used by the backtesting engine to avoid re-loading the agent on
    every rebalance step.  Applies the scaler to the observation and
    softmax to the raw action before returning weights.

    Parameters
    ----------
    env:
        ``PortfolioEnv`` at the current time step.
    agent:
        Pre-loaded PPO agent.

    Returns
    -------
    (weights, info)
        ``weights``: np.ndarray of shape ``(num_assets,)``, sums to 1.0.
        ``info``: dict with raw action metadata.
    """
    from data_pipeline.ingestion.yfinance_fetcher import apply_scaler

    obs = env._get_observation()[np.newaxis, ...]  # (1, window, input_dim)

    try:
        scaler = load_scaler()
        # Reshape to (T*A, F) for apply_scaler, then restore shape
        orig_shape = obs.shape
        obs = apply_scaler(obs.reshape(-1, orig_shape[-1]), scaler).reshape(orig_shape)
    except FileNotFoundError:
        logger.warning("Scaler not found in predict_weights_from_env — using raw obs.")

    action, _ = agent.predict(obs, deterministic=True)
    weights = _softmax(action.flatten())
    return weights, {"raw_action": action.tolist()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def make_env_placeholder(obs_dim: int, num_assets: int):
    """Minimal stub environment matching PortfolioEnv's spaces.

    Used when a Gymnasium environment is required by a framework wrapper
    but actual stepping is not needed (e.g. for VecNormalize loading).

    Parameters
    ----------
    obs_dim:
        Flattened observation dimension (num_assets * num_features).
    num_assets:
        Number of assets; determines action space shape.
    """
    import gymnasium as gym
    from gymnasium import spaces

    class PlaceholderEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(WINDOW_SIZE, obs_dim), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(num_assets,), dtype=np.float32
            )

        def reset(self, seed=None, options=None):
            return np.zeros((WINDOW_SIZE, obs_dim), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((WINDOW_SIZE, obs_dim), dtype=np.float32), 0.0, True, False, {}

    return PlaceholderEnv()
