"""
config.py — Centralized configuration for Portfolio Optimizer.

All modules import constants from here rather than reading env vars directly.
Copy .env.example to .env and adjust values before running.
"""

from __future__ import annotations

import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"
MODELS_DIR: Path = ROOT_DIR / "models"

# Ensure local directories exist at import time (data/ is gitignored)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data flags
# ---------------------------------------------------------------------------
USE_SYNTHETIC_DATA: bool = os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true"

_raw_tickers = os.getenv("TICKERS", "SPY,QQQ,GLD,TLT,BTC-USD")
TICKERS: list[str] = [t.strip() for t in _raw_tickers.split(",") if t.strip()]

START_DATE: str = os.getenv("START_DATE", "2015-01-01")
END_DATE: str = os.getenv("END_DATE", "2024-12-31")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
NUM_FEATURES: int = int(os.getenv("NUM_FEATURES", "5"))  # open, high, low, close, volume
WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "30"))   # rolling look-back in trading days

# ---------------------------------------------------------------------------
# Transformer hyperparameters
# ---------------------------------------------------------------------------
D_MODEL: int = int(os.getenv("D_MODEL", "64"))
NHEAD: int = int(os.getenv("NHEAD", "4"))
NUM_ENCODER_LAYERS: int = int(os.getenv("NUM_ENCODER_LAYERS", "2"))
DIM_FEEDFORWARD: int = int(os.getenv("DIM_FEEDFORWARD", "128"))
DROPOUT: float = float(os.getenv("DROPOUT", "0.1"))

# ---------------------------------------------------------------------------
# RL training hyperparameters
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS: int = int(os.getenv("TOTAL_TIMESTEPS", "100000"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "3e-4"))
N_STEPS: int = int(os.getenv("N_STEPS", "2048"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
N_EPOCHS: int = int(os.getenv("N_EPOCHS", "10"))
REWARD_MODE: str = os.getenv("REWARD_MODE", "sharpe")  # "sharpe" | "return" | "penalized"

# ---------------------------------------------------------------------------
# Data pipeline — splits and training guards
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = float(os.getenv("TRAIN_RATIO", "0.70"))
VAL_RATIO: float   = float(os.getenv("VAL_RATIO",   "0.15"))
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.15 (implied)

# Max fraction of NaN cells tolerated in the feature array after warmup.
# Exceeding this raises ValueError — prevents silent data corruption reaching the model.
NAN_THRESHOLD: float = float(os.getenv("NAN_THRESHOLD", "0.01"))

# PPO gradient clipping — matches SB3 default but is now explicit and configurable.
MAX_GRAD_NORM: float = float(os.getenv("MAX_GRAD_NORM", "0.5"))

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "portfolio_optimizer")
MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "ppo_portfolio_agent")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_VERSION: str = "0.1.0"

DISCLAIMER: str = (
    "This project is strictly educational and for demonstration purposes only. "
    "It does not provide financial advice and must not be used for live trading "
    "or investment decisions. Market data is accessed via yfinance. "
    "Users are responsible for complying with Yahoo Finance terms of service."
)


if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Optimizer — Configuration")
    print("=" * 60)
    print(f"ROOT_DIR            : {ROOT_DIR}")
    print(f"DATA_DIR            : {DATA_DIR}")
    print(f"USE_SYNTHETIC_DATA  : {USE_SYNTHETIC_DATA}")
    print(f"TICKERS             : {TICKERS}")
    print(f"START_DATE          : {START_DATE}")
    print(f"END_DATE            : {END_DATE}")
    print(f"WINDOW_SIZE         : {WINDOW_SIZE}")
    print(f"NUM_FEATURES        : {NUM_FEATURES}")
    print(f"D_MODEL             : {D_MODEL}")
    print(f"TOTAL_TIMESTEPS     : {TOTAL_TIMESTEPS}")
    print(f"MLFLOW_TRACKING_URI : {MLFLOW_TRACKING_URI}")
    print(f"EXPERIMENT_NAME     : {EXPERIMENT_NAME}")
    print("=" * 60)
