"""
api/schemas.py — Pydantic request/response models for the Portfolio Optimizer API.

Every response model includes a ``disclaimer`` field containing the educational
notice.  This ensures any consumer of the API — including frontends and
third-party integrations — always receives the caveat alongside the data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from config import DISCLAIMER, END_DATE, START_DATE, TICKERS, USE_SYNTHETIC_DATA


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigResponse(BaseModel):
    """Public subset of backend configuration for frontend consumption."""

    tickers: list[str] = Field(default_factory=lambda: TICKERS)
    start_date: str = START_DATE
    end_date: str = END_DATE
    use_synthetic: bool = USE_SYNTHETIC_DATA


# ---------------------------------------------------------------------------
# Portfolio Allocation
# ---------------------------------------------------------------------------

class AllocateRequest(BaseModel):
    """Request body for ``POST /portfolio/allocate``."""

    tickers: list[str] = Field(
        default=TICKERS,
        description="Asset ticker symbols to include in the allocation.",
        examples=[["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]],
    )
    model_version: str = Field(
        default="latest",
        description="MLflow model registry version to use.",
    )
    use_synthetic: bool = Field(
        default=True,
        description="Use synthetic GBM data instead of live yfinance data.",
    )


class AllocateResponse(BaseModel):
    """Response body for ``POST /portfolio/allocate``."""

    allocations: dict[str, float] = Field(
        description="Portfolio weights per ticker, summing to 1.0."
    )
    model_version: str
    timestamp: str = Field(default_factory=_now_iso)
    disclaimer: str = Field(default=DISCLAIMER)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    """Request body for ``POST /backtest/run``."""

    tickers: list[str] = Field(
        default=TICKERS,
        description="Asset ticker symbols to backtest.",
    )
    start_date: str = Field(
        default="2020-01-01",
        description="Start date of the backtest period (YYYY-MM-DD).",
        examples=["2020-01-01"],
    )
    end_date: str = Field(
        default="2023-12-31",
        description="End date of the backtest period (YYYY-MM-DD).",
        examples=["2023-12-31"],
    )
    model_version: str = Field(default="latest")
    rebalance_freq: str = Field(
        default="daily",
        description="Rebalancing frequency: 'daily', 'weekly', or 'monthly'.",
    )
    transaction_cost: float = Field(
        default=0.001,
        ge=0.0,
        le=0.05,
        description="Proportional transaction cost applied at each rebalance.",
    )
    initial_capital: float = Field(
        default=10_000.0,
        gt=0.0,
        description="Starting portfolio value in arbitrary currency units.",
    )


class BacktestResponse(BaseModel):
    """Response body for ``POST /backtest/run``."""

    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    calmar_ratio: float
    cumulative_returns: list[float] = Field(
        description="Cumulative return series (starting at 0.0) for the strategy."
    )
    benchmark_cumulative_returns: list[float] = Field(
        description="Equal-weight buy-and-hold cumulative returns for comparison."
    )
    num_rebalances: int = Field(description="Number of rebalancing steps simulated.")
    disclaimer: str = Field(default=DISCLAIMER)
