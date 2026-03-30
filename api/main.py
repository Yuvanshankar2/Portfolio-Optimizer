"""
api/main.py — FastAPI application for the Portfolio Optimizer.

Endpoints:
    GET  /health              — Liveness check
    POST /portfolio/allocate  — Predict portfolio weights
    POST /backtest/run        — Simulate historical portfolio performance

The MLflow client and (optionally) a cached agent are initialised once during
application startup via the async lifespan handler.

For educational and demonstration purposes only.  Not a production trading system.
"""

from __future__ import annotations

import logging
import numpy as np
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import API_VERSION, DISCLAIMER, END_DATE, MODEL_NAME, START_DATE, TICKERS, USE_SYNTHETIC_DATA
from api.schemas import (
    AllocateRequest,
    AllocateResponse,
    BacktestRequest,
    BacktestResponse,
    ConfigResponse,
    CorrelationResponse,
    HealthResponse,
    RiskProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk-profile weight adjustment
# ---------------------------------------------------------------------------

def _adjust_for_risk(
    weights: dict[str, float],
    risk_profile: RiskProfile,
    tickers: list[str],
    use_synthetic: bool,
) -> dict[str, float]:
    """Post-process agent weights to match the requested risk tolerance."""
    from datetime import datetime, timedelta

    if risk_profile == RiskProfile.moderate:
        return weights

    w = np.array([weights[t] for t in tickers], dtype=float)

    if risk_profile == RiskProfile.aggressive:
        # Power-scale (exponent 2) concentrates weight on top picks
        w_adj = w ** 2
        w_adj /= w_adj.sum()
        return {t: round(float(w_adj[i]), 6) for i, t in enumerate(tickers)}

    # conservative: blend 50% agent + 50% inverse-volatility weights
    inv_vol_w = np.ones(len(tickers)) / len(tickers)  # fallback = equal weight
    if not use_synthetic:
        try:
            import yfinance as yf
            end = datetime.utcnow()
            start = end - timedelta(days=90)
            prices = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )["Close"]
            prices = prices[tickers].dropna()
            vols = prices.pct_change().std().values + 1e-8
            inv_vol = 1.0 / vols
            inv_vol_w = inv_vol / inv_vol.sum()
        except Exception as exc:
            logger.warning("Conservative vol fetch failed — using equal weights: %s", exc)

    w_adj = 0.5 * w + 0.5 * inv_vol_w
    w_adj /= w_adj.sum()
    return {t: round(float(w_adj[i]), 6) for i, t in enumerate(tickers)}

# ---------------------------------------------------------------------------
# Application state (populated during lifespan startup)
# ---------------------------------------------------------------------------

_app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; clean up on shutdown."""
    logger.info("Starting Portfolio Optimizer API v%s", API_VERSION)

    # Initialise MLflow client (does not require a running server at import time)
    try:
        from mlflow_client import PortfolioMLflowClient
        _app_state["mlflow_client"] = PortfolioMLflowClient()
        logger.info("MLflow client initialised.")
    except Exception as exc:
        logger.warning("MLflow client could not be initialised: %s", exc)
        _app_state["mlflow_client"] = None

    # Optionally pre-warm the agent cache (skip if no registered model yet)
    _app_state["agent"] = None
    try:
        from ml.inference.predict_allocation import load_agent
        _app_state["agent"] = load_agent(model_name=MODEL_NAME, version="latest")
        logger.info("Agent pre-loaded from MLflow registry.")
    except Exception as exc:
        logger.info(
            "No pre-trained agent found in registry — allocate endpoint will "
            "use equal weights until a model is trained. (%s)", exc
        )

    yield  # ← application runs here

    _app_state.clear()
    logger.info("Portfolio Optimizer API shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Portfolio Optimizer API",
    description=(
        "AI-driven portfolio weight prediction using a Transformer + PPO RL agent.\n\n"
        f"**Disclaimer:** {DISCLAIMER}"
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/config",
    response_model=ConfigResponse,
    summary="Return active backend configuration",
    tags=["Infrastructure"],
)
async def get_config() -> ConfigResponse:
    """Expose the subset of backend config the frontend needs to build requests."""
    return ConfigResponse(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        use_synthetic=USE_SYNTHETIC_DATA,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Infrastructure"],
)
async def health() -> HealthResponse:
    """Return service liveness status and API version."""
    return HealthResponse(version=API_VERSION)


@app.post(
    "/portfolio/allocate",
    response_model=AllocateResponse,
    summary="Predict portfolio allocation weights",
    tags=["Portfolio"],
)
async def allocate(request: AllocateRequest) -> AllocateResponse:
    """Predict portfolio weights for the requested tickers.

    If a trained model is available in the MLflow registry it is used.
    Otherwise, an equal-weight fallback is returned with a warning.

    The response always includes an educational disclaimer.
    """
    import os

    os.environ["USE_SYNTHETIC_DATA"] = "true" if request.use_synthetic else "false"

    agent = _app_state.get("agent")

    try:
        if agent is not None:
            from ml.inference.predict_allocation import predict_weights
            allocations = predict_weights(
                model_name=MODEL_NAME,
                version=request.model_version,
                tickers=request.tickers,
            )
        else:
            # Equal-weight fallback when no model is registered
            logger.warning(
                "No trained agent available — returning equal-weight allocation."
            )
            equal = 1.0 / len(request.tickers)
            allocations = {ticker: equal for ticker in request.tickers}

    except Exception as exc:
        logger.error("Allocation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Allocation failed: {exc}",
        ) from exc

    allocations = _adjust_for_risk(
        allocations, request.risk_profile, request.tickers, request.use_synthetic
    )

    dollar_allocations = None
    if request.investment_amount is not None:
        dollar_allocations = {
            ticker: round(weight * request.investment_amount, 2)
            for ticker, weight in allocations.items()
        }

    return AllocateResponse(
        allocations=allocations,
        dollar_allocations=dollar_allocations,
        model_version=request.model_version,
        risk_profile=request.risk_profile,
    )


@app.post(
    "/backtest/run",
    response_model=BacktestResponse,
    summary="Run historical portfolio backtest",
    tags=["Backtesting"],
)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Simulate portfolio rebalancing over a historical period.

    Fetches or generates data for the requested tickers and date range, then
    runs the ``BacktestEngine``.  If a trained agent is available it is used
    for allocation decisions; otherwise an equal-weight strategy is simulated.
    """
    from data_pipeline.ingestion.yfinance_fetcher import compute_features, load_data
    from backtesting.engine import BacktestEngine

    # Validate dates
    try:
        from datetime import date
        start = date.fromisoformat(request.start_date)
        end = date.fromisoformat(request.end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date.")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        df = load_data(tickers=request.tickers, use_synthetic=False)
        raw_features = compute_features(df)

        # Scale only for agent observations, keep raw for return simulation
        try:
            import joblib
            from config import MODELS_DIR
            from data_pipeline.ingestion.yfinance_fetcher import apply_scaler
            scaler = joblib.load(MODELS_DIR / "latest.scaler.pkl")
            scaled_features = apply_scaler(raw_features, scaler)
            logger.info("Scaler applied to backtest features.")
        except FileNotFoundError:
            scaled_features = raw_features
            logger.warning("Scaler not found — backtest uses unscaled features.")

        engine = BacktestEngine(
            price_data=raw_features,       # unscaled for return simulation
            scaled_data=scaled_features,   # scaled for agent observations
            tickers=request.tickers,
            rebalance_freq=request.rebalance_freq,
            transaction_cost=request.transaction_cost,
            initial_capital=request.initial_capital,
        )
        agent = _app_state.get("agent")
        num_assets = len(request.tickers)
        if agent is not None:
            result = engine.run(agent=agent)
        else:
            result = engine.run(predict_fn=lambda obs: np.ones(num_assets))

        result.spy_benchmark_returns = engine.compute_spy_benchmark(
            request.start_date, request.end_date
        )

    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Backtest failed: {exc}",
        ) from exc

    return BacktestResponse(
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        total_return=result.total_return,
        annualized_return=result.annualized_return,
        annualized_volatility=result.annualized_volatility,
        calmar_ratio=result.calmar_ratio,
        sortino_ratio=result.sortino_ratio,
        var_95=result.var_95,
        cumulative_returns=result.cumulative_returns,
        benchmark_cumulative_returns=result.benchmark_cumulative_returns,
        spy_benchmark_returns=result.spy_benchmark_returns,
        num_rebalances=len(result.period_returns),
    )


@app.get(
    "/portfolio/correlation",
    response_model=CorrelationResponse,
    summary="Pairwise log-return correlation matrix",
    tags=["Portfolio"],
)
async def get_correlation(use_synthetic: bool = False) -> CorrelationResponse:
    """Compute pairwise Pearson correlations of daily log-returns over the training period."""
    import pandas as pd
    from data_pipeline.ingestion.yfinance_fetcher import load_data

    try:
        df = load_data(tickers=TICKERS, use_synthetic=use_synthetic)
    except Exception as exc:
        logger.error("Correlation data load failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Data load failed: {exc}",
        ) from exc

    try:
        if isinstance(df.columns, pd.MultiIndex):
            close = df.xs("Close", axis=1, level=0)[TICKERS]
        else:
            close = df[["Close"]]

        log_returns = np.log(close / close.shift(1)).dropna()
        corr_df = log_returns.corr(method="pearson")

        matrix = [
            [round(float(corr_df.loc[r, c]), 4) for c in TICKERS]
            for r in TICKERS
        ]
    except Exception as exc:
        logger.error("Correlation computation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Correlation computation failed: {exc}",
        ) from exc

    return CorrelationResponse(
        tickers=TICKERS,
        matrix=matrix,
        start_date=START_DATE,
        end_date=END_DATE,
    )
