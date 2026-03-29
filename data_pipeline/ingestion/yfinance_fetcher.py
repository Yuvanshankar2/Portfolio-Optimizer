"""
yfinance_fetcher.py — ETL entry point for market data acquisition.

All data access is controlled by the USE_SYNTHETIC_DATA flag in config.py.
When True (the default), synthetic geometric Brownian motion data is used so
the full pipeline can run offline without any yfinance quota consumption.

This project is strictly educational. Raw financial data fetched via yfinance
is stored locally only and is never committed to version control.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    END_DATE,
    NAN_THRESHOLD,
    NUM_FEATURES,
    START_DATE,
    TICKERS,
    TRAIN_RATIO,
    USE_SYNTHETIC_DATA,
    VAL_RATIO,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_from_yfinance(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    output_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Download OHLCV data for *tickers* via yfinance and cache locally as CSV.

    Parameters
    ----------
    tickers:
        List of ticker symbols, e.g. ``["AAPL", "MSFT"]``.
    start_date:
        ISO-8601 start date string, e.g. ``"2018-01-01"``.
    end_date:
        ISO-8601 end date string, e.g. ``"2023-12-31"``.
    output_dir:
        Directory where per-ticker CSV files are written.  Defaults to the
        gitignored ``data/`` directory.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with columns ``(field, ticker)`` and a DatetimeIndex.

    Raises
    ------
    ValueError
        If *tickers* is empty.
    RuntimeError
        If yfinance fails to return data for all requested tickers.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is not installed. Run: pip install yfinance") from exc

    logger.info("Fetching %d tickers from yfinance (%s → %s)", len(tickers), start_date, end_date)

    raw: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned an empty DataFrame. Check ticker symbols and date range.")

    # Persist per-ticker CSVs for reuse in subsequent ETL runs
    # after all cleaning
    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in raw.columns.get_level_values(1).unique():
            ticker_df = raw.xs(ticker, axis=1, level=1)
            ticker_df.to_csv(output_dir / f"{ticker}.csv")
    else:
        raw.to_csv(output_dir / "single_asset.csv")
    logger.info("yfinance fetch complete. Shape: %s", raw.shape)
    return raw


def generate_synthetic_data(
    num_assets: int = 5,
    window_size: int = WINDOW_SIZE,
    num_periods: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV-like price data using geometric Brownian motion.

    Safe for offline demos — no network dependency.  The generated data has
    realistic statistical properties (log-normal returns, mean-reversion noise)
    but carries no predictive value.

    Parameters
    ----------
    num_assets:
        Number of synthetic assets to simulate.
    window_size:
        Minimum number of periods required for a valid observation window.
    num_periods:
        Total number of daily time steps to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and MultiIndex columns ``(field, asset)``.
        Fields: Open, High, Low, Close, Volume.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end="2023-12-31", periods=num_periods)
    asset_names = [f"ASSET_{i:02d}" for i in range(num_assets)]

    # GBM parameters
    mu = rng.uniform(0.0001, 0.0005, num_assets)       # daily drift
    sigma = rng.uniform(0.01, 0.025, num_assets)        # daily volatility
    s0 = rng.uniform(50.0, 200.0, num_assets)           # initial price

    # Simulate close prices
    dt = 1.0
    returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((num_periods, num_assets)))
    close = s0 * np.cumprod(returns, axis=0)

    # Derive OHLV from close with realistic intra-day noise
    noise = rng.uniform(0.005, 0.015, (num_periods, num_assets))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, (num_periods, num_assets)))
    high = close * (1 + noise)
    low = close * (1 - noise)
    volume = rng.integers(1_000_000, 10_000_000, (num_periods, num_assets)).astype(float)

    arrays: dict[str, np.ndarray] = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }

    columns = pd.MultiIndex.from_product([arrays.keys(), asset_names], names=["field", "ticker"])
    data = np.concatenate([arr for arr in arrays.values()], axis=1)
    df = pd.DataFrame(data, index=dates, columns=columns)

    logger.info("Generated synthetic data: %d periods × %d assets", num_periods, num_assets)
    return df


def load_data(
    tickers: list[str] | None = None,
    use_synthetic: bool | None = None,
) -> pd.DataFrame:
    """Dispatcher: return market data according to the USE_SYNTHETIC_DATA flag.

    Parameters
    ----------
    tickers:
        Ticker list override.  Defaults to ``config.TICKERS``.
    use_synthetic:
        Override for ``config.USE_SYNTHETIC_DATA``.  Pass ``True`` for demos.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV DataFrame (multi-index columns) ready for ``compute_features``.
    """
    tickers = tickers or TICKERS
    synthetic = use_synthetic if use_synthetic is not None else USE_SYNTHETIC_DATA

    if synthetic:
        logger.info("USE_SYNTHETIC_DATA=True — using synthetic GBM data.")
        return generate_synthetic_data(num_assets=len(tickers))
    
    df = fetch_from_yfinance(tickers=tickers)

    # Reindex to a complete Mon–Fri business day calendar.
    # Unlike asfreq("B"), reindex() works regardless of whether the source
    # index has a freq attribute set (yfinance never sets one).
    full_bday_index = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(full_bday_index)

    if isinstance(df.columns, pd.MultiIndex):
        # df[field] uses __getitem__/__setitem__ which correctly target the
        # top-level label in a MultiIndex — unlike df.loc[:, field] which
        # fails silently on assignment.
        for field in ["Open", "High", "Low", "Close"]:
            df[field] = df[field].ffill()
        df["Volume"] = df["Volume"].fillna(0)
    else:
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].ffill()
        df["Volume"] = df["Volume"].fillna(0)

    # Drop leading rows that pre-date the first trading day (cannot ffill).
    df = df.dropna(how="all")
    return df


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute a normalized feature matrix from raw OHLCV data.

    Extracts log-returns and volume z-scores from the raw DataFrame and
    returns a 3-D array suitable for the RL environment and Transformer model.

    Parameters
    ----------
    df:
        Multi-index DataFrame as returned by ``load_data``.

    Returns
    -------
    np.ndarray
        Shape ``(T, num_assets, num_features)`` where *T* is the number of
        valid time steps after dropping the initial NaN rows from rolling ops.
    """
    # Determine assets from level-1 column index
    if isinstance(df.columns, pd.MultiIndex):
        tickers_in_df = df.columns.get_level_values(1).unique().tolist()
        close = df.xs("Close", axis=1, level=0)
        volume = df.xs("Volume", axis=1, level=0)
    else:
        # Single-asset fallback: columns are field names
        tickers_in_df = ["ASSET_00"]
        close = df[["Close"]]
        volume = df[["Volume"]]

    num_assets = len(tickers_in_df)

    # --- Feature 1 & 2: log-returns and 5-day rolling volatility ---
    log_returns = np.log(close / close.shift(1))
    rolling_vol = log_returns.rolling(5).std()

    # --- Feature 3: RSI-14 (stub) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi_normalized = rsi / 100.0  # scale to [0, 1]

    # --- Feature 4: Volume z-score (20-day) ---
    vol_zscore = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-8)

    # --- Feature 5: Price momentum (10-day return) ---
    momentum = close.pct_change(10)

    features_list = [log_returns, rolling_vol, rsi_normalized, vol_zscore, momentum]
    assert len(features_list) == NUM_FEATURES, (
        f"Expected {NUM_FEATURES} features, got {len(features_list)}. "
        "Adjust NUM_FEATURES in config.py."
    )

    # Stack into (T, num_assets, num_features), drop rolling-window warmup rows.
    # Max lookback across all features is 20 (vol_zscore). Dropping 20 rows
    # eliminates all initialization NaNs without masking legitimate data issues.
    stacked = np.stack([f.values for f in features_list], axis=-1)  # (T, assets, feats)
    warmup = 20
    stacked = stacked[warmup:]

    # ── NaN audit ──────────────────────────────────────────────────────────
    # Any NaN surviving past the warmup drop indicates a real data quality
    # problem (e.g. a gap in market data not caught by load_data()).
    nan_count = int(np.isnan(stacked).sum())
    if nan_count > 0:
        nan_frac = nan_count / stacked.size
        logger.warning(
            "NaN cells after warmup: %d / %d (%.2f%%)",
            nan_count, stacked.size, 100 * nan_frac,
        )
        _FEATURE_NAMES = ["log_returns", "rolling_vol", "rsi", "vol_zscore", "momentum"]
        for i, name in enumerate(_FEATURE_NAMES):
            cnt = int(np.isnan(stacked[:, :, i]).sum())
            if cnt:
                logger.warning("  [%s]: %d NaN cells", name, cnt)
        if nan_frac > NAN_THRESHOLD:
            raise ValueError(
                f"NaN fraction {nan_frac:.2%} exceeds threshold {NAN_THRESHOLD:.2%}. "
                "Check input data quality or raise NAN_THRESHOLD in config / .env."
            )
    # Replace any residual boundary NaNs (e.g. first row of a feature) with 0.
    # Do NOT z-score here — the caller must fit the scaler on training data only.
    # Final safety check — fail if any NaNs remain
    stacked = np.nan_to_num(stacked, nan=0.0)
    if np.isnan(stacked).any():
        raise ValueError(
            "NaNs detected in features. "
            "Fix data pipeline or feature computation before training."
        )

    stacked = stacked.astype(np.float32)

    logger.info("Feature array shape: %s", stacked.shape)
    return stacked


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_features(
    features: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered train / val / test split — no shuffling, no leakage.

    Splits are contiguous slices of the time axis.  The scaler must be fitted
    on the returned *train* split only and then applied to all three splits.

    Parameters
    ----------
    features:
        Raw feature array of shape ``(T, num_assets, num_features)`` as
        returned by ``compute_features``.
    train_ratio:
        Fraction of time steps for training (default 0.70).
    val_ratio:
        Fraction of time steps for validation (default 0.15).
        Test ratio = 1 - train_ratio - val_ratio.

    Returns
    -------
    (train, val, test)
        Three arrays with shapes ``(T_train, A, F)``, ``(T_val, A, F)``,
        ``(T_test, A, F)``.

    Raises
    ------
    ValueError
        If *val* or *test* split is too small to form a single RL episode.
    """
    T = len(features)
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    train = features[:train_end]
    val   = features[train_end:val_end]
    test  = features[val_end:]

    for name, split in [("val", val), ("test", test)]:
        if len(split) <= WINDOW_SIZE:
            raise ValueError(
                f"{name} split has only {len(split)} rows — too small for "
                f"window_size={WINDOW_SIZE}. Increase the date range or reduce WINDOW_SIZE."
            )

    logger.info(
        "Split: train=%d | val=%d | test=%d steps (total %d)",
        len(train), len(val), len(test), T,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Feature scaling (fit on train only, apply to all splits)
# ---------------------------------------------------------------------------

def fit_scaler(train_features: np.ndarray) -> dict[str, np.ndarray]:
    """Fit a per-feature z-score scaler on *training data only*.

    Statistics are computed across the ``(T, assets)`` axes so that each of
    the five feature types is normalised independently of scale differences
    between assets (e.g. high-volume SPY vs. low-volume GLD are placed on the
    same scale for log-returns).

    Parameters
    ----------
    train_features:
        Training split, shape ``(T_train, num_assets, num_features)``.

    Returns
    -------
    dict
        ``{"mean": np.ndarray(F,), "std": np.ndarray(F,)}`` — broadcast-ready
        statistics for ``apply_scaler``.
    """
    _, _, F = train_features.shape
    flat = train_features.reshape(-1, F)   # (T*A, F)
    mean = flat.mean(axis=0)               # (F,)
    std  = flat.std(axis=0) + 1e-8         # (F,) — epsilon prevents /0

    logger.info(
        "Scaler fitted on %d training rows. Per-feature std: %s",
        len(flat), std.round(6),
    )
    return {"mean": mean, "std": std}


def apply_scaler(
    features: np.ndarray,
    scaler: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply a fitted z-score scaler and clip outliers to ±5 std.

    The clip is applied *after* z-scoring so the ±5 boundary is in standard
    deviation units — equivalent to removing values more than 5σ from the
    training mean.

    Parameters
    ----------
    features:
        Any split (train / val / test), shape ``(T, num_assets, num_features)``.
    scaler:
        Dict returned by ``fit_scaler``.

    Returns
    -------
    np.ndarray
        Scaled float32 array with the same shape as *features*.
    """
    scaled = (features - scaler["mean"]) / scaler["std"]
    return np.clip(scaled, -5.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Portfolio Optimizer — ETL pipeline")
    parser.add_argument("--synthetic", action="store_true", default=USE_SYNTHETIC_DATA,
                        help="Force synthetic data mode (default: USE_SYNTHETIC_DATA env var)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="Ticker symbols to fetch (ignored in synthetic mode)")
    args = parser.parse_args()

    df = load_data(tickers=args.tickers, use_synthetic=args.synthetic)
    print(f"\nRaw DataFrame shape : {df.shape}")

    features = compute_features(df)
    print(f"Feature array shape : {features.shape}  (T × assets × features)")
    print(f"Value range         : [{features.min():.4f}, {features.max():.4f}]")
    print(f"\nSynthetic mode      : {args.synthetic}")
    print("ETL pipeline complete.")
