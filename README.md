# Portfolio Optimizer

![Python](https://img.shields.io/badge/Python-3.11-3572A5?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-4CAF50?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=flat-square&logo=next.js&logoColor=white)

> [!WARNING]
> **Educational Use Only.** This project is built exclusively for demonstration purposes.
> It **must not** be used for actual trading, investment decisions, or financial advice.
> Past simulated performance does not predict future returns.

---

## Overview

Portfolio Optimizer is an end-to-end AI-driven portfolio management platform that combines deep learning and reinforcement learning to allocate capital across a configurable set of assets. The system ingests real market data via yfinance, engineers financial features, trains a Proximal Policy Optimization (PPO) agent backed by a custom Transformer architecture, and exposes predictions and backtesting results through a FastAPI backend and a Next.js dashboard.

The project demonstrates a complete production-grade machine learning pipeline — from raw data ingestion to a live inference API — covering quantitative finance, deep reinforcement learning, MLOps best practices, and full-stack deployment.

**Core capabilities:**

- Predicts portfolio allocation weights across multiple assets using a trained RL agent
- Backtests historical performance with transaction costs and rebalancing frequency controls
- Tracks experiments, hyperparameters, and model versions with MLflow
- Serves predictions through a REST API with a real-time frontend dashboard
- Computes standard portfolio metrics: Sharpe ratio, max drawdown, annualized return, Calmar ratio, and volatility

---

## Architecture

```
yfinance API / Synthetic GBM
          │
          ▼
 ┌─────────────────────┐
 │   ETL Pipeline      │  fetch_from_yfinance()
 │  data_pipeline/     │  generate_synthetic_data()
 │  ingestion/         │  compute_features()
 └────────┬────────────┘
          │  (T, assets, features) — raw float32 array
          ▼
 ┌─────────────────────┐
 │  Feature Engineering│  log-returns, rolling volatility,
 │  + Scaling          │  RSI-14, volume z-score, momentum
 └────────┬────────────┘  z-score normalization (train stats only)
          │
          ▼
 ┌─────────────────────┐
 │  RL Environment     │  Gymnasium — observation: (window, assets×features)
 │  portfolio_env.py   │  action: softmax weights (long-only, Σwᵢ = 1)
 └────────┬────────────┘  reward: Sharpe | return | penalized
          │
          ▼
 ┌─────────────────────┐
 │  Transformer + PPO  │  TransformerFeatureExtractor (PyTorch)
 │  ml/models/         │  TransformerExtractor (SB3 BaseFeaturesExtractor)
 │  ml/training/       │  PPO agent (Stable-Baselines3)
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  MLflow Registry    │  Hyperparameters + metrics logging
 │  mlflow_client.py   │  Model versioning + artifact storage
 └────────┬────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 Inference    Backtest
 predict_     engine.py
 allocation   BacktestEngine
    │            │
    └─────┬──────┘
          ▼
 ┌─────────────────────┐
 │  FastAPI            │  GET  /health
 │  api/main.py        │  GET  /config
 └────────┬────────────┘  POST /portfolio/allocate
          │               POST /backtest/run
          ▼
 ┌─────────────────────┐
 │  Next.js Dashboard  │  Allocation weights + dollar amounts
 │  frontend/          │  Backtest statistics + cumulative return chart
 └─────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/portfolio-optimizer.git
cd portfolio-optimizer
```

### 2. Set up the Python environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set the following at minimum:

```
MLFLOW_TRACKING_URI=http://localhost:5000
USE_SYNTHETIC_DATA=false
```

See the [Environment Variables](#environment-variables) section for the full reference.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Start the MLflow server

Open a dedicated terminal and keep it running:

```bash
.venv\Scripts\Activate.ps1   # Windows
# or: source .venv/bin/activate

mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### 6. Train the RL agent

Run a quick smoke test first to verify the full pipeline end to end:

```bash
python -m ml.training.train_rl_agent --timesteps 10000 --run-name smoke_test
```

For a full training run on real market data:

```bash
python -m ml.training.train_rl_agent --timesteps 100000 --run-name real_data_run
```

Training will automatically register the model in the MLflow registry on completion.

### 7. Start the API

In a second terminal:

```bash
.venv\Scripts\Activate.ps1   # Windows
# or: source .venv/bin/activate

uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 8. Start the frontend

In a third terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` for the dashboard.

---

## Technical Deep Dive

### ETL Pipeline

The data pipeline (`data_pipeline/ingestion/yfinance_fetcher.py`) handles two modes controlled by the `USE_SYNTHETIC_DATA` flag:

**Real data mode** fetches OHLCV data for the configured tickers via yfinance, reindexes to a complete business day calendar, forward-fills price gaps caused by non-trading days, and fills missing volume with zero. Data is cached locally as CSV and never committed to version control.

**Synthetic mode** generates geometric Brownian motion (GBM) price paths with configurable drift and volatility parameters, enabling the full pipeline to run offline without any network dependency or yfinance quota consumption. This is useful for rapid iteration and testing.

---

### Feature Engineering

Five features are computed per asset per time step, forming a `(T, num_assets, num_features)` array:

| Index | Feature | Description |
|---|---|---|
| 0 | Log-returns | `log(Pₜ / Pₜ₋₁)` — used directly as portfolio return proxy in the backtester |
| 1 | Rolling volatility | 5-day standard deviation of log-returns |
| 2 | RSI-14 | Relative Strength Index normalized to [0, 1] |
| 3 | Volume z-score | 20-day rolling volume z-score |
| 4 | Momentum | 10-day price percent change |

A **temporal train/val/test split** (70/15/15) is applied before scaling to prevent look-ahead leakage. A z-score scaler is fitted on the training split only and applied to all three splits. Outliers are clipped to ±5 standard deviations. The fitted scaler is saved alongside the model so inference applies identical normalization at prediction time.

---

### Transformer + PPO Architecture

**Feature extractor (`TransformerFeatureExtractor`):**

The raw observation `(window_size, num_assets × num_features)` is projected into a `d_model`-dimensional embedding space via a linear layer. Sinusoidal positional encodings are added to preserve temporal ordering. The sequence is then passed through a multi-head self-attention Transformer encoder, which learns cross-asset and cross-time dependencies. The encoder output is mean-pooled across the time dimension to produce a fixed-size latent vector that summarizes the entire observation window.

**PPO agent:**

The latent vector feeds into Stable-Baselines3's PPO algorithm with a standard MLP policy head. The agent outputs a continuous action vector that is passed through a numerically stable softmax to produce normalized, long-only portfolio weights summing to 1. Three reward modes are available:

- `sharpe` — rewards risk-adjusted return at each step
- `return` — rewards raw log-portfolio return
- `penalized` — rewards return with explicit drawdown and volatility penalties

**Training setup:**

VecNormalize is applied with `norm_obs=False` (observations are already z-scored) and `norm_reward=True` to stabilize the PPO value function under high reward variance. An EvalCallback monitors performance on the validation split every 5 rollout collections.

---

### MLflow Experiment Tracking

Every training run logs the following to the MLflow registry:

**Parameters:** total timesteps, learning rate, batch size, PPO epochs, gradient clipping norm, window size, Transformer hyperparameters (d_model, nhead, num_encoder_layers), reward mode, train/val/test split sizes.

**Metrics:** mean reward, standard deviation of reward on the held-out test split.

**Artifacts:** trained model `.zip`, fitted scaler `.pkl`.

The model is registered under a versioned name in the MLflow Model Registry, enabling `load_agent(model_name, version="latest")` at inference time with automatic caching via `@lru_cache`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `portfolio_optimizer` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `ppo_portfolio_agent` | Registered model name |
| `USE_SYNTHETIC_DATA` | `false` | Use GBM synthetic data instead of yfinance |
| `TICKERS` | `SPY,QQQ,GLD,TLT,BTC-USD` | Comma-separated ticker symbols |
| `START_DATE` | `2015-01-01` | Historical data start date |
| `END_DATE` | `2024-12-31` | Historical data end date |
| `WINDOW_SIZE` | `30` | Rolling observation window in trading days |
| `TOTAL_TIMESTEPS` | `100000` | Total PPO training timesteps |
| `REWARD_MODE` | `sharpe` | `sharpe` \| `return` \| `penalized` |
| `D_MODEL` | `64` | Transformer embedding dimension |
| `NHEAD` | `4` | Number of attention heads |
| `NUM_ENCODER_LAYERS` | `2` | Number of Transformer encoder layers |
| `LEARNING_RATE` | `3e-4` | PPO learning rate |
| `BATCH_SIZE` | `64` | PPO minibatch size |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns API version |
| `GET` | `/config` | Returns active backend configuration |
| `POST` | `/portfolio/allocate` | Predict portfolio weights using the trained agent |
| `POST` | `/backtest/run` | Run historical simulation and return performance statistics |

Full interactive documentation is available at `http://localhost:8000/docs` when the API is running.

---

## Data & Legal Notice

This project accesses market data through [yfinance](https://github.com/ranaroussi/yfinance), an open-source library that retrieves publicly available data from Yahoo Finance.

- Raw financial data is **never stored in this repository** — the `data/` directory is gitignored
- Trained model weights derived from market data are **not redistributed**
- This project is **strictly non-commercial** and intended solely for educational demonstration
- Users are solely responsible for complying with [Yahoo Finance's Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
- **This project does not constitute financial advice.** Nothing in this repository should be interpreted as a recommendation to buy, sell, or hold any financial instrument

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The MIT License applies to the source code only. It does not grant any rights to financial data retrieved at runtime via yfinance, which remains subject to Yahoo Finance's terms of service.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | yfinance, pandas, NumPy |
| Feature Engineering | NumPy, pandas |
| Deep Learning | PyTorch |
| Reinforcement Learning | Stable-Baselines3 (PPO), Gymnasium |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Next.js, React, TypeScript |

---

*This project is built for educational and demonstration purposes only and is not a production trading system.*