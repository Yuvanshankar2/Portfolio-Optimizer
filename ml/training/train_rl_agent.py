"""
train_rl_agent.py — End-to-end PPO training pipeline for the Portfolio Optimizer.

Pipeline:
    1. Load or generate market data (yfinance or synthetic GBM)
    2. Compute normalized feature array
    3. Instantiate the Gymnasium PortfolioEnv
    4. Build a PPO agent with the TransformerExtractor feature extractor
    5. Train for ``total_timesteps`` steps
    6. Evaluate on the same environment (deterministic rollouts)
    7. Log hyperparameters, metrics, and the saved model to MLflow
    8. Register the model in the MLflow Model Registry

Run from project root:
    python ml/training/train_rl_agent.py
    python ml/training/train_rl_agent.py --timesteps 10000 --run-name smoke_test

For educational and demonstration purposes only.
"""

from __future__ import annotations

import argparse
import logging
import dotenv

from pathlib import Path
from typing import Any
import numpy as np

from config import (
    BATCH_SIZE,
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    EXPERIMENT_NAME,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_NAME,
    MODELS_DIR,
    N_EPOCHS,
    N_STEPS,
    NAN_THRESHOLD,
    NHEAD,
    NUM_ENCODER_LAYERS,
    REWARD_MODE,
    TICKERS,
    TOTAL_TIMESTEPS,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_feature_array(df) -> np.ndarray:
    """Convert raw ``load_data()`` DataFrame to ``(T, assets, features)`` array.

    Parameters
    ----------
    df:
        Multi-index OHLCV DataFrame from ``data_pipeline.ingestion.yfinance_fetcher.load_data``.

    Returns
    -------
    np.ndarray
        Shape ``(T, num_assets, num_features)`` of float32 features.
    """
    from data_pipeline.ingestion.yfinance_fetcher import compute_features

    features = compute_features(df)  # (T, assets, feats)
    logger.info("Feature array: shape=%s, dtype=%s", features.shape, features.dtype)
    return features.astype(np.float32)


def make_env(
    price_data: np.ndarray,
    reward_mode: str = REWARD_MODE,
) -> "PortfolioEnv":
    """Instantiate a PortfolioEnv wrapped with SB3's Monitor.

    Parameters
    ----------
    price_data:
        Feature array ``(T, num_assets, num_features)``.
    reward_mode:
        Reward mode passed to ``PortfolioEnv``.

    Returns
    -------
    PortfolioEnv
        Monitor-wrapped environment ready for DummyVecEnv.
    """
    from stable_baselines3.common.monitor import Monitor

    from ml.environments.portfolio_env import PortfolioEnv

    env = PortfolioEnv(
        price_data=price_data,
        window_size=WINDOW_SIZE,
        reward_mode=reward_mode,
    )
    return Monitor(env)


def build_ppo_agent(env) -> "PPO":
    """Construct a PPO agent with the Transformer feature extractor.

    Parameters
    ----------
    env:
        A VecEnv (e.g. DummyVecEnv wrapping Monitor(PortfolioEnv)).

    Returns
    -------
    stable_baselines3.PPO
        Untrained PPO agent ready for ``.learn()``.
    """
    from stable_baselines3 import PPO

    from ml.models.transformer_extractor import TransformerExtractor

    policy_kwargs: dict[str, Any] = {
        "features_extractor_class": TransformerExtractor,
        "features_extractor_kwargs": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
        },
    }

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,   # explicit gradient clipping (0.5)
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    logger.info("PPO agent constructed with TransformerExtractor.")
    return agent


def evaluate_agent(
    agent,
    env,
    n_eval_episodes: int = 5,
) -> dict[str, float]:
    """Run deterministic evaluation rollouts and return summary metrics.

    Parameters
    ----------
    agent:
        Trained ``PPO`` agent.
    env:
        An unwrapped ``PortfolioEnv`` (not VecEnv).
    n_eval_episodes:
        Number of episodes to average over.

    Returns
    -------
    dict
        Keys: ``mean_reward``, ``std_reward``, ``mean_ep_length``.
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        agent, env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    logger.info(
        "Evaluation: mean_reward=%.4f ± %.4f over %d episodes",
        mean_reward, std_reward, n_eval_episodes,
    )
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_ep_length": 0.0,  # placeholder; use EvalCallback for detailed tracking
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    run_name: str = "ppo_run",
    reward_mode: str = REWARD_MODE,
    save_path: Path | None = None,
) -> str:
    """Run the full training pipeline and log results to MLflow.

    Parameters
    ----------
    total_timesteps:
        Total environment steps for PPO training.
    run_name:
        MLflow run name label.
    reward_mode:
        Reward mode for ``PortfolioEnv``.
    save_path:
        Where to save the trained model ``.zip``.  Defaults to
        ``models/{run_name}.zip``.

    Returns
    -------
    str
        MLflow ``run_id`` for the completed run.
    """
    import mlflow

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        # everything below goes INSIDE this block
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from data_pipeline.ingestion.yfinance_fetcher import (
            apply_scaler,
            fit_scaler,
            load_data,
            split_features,
        )
        from ml.environments.portfolio_env import PortfolioEnv

        # ── 1. Raw feature array ────────────────────────────────────────────────
        logger.info("Loading data...")
        df = load_data(tickers=TICKERS)
        raw_features = build_feature_array(df)   # (T, assets, features) — unscaled
        nan_count = np.isnan(raw_features).sum()
        if nan_count > 0:
            raise ValueError(f"NaNs found in feature array: {nan_count}")
        # ── 2. Time-based split — NO shuffling, preserves temporal order ────────
        # train → scaler fitting + PPO training
        # val   → EvalCallback during training (early stopping signal)
        # test  → final unseen evaluation reported to MLflow
        train_data, val_data, test_data = split_features(
            raw_features, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
        )

        # ── 3. Scaler fitted on training data ONLY ──────────────────────────────
        # Applying train statistics to val/test is the only way to prevent
        # look-ahead leakage through the normalisation step.
        scaler = fit_scaler(train_data)
        train_data = apply_scaler(train_data, scaler)
        val_data   = apply_scaler(val_data,   scaler)
        test_data  = apply_scaler(test_data,  scaler)

        # ── 4. Training environment with reward normalisation ───────────────────
        # VecNormalize(norm_obs=False) because observations are already z-scored.
        # norm_reward=True stabilises PPO's value function when reward variance is
        # high (as is typical for Sharpe-ratio rewards on financial data).
        logger.info("Creating PortfolioEnv (training split: %d steps)...", len(train_data))
        env_fn = lambda: make_env(train_data, reward_mode=reward_mode)
        vec_env = DummyVecEnv([env_fn])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=5.0,
            gamma=0.99,
        )

        # ── 5. Validation callback (runs on val split, not train) ───────────────
        from stable_baselines3.common.vec_env import DummyVecEnv

        val_env = DummyVecEnv([lambda: make_env(val_data, reward_mode=reward_mode)])

        val_env = VecNormalize(
            val_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=5.0,
            gamma=0.99,
        )
        eval_callback = EvalCallback(
            val_env,
            eval_freq=N_STEPS * 5,   # every 5 rollout collections
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )

        # ── 6. Build and train agent ────────────────────────────────────────────
        agent = build_ppo_agent(vec_env)
        logger.info("Training for %d timesteps...", total_timesteps)
        agent.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
        vec_env.save("vecnormalize.pkl")
        # ── 7. Save model + scaler ──────────────────────────────────────────────
        save_path = save_path or (MODELS_DIR / f"{run_name}.zip")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(save_path))
        logger.info("Model saved: %s", save_path)

        # Scaler is saved alongside the model so inference can apply the same
        # normalisation without re-fitting on live data.
        import joblib
        import shutil

        scaler_path = save_path.with_suffix(".scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved: %s", scaler_path)

        # Copy to stable "latest" paths so inference can always find them
        # without knowing the run_name used at training time.
        shutil.copy(str(scaler_path), str(MODELS_DIR / "latest.scaler.pkl"))
        vec_env.save(str(MODELS_DIR / "latest.vecnormalize.pkl"))
        logger.info("Copied scaler and vecnormalize to latest paths in %s.", MODELS_DIR)
        # ── 8. Final evaluation on TEST split only ──────────────────────────────
        # No VecNormalize on the test env — we want raw reward values in metrics.
        test_env = make_env(test_data, reward_mode=reward_mode)
        metrics = evaluate_agent(agent, test_env)
        logger.info(
            "Test evaluation: mean_reward=%.4f ± %.4f (unseen data)",
            metrics["mean_reward"], metrics["std_reward"],
        )
        params = {
            "total_timesteps": total_timesteps,
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "max_grad_norm": MAX_GRAD_NORM,
            "window_size": WINDOW_SIZE,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "reward_mode": reward_mode,
            "num_assets": raw_features.shape[1],
        # Split info
            "train_steps": len(train_data),
            "val_steps":   len(val_data),
            "test_steps":  len(test_data),
            "train_ratio": TRAIN_RATIO,
            "val_ratio":   VAL_RATIO,
            "nan_threshold": NAN_THRESHOLD,
        }
        # ── 9. Log to MLflow ────────────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # artifact_path="model" so register_model() can use model_uri=runs:/{id}/model
        mlflow.log_artifact(str(save_path), artifact_path="model")
        mlflow.log_artifact(str(scaler_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id

        # ── 10. Register model ──────────────────────────────────────────────────
        client = mlflow.tracking.MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass  # already exists, that's fine
        client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        logger.info("Model registered: %s", MODEL_NAME)
        return run_id

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Optimizer — RL Training")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total PPO training timesteps")
    parser.add_argument("--run-name", type=str, default="ppo_run",
                        help="MLflow run name")
    parser.add_argument("--reward-mode", type=str, default=REWARD_MODE,
                        choices=["sharpe", "return", "penalized"],
                        help="Reward function for PortfolioEnv")
    args = parser.parse_args()

    run_id = train(
        total_timesteps=args.timesteps,
        run_name=args.run_name,
        reward_mode=args.reward_mode,
    )
    print(f"\nDone. MLflow run_id: {run_id}")
