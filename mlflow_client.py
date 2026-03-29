"""
mlflow_client.py — Centralised MLflow wrapper for the Portfolio Optimizer.

All training, inference, and evaluation code interacts with MLflow through
this module so that the tracking URI and experiment configuration are
managed in one place.

For educational and demonstration purposes only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc

from config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME

logger = logging.getLogger(__name__)


class PortfolioMLflowClient:
    """Thin wrapper around the MLflow Python client.

    Instantiate once at application startup (e.g. in the FastAPI lifespan
    handler or at the top of the training script) and reuse throughout.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI.  Defaults to ``config.MLFLOW_TRACKING_URI``.
    experiment_name:
        Name of the MLflow experiment to create or reuse.
    """

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = EXPERIMENT_NAME,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(
            "MLflow client initialised. Tracking URI: %s | Experiment: %s",
            tracking_uri,
            experiment_name,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        model_path: str | Path,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start an MLflow run, log params/metrics, and save the model artifact.

        Parameters
        ----------
        run_name:
            Human-readable label for the run (e.g. ``"ppo_sharpe_100k"``).
        params:
            Hyperparameters to log (e.g. learning rate, window size).
        metrics:
            Scalar evaluation metrics (e.g. Sharpe ratio, max drawdown).
        model_path:
            Local path to the saved SB3 model file (``.zip``).
        tags:
            Optional key-value tags attached to the run.

        Returns
        -------
        str
            The MLflow ``run_id`` for this run.
        """
        model_path = Path(model_path)
        tags = tags or {}

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")
                logger.info("Logged model artifact: %s", model_path)
            else:
                logger.warning("Model path does not exist, skipping artifact: %s", model_path)

            run_id = run.info.run_id
            logger.info("MLflow run logged. run_id=%s", run_id)

        return run_id

    def register_model(
        self,
        run_id: str,
        model_name: str = MODEL_NAME,
    ) -> None:
        """Register a previously logged model artifact in the Model Registry.

        Parameters
        ----------
        run_id:
            The run that contains the model artifact (returned by ``log_experiment``).
        model_name:
            Name under which to register the model in the registry.
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info("Model registered. name=%s, run_id=%s", model_name, run_id)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_registered_model(
        self,
        model_name: str = MODEL_NAME,
        version: str | int = "latest",
    ) -> Any:
        """Load a registered SB3 model from the MLflow Model Registry.

        Parameters
        ----------
        model_name:
            Registry model name.
        version:
            Registry version number or ``"latest"`` to load the most recent.

        Returns
        -------
        Any
            The loaded Stable-Baselines3 model object.

        Raises
        ------
        mlflow.exceptions.MlflowException
            If the model or version is not found in the registry.
        """
        from stable_baselines3 import PPO

        if version == "latest":
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name)
            if not versions:
                raise ValueError(
                    f"No registered versions found for model '{model_name}'."
                )
            version = versions[-1].version
            logger.info("Loading latest version (%s) of model '%s'", version, model_name)

        model_uri = f"models:/{model_name}/{version}"
        local_path = mlflow.artifacts.download_artifacts(model_uri)

        # SB3 saves models as .zip; find the file inside the downloaded directory
        import os
        zip_files = [f for f in os.listdir(local_path) if f.endswith(".zip")]
        if not zip_files:
            raise FileNotFoundError(
                f"No .zip model file found in downloaded artifacts at {local_path}"
            )

        model = PPO.load(os.path.join(local_path, zip_files[0]))
        logger.info("Model loaded successfully from registry. name=%s, version=%s", model_name, version)
        return model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_best_run(
        self,
        metric: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch the run with the best value for *metric* in the experiment.

        Parameters
        ----------
        metric:
            MLflow metric name to rank by.
        ascending:
            If ``True``, lower is better (e.g. ``max_drawdown``).

        Returns
        -------
        dict or None
            MLflow run data dict, or ``None`` if no runs exist.
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if runs.empty:
            return None
        return runs.iloc[0].to_dict()
