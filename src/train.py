import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(config):
    """Sets the MLflow tracking URI and experiment name."""
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_config"]["tracking_uri"]
    mlflow.set_experiment(config["mlflow_config"]["experiment_name"])


def train_model(config):
    """
    Loads training data, tunes, and saves the model pipeline.
    Logs all parameters, metrics, and artifacts to MLflow.
    """
    setup_mlflow(config)

    print("Starting model training...")
    train_file = config["data"]["train_file"]

    try:
        train_df = pd.read_csv(train_file)
    except FileNotFoundError:
        print(
            f"Error: Training file '{train_file}' not found. Run preprocess.py first."
        )
        return

    X_train = train_df.drop("loan_status", axis=1)
    y_train = train_df["loan_status"]

    numerical_cols = X_train.select_dtypes(include=np.number).columns
    categorical_cols = X_train.select_dtypes(include="object").columns

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                xgb.XGBClassifier(
                    random_state=config["preprocessing"]["random_state"],
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    param_grid = config["training"]["param_grid"]

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=config["training"]["cv_folds"],
        scoring=config["training"]["scoring"],
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        mlflow.log_param("random_state", config["preprocessing"]["random_state"])
        mlflow.log_param("cv_folds", config["training"]["cv_folds"])
        mlflow.log_param("scoring", config["training"]["scoring"])

        print("Running GridSearchCV...")
        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        # Log tuning results
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        best_model = grid_search.best_estimator_

        input_example = X_train.head(5)
        signature = infer_signature(
            input_example, best_model.predict_proba(input_example)
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        print("Model logged to MLflow.")

    print("\nTraining complete.")


if __name__ == "__main__":
    config = load_config()
    train_model(config)
