import os

import pandas as pd
import yaml
import mlflow
from dotenv.main import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()  # for local dev

def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    """Sets the MLflow tracking URI and experiment name."""
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_config"]["tracking_uri"]
    mlflow.set_experiment(config["mlflow_config"]["experiment_name"])

def preprocess_data(config):
    """
    Loads raw data, splits it, and saves the train/test sets, AND logs them to MLflow.
    """
    # 1. Setup MLflow connection
    setup_mlflow(config)
    
    raw_path = config["data"]["raw_url"]
    print(f"Starting preprocessing of {raw_path}...")

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Error: Raw data file '{raw_path}' not found.")
        return

    # Drop ID column
    if "loan_id" in df.columns:
        df = df.drop("loan_id", axis=1)

    # Map target variable
    df["loan_status"] = (
        df["loan_status"]
        .str.strip()
        .replace({"Approved": 1, "Rejected": 0})
        .infer_objects(copy=False)
    )

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["preprocessing"]["random_state"],
        stratify=y,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Create output directory
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    
    train_file_path = config["data"]["train_file"]
    test_file_path = config["data"]["test_file"]
        
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print("Preprocessing complete.")
    print(f"Train data saved to: {train_file_path}")
    print(f"Test data saved to: {test_file_path}")
    
    # --- 2. Log Artifacts to MLflow ---
    print("Logging datasets to MLflow...")
    with mlflow.start_run(run_name="Preprocess_Step") as run:
        mlflow.log_artifact(train_file_path, artifact_path="datasets")
        mlflow.log_artifact(test_file_path, artifact_path="datasets")
        print(f"Datasets logged to run: {run.info.run_id}")

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)
