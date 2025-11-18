import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(config):
    """
    Loads raw data, splits it, and saves the train/test sets.
    """
    raw_path = config["data"]["raw_file"]
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
    train_df.to_csv(config["data"]["train_file"], index=False)
    test_df.to_csv(config["data"]["test_file"], index=False)

    print("Preprocessing complete.")
    print(f"Train data saved to: {config['data']['train_file']}")
    print(f"Test data saved to: {config['data']['test_file']}")


if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)
