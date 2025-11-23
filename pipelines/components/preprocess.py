import yaml
from kfp.dsl import component

# Load config
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)

BASE_IMAGE = full_config["system"]["docker_image"]


@component(base_image=BASE_IMAGE)
def preprocess_op(config_path: str):
    import sys

    sys.path.append("/app")
    from src.preprocess import load_config, setup_mlflow, preprocess_data

    print("--- Preprocessing Started ---")
    config = load_config(config_path)
    setup_mlflow(config)
    preprocess_data(config)
    print("--- Preprocessing Finished ---")
