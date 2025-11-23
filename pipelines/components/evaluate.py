import yaml
from kfp.dsl import component

# Load config
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)

BASE_IMAGE = full_config["system"]["docker_image"]


@component(base_image=BASE_IMAGE)
def evaluate_op(config_path: str, run_id: str):
    import os
    import sys

    sys.path.append("/app")

    os.environ["MLFLOW_RUN_ID"] = run_id

    from src.evaluate import evaluate_model, load_config, setup_mlflow

    print(f"--- Evaluation Started (run_id={run_id}) ---")
    config = load_config(config_path)
    setup_mlflow(config)
    evaluate_model(config)
    print("--- Evaluation Finished ---")
