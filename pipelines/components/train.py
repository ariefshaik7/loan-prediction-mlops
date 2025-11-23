import yaml
from kfp.dsl import component

# Load config
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)

BASE_IMAGE = full_config["system"]["docker_image"]


@component(base_image=BASE_IMAGE)
def train_op(config_path: str) -> str:
    import sys

    sys.path.append("/app")
    from src.train import load_config, setup_mlflow, train_model

    print("--- Training Started ---")
    config = load_config(config_path)

    setup_mlflow(config)

    run_id = train_model(config)

    print(f"--- Training Finished (run_id={run_id}) ---")
    return run_id
