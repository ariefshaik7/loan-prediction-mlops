import yaml
from kfp.dsl import component

# Load config
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)

BASE_IMAGE = full_config["system"]["docker_image"]


@component(base_image=BASE_IMAGE, packages_to_install=["pandas", "requests"])
def ingest_op(data_url: str, output_data_path: str):
    import os

    import pandas as pd

    print(f"--- Downloading data from {data_url} ---")

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    df = pd.read_csv(data_url)

    df.to_csv(output_data_path, index=False)

    print(f"--- Data saved to {output_data_path} ---")
