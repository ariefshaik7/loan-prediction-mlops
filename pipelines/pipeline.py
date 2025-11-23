import yaml
from kfp import dsl, kubernetes

from pipelines.components.evaluate import evaluate_op
from pipelines.components.ingest import ingest_op
from pipelines.components.preprocess import preprocess_op
from pipelines.components.train import train_op

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_URL = CONFIG["data"]["raw_url"]


# --- 1. Define the Custom PVC Creator Component (The "New Way") ---
@dsl.component(packages_to_install=["kubernetes"])
def create_pvc_op(
    pvc_name: str, size: str, storage_class: str, access_mode: str
) -> str:
    """
    Custom component to create a PVC using the Kubernetes Python Client.
    This replaces the deprecated VolumeOp and the buggy kfp.kubernetes.CreatePVC.
    """
    import time

    from kubernetes import client, config

    print(f"--- Creating PVC: {pvc_name} ---")

    # 1. Authenticate with the cluster
    try:
        config.load_incluster_config()
        print("Loaded in-cluster config.")
    except Exception:
        print(
            "Warning: Could not load in-cluster config. Trying local (only works if running locally)."
        )
        config.load_kube_config()

    v1 = client.CoreV1Api()

    # 2. Define the PVC object
    pvc_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": pvc_name
            # Namespace is automatic (defaults to the workflow's namespace)
        },
        "spec": {
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": size}},
            # If storage_class is not empty, set it
            "storageClassName": storage_class if storage_class else None,
        },
    }

    # 3. Create it (Idempotent check)
    try:
        # We need the current namespace. In KFP, it's usually in /var/run/secrets...
        # or we can read it from the service account context.
        # However, create_namespaced_pvc requires a namespace.
        # A safe bet in KFP is to read the namespace file standard in K8s pods.
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            current_namespace = f.read().strip()

        print(f"Target Namespace: {current_namespace}")

        v1.create_namespaced_persistent_volume_claim(
            namespace=current_namespace, body=pvc_manifest
        )
        print(f"PVC {pvc_name} created successfully.")

    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"PVC {pvc_name} already exists. Using existing volume.")
        else:
            raise e

    # 4. Return the name so subsequent steps can mount it
    return pvc_name


# --- 2. The Pipeline Definition ---
@dsl.pipeline(
    name="Loan Prediction Pipeline",
    description="End-to-end MLOps pipeline using Custom PVC Component",
)
def loan_prediction_pipeline(
    data_url: str = DATA_URL,
    config_path: str = "config.yaml",
):
    # --- Infrastructure Phase ---
    # Call our custom component to create the volume
    pvc_task = create_pvc_op(
        pvc_name="loan-data-pvc-v2",  # Static name is safer for re-runs
        size="1Gi",
        storage_class="microk8s-hostpath",  # <--- CHANGE IF NEEDED (e.g. 'standard', 'managed-csi')
        access_mode="ReadWriteOnce",
    )

    # --- Data Phase ---
    # Ingest Data
    ingest = ingest_op(
        data_url=data_url, output_data_path="/app/data/raw/loan_dataset.csv"
    )
    # Ensure Ingest happens AFTER PVC creation
    ingest.after(pvc_task)

    # Mount the PVC created by the previous step
    kubernetes.mount_pvc(ingest, pvc_name=pvc_task.output, mount_path="/app/data")

    # --- Preprocessing Phase ---
    preprocess = preprocess_op(config_path=config_path)
    preprocess.after(ingest)
    kubernetes.mount_pvc(preprocess, pvc_name=pvc_task.output, mount_path="/app/data")
    
    kubernetes.use_secret_as_env(
        preprocess,
        secret_name="databricks-secret",
        secret_key_to_env={"host": "DATABRICKS_HOST", "token": "DATABRICKS_TOKEN"},
    )
    
    # --- Training Phase ---
    train = train_op(config_path=config_path)
    train.after(preprocess)
    kubernetes.mount_pvc(train, pvc_name=pvc_task.output, mount_path="/app/data")

    # Inject Databricks Credentials
    kubernetes.use_secret_as_env(
        train,
        secret_name="databricks-secret",
        secret_key_to_env={"host": "DATABRICKS_HOST", "token": "DATABRICKS_TOKEN"},
    )

    # --- Evaluation Phase ---
    evaluate = evaluate_op(config_path=config_path, run_id=train.output)
    evaluate.after(train)
    kubernetes.mount_pvc(evaluate, pvc_name=pvc_task.output, mount_path="/app/data")

    kubernetes.use_secret_as_env(
        evaluate,
        secret_name="databricks-secret",
        secret_key_to_env={"host": "DATABRICKS_HOST", "token": "DATABRICKS_TOKEN"},
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(loan_prediction_pipeline, "loan_pipeline.yaml")
