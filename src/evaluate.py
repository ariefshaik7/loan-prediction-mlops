import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(config):
    """Sets the MLflow tracking URI and experiment name."""
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_config"]["tracking_uri"]
    mlflow.set_experiment(config["mlflow_config"]["experiment_name"])


# --- Plotting Functions ---


def plot_confusion_matrix(y_true, y_pred):
    """Generates a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    return plt.gcf()


def plot_roc_curve(y_true, y_proba):
    """Generates an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="orange")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def plot_feature_importance(model):
    """Generates a feature importance plot."""
    try:
        model_features = model.named_steps["model"].get_booster().feature_names
        importance = model.named_steps["model"].feature_importances_

        fi_df = pd.DataFrame(
            {"Feature": model_features, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Importance",
            y="Feature",
            data=fi_df.head(15),
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")
        return None


# --- Main Evaluation Function ---


def evaluate_model(config):
    """
    Loads the test set and saved model, logs evaluation
    metrics and plots to the *original MLflow run*.
    """
    setup_mlflow(config)

    print("Starting model evaluation...")

    # load test data
    test_df = pd.read_csv(config["data"]["test_file"])

    # determine run_id: prefer env var MLFLOW_RUN_ID, otherwise find latest run for the experiment
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        client = MlflowClient()
        exp = mlflow.get_experiment_by_name(config["mlflow_config"]["experiment_name"])
        if exp is None:
            raise RuntimeError(
                "Experiment not found. Ensure experiment name in config.yaml is correct."
            )
        runs = client.search_runs(
            exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1
        )
        if not runs:
            raise RuntimeError(
                "No runs found for experiment; set MLFLOW_RUN_ID or run training first."
            )
        run_id = runs[0].info.run_id

    # load model from MLflow run artifacts
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"Loaded model from run: {run_id}")

    X_test = test_df.drop("loan_status", axis=1)
    y_test = test_df["loan_status"]

    

    with mlflow.start_run(run_id=run_id) as run:
        print("Re-opened MLflow run to log evaluation metrics.")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)

        print("\n--- Evaluation Metrics on Test Set ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report_str)

        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision_class_0", report_dict["0"]["precision"])
        mlflow.log_metric("test_recall_class_0", report_dict["0"]["recall"])
        mlflow.log_metric("test_f1_class_0", report_dict["0"]["f1-score"])
        mlflow.log_metric("test_precision_class_1", report_dict["1"]["precision"])
        mlflow.log_metric("test_recall_class_1", report_dict["1"]["recall"])
        mlflow.log_metric("test_f1_class_1", report_dict["1"]["f1-score"])


        # Generate, save, and log plots
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "reports/confusion_matrix.png")
        plt.close(cm_fig)

        roc_fig = plot_roc_curve(y_test, y_proba)
        mlflow.log_figure(roc_fig, "reports/roc_curve.png")
        plt.close(roc_fig)

        fi_fig = plot_feature_importance(model)
        if fi_fig:
            mlflow.log_figure(fi_fig, "reports/feature_importance.png")
            plt.close(fi_fig)

    print("\nEvaluation complete. All metrics and plots logged to MLflow.")


if __name__ == "__main__":
    config = load_config()
    evaluate_model(config)
