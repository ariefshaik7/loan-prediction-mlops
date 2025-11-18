import os

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import LoanFeatures

load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow_cfg = config["mlflow_config"]
mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

MODEL_URI = os.getenv("MODEL_URI")

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    model_source = MODEL_URI
    print(f"Loaded model from: {MODEL_URI}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model from MLflow: {e}")


app = FastAPI(title="Loan Approval API")


@app.get("/health")
def health():
    return {"status": "ok", "model_source": model_source}


@app.post("/predict")
def predict(data: list[LoanFeatures]):
    try:
        df = pd.DataFrame([d.dict() for d in data])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
