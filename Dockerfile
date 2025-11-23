FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install "dvc[azure]>=3.0" "mlflow" "pandas" "scikit-learn" "xgboost" "pyyaml" "kfp" "python-dotenv" "requests" "seaborn" "matplotlib"

# Copy Code and Config (BUT NOT DATA)
COPY src/ ./src/
COPY config.yaml .

# Create the data folder structure (empty) for the volume to mount into
RUN mkdir -p data/processed data/raw

# Ensure python can find your src module
ENV PYTHONPATH="/app"
