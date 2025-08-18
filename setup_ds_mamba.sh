#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ds}"

# 1) Asegurar mamba (si usás conda pura, podés: conda install -n base -c conda-forge mamba -y)
#    Recomendado: Mambaforge o Micromamba instalados previamente

# 2) Crear entorno base
mamba create -y -n "$ENV_NAME" -c conda-forge python=3.11

# 3) Instalar paquetes DS + JupyterLab + MLflow
mamba install -y -n "$ENV_NAME" -c conda-forge \
  jupyterlab ipykernel mlflow numpy pandas scikit-learn lightgbm xgboost \
  pyarrow polars matplotlib jupytext

# 4) Registrar kernel
mamba run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 5) Carpetas
mkdir -p notebooks artifacts

# 6) Levantar servicios (dos terminales)
# JupyterLab
( mamba run -n "$ENV_NAME" jupyter lab & ) >/dev/null 2>&1
# MLflow
( mamba run -n "$ENV_NAME" mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 127.0.0.1 --port 5000 & ) >/dev/null 2>&1

echo
echo "Listo. Abrí JupyterLab y usá el kernel: Python ($ENV_NAME)"
echo "MLflow UI: http://127.0.0.1:5000"
