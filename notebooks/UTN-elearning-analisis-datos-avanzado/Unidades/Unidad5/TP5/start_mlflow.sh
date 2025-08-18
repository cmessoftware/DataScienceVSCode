#!/bin/bash

# Clear screen and show header
clear
echo ""
echo -e "\033[36m==========================================\033[0m"
echo -e "\033[36m   MLflow UI Dashboard Launcher\033[0m"
echo -e "\033[36m==========================================\033[0m"
echo ""

# Change to script directory
cd "$(dirname "$0")"

echo -e "\033[33m📁 Directorio actual: $(pwd)\033[0m"
echo ""

# Check if MLflow is installed
echo -e "\033[34m🔍 Verificando instalación de MLflow...\033[0m"
mlflow_version=$(python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "\033[32m✅ MLflow encontrado: $mlflow_version\033[0m"
else
    echo -e "\033[31m❌ MLflow no está instalado\033[0m"
    echo -e "\033[33m💡 Instalando MLflow...\033[0m"
    pip install mlflow mlflow[extras]
    if [ $? -ne 0 ]; then
        echo -e "\033[31m❌ Error instalando MLflow\033[0m"
        read -p "Presiona Enter para continuar"
        exit 1
    fi
    echo -e "\033[32m✅ MLflow instalado correctamente\033[0m"
fi

echo ""
echo -e "\033[32m🚀 Iniciando servidor MLflow UI...\033[0m"
echo -e "\033[36m🌐 El dashboard estará disponible en: http://localhost:5000\033[0m"
echo -e "\033[36m🧪 Experimento: Churn_Prediction_TP5\033[0m"
echo ""
echo -e "\033[33m💡 Presiona Ctrl+C para detener el servidor\033[0m"
echo ""

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

echo ""
echo -e "\033[33m👋 Servidor MLflow detenido\033[0m"
read -p "Presiona Enter para continuar"
