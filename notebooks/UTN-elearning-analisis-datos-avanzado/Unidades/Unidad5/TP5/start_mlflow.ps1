# Clear screen and show header
Clear-Host
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   MLflow UI Dashboard Launcher" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location -Path $PSScriptRoot

Write-Host "📁 Directorio actual: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check if MLflow is installed
Write-Host "🔍 Verificando instalación de MLflow..." -ForegroundColor Blue
try {
    $mlflowVersion = python -c "import mlflow; print(mlflow.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ MLflow encontrado: $mlflowVersion" -ForegroundColor Green
    } else {
        throw "MLflow not found"
    }
} catch {
    Write-Host "❌ MLflow no está instalado" -ForegroundColor Red
    Write-Host "💡 Instalando MLflow..." -ForegroundColor Yellow
    pip install mlflow mlflow[extras]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error instalando MLflow" -ForegroundColor Red
        Read-Host "Presiona Enter para continuar"
        exit 1
    }
    Write-Host "✅ MLflow instalado correctamente" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Iniciando servidor MLflow UI..." -ForegroundColor Green
Write-Host "🌐 El dashboard estará disponible en: http://localhost:5000" -ForegroundColor Cyan
Write-Host "🧪 Experimento: Churn_Prediction_TP5" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

Write-Host ""
Write-Host "👋 Servidor MLflow detenido" -ForegroundColor Yellow
Read-Host "Presiona Enter para continuar"
