#!/bin/bash

# Quick Start Script for Jupyter Lab (Inside Container)
# Este script inicia Jupyter Lab desde dentro del contenedor

set -e  # Salir si ocurre algún error

echo "🚀 Iniciando Jupyter Lab..."

# Verificar si Jupyter está instalado
if ! command -v jupyter &> /dev/null; then
    echo "❌ Error: Jupyter no está instalado. Instalando..."
    pip install jupyterlab jupyter
fi

# Verificar si hay algún proceso de Jupyter corriendo
JUPYTER_PID=$(ps aux | grep '[j]upyter-lab' | awk '{print $2}' | head -1)

if [ ! -z "$JUPYTER_PID" ]; then
    echo "⚠️  Jupyter Lab ya está corriendo (PID: $JUPYTER_PID)"
    echo "🔄 Deteniendo instancia anterior..."
    kill $JUPYTER_PID 2>/dev/null || true
    sleep 2
fi

echo "🔧 Configurando Jupyter Lab..."

# Crear directorios necesarios si no existen
mkdir -p /workspace/.jupyter
mkdir -p /workspace/notebooks

# Configurar Jupyter si no existe la configuración
if [ ! -f /workspace/.jupyter/jupyter_lab_config.py ]; then
    echo "� Creando configuración de Jupyter Lab..."
    jupyter lab --generate-config --config-dir=/workspace/.jupyter
fi

echo "🌐 Iniciando Jupyter Lab en segundo plano..."

# Iniciar Jupyter Lab en segundo plano
nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=/workspace \
    --config-dir=/workspace/.jupyter \
    --NotebookApp.token='datascience2024' \
    --NotebookApp.password='' \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.disable_check_xsrf=True \
    > /workspace/.jupyter/jupyter.log 2>&1 &

JUPYTER_PID=$!
echo $JUPYTER_PID > /workspace/.jupyter/jupyter.pid

# Esperar a que el servicio esté listo
echo "⏳ Esperando a que Jupyter Lab esté listo..."
sleep 5

# Verificar que Jupyter está corriendo
if ps -p $JUPYTER_PID > /dev/null 2>&1; then
    echo "✅ Jupyter Lab iniciado exitosamente (PID: $JUPYTER_PID)"
    
    # Mostrar información de acceso
    echo ""
    echo "📊 Jupyter Lab está listo!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 URL: http://localhost:8888"
    echo "🔑 Token: datascience2024"
    echo "📁 Directorio de trabajo: /workspace"
    echo "📋 Logs: /workspace/.jupyter/jupyter.log"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 Comandos útiles:"
    echo "   • Ver logs: tail -f /workspace/.jupyter/jupyter.log"
    echo "   • Parar Jupyter: ./quick-stop.sh"
    echo "   • Reiniciar: ./quick-stop.sh && ./quick-start.sh"
    echo ""
    
    # Intentar abrir en el navegador del host (si está disponible)
    if [ ! -z "$BROWSER" ]; then
        echo "🌍 Abriendo Jupyter Lab en el navegador del host..."
        "$BROWSER" "http://localhost:8888/lab?token=datascience2024" &
    else
        echo "💻 Abre manualmente en el navegador del host: http://localhost:8888/lab?token=datascience2024"
    fi
    
else
    echo "❌ Error: Jupyter Lab no se pudo iniciar correctamente."
    echo "📋 Revisa los logs en: /workspace/.jupyter/jupyter.log"
    exit 1
fi

echo ""
echo "🎉 ¡Jupyter Lab iniciado exitosamente!"
echo "ℹ️  Para ver el estado: ps aux | grep jupyter"