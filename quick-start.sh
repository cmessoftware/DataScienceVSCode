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

# Verificar si hay algún proceso de Jupyter corriendo usando archivos PID
if [ -f /workspace/.jupyter/jupyter.pid ]; then
    JUPYTER_PID=$(cat /workspace/.jupyter/jupyter.pid)
    if [ ! -z "$JUPYTER_PID" ] && kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "⚠️  Jupyter Lab ya está corriendo (PID: $JUPYTER_PID)"
        echo "🔄 Deteniendo instancia anterior..."
        kill $JUPYTER_PID 2>/dev/null || true
        sleep 2
    fi
    rm -f /workspace/.jupyter/jupyter.pid
fi

echo "🔧 Configurando Jupyter Lab..."

# Crear directorios necesarios si no existen
mkdir -p /workspace/.jupyter
mkdir -p /workspace/notebooks_final

# Configurar Jupyter si no existe la configuración
if [ ! -f /workspace/.jupyter/jupyter_lab_config.py ]; then
    echo "📝 Creando configuración de Jupyter Lab..."
    cd /workspace
    export JUPYTER_CONFIG_DIR=/workspace/.jupyter
    jupyter lab --generate-config --allow-root
fi

echo "🌐 Iniciando Jupyter Lab en segundo plano..."

# Iniciar Jupyter Lab en segundo plano
export JUPYTER_CONFIG_DIR=/workspace/.jupyter
nohup jupyter lab \
    --config=/workspace/.jupyter/jupyter_lab_config_custom.py \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
        --notebook-dir=/workspace/notebooks_final \
    --ServerApp.token='datascience2024' \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    > /workspace/.jupyter/jupyter.log 2>&1 &

JUPYTER_PID=$!
echo $JUPYTER_PID > /workspace/.jupyter/jupyter.pid

# Esperar a que el servicio esté listo
echo "⏳ Esperando a que Jupyter Lab esté listo..."
sleep 5

# Verificar que Jupyter está corriendo usando el archivo PID
if [ -f /workspace/.jupyter/jupyter.pid ] && kill -0 $(cat /workspace/.jupyter/jupyter.pid) 2>/dev/null; then
    JUPYTER_PID=$(cat /workspace/.jupyter/jupyter.pid)
    echo "✅ Jupyter Lab iniciado exitosamente (PID: $JUPYTER_PID)"
    
    # Mostrar información de acceso
    echo ""
    echo "📊 Jupyter Lab está listo!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 URL: http://localhost:8888"
    echo "🔑 Token: datascience2024"
    echo "    echo "📁 Directorio de trabajo: /workspace/notebooks_final""
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
    if [ -f /workspace/.jupyter/jupyter.log ]; then
        echo "📄 Últimas líneas del log:"
        tail -10 /workspace/.jupyter/jupyter.log
    fi
    exit 1
fi
echo ""
echo "🎉 ¡Jupyter Lab iniciado exitosamente!"
echo "ℹ️  Para ver el estado: cat /workspace/.jupyter/jupyter.pid"