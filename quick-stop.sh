#!/bin/bash

# Quick Stop Script for Jupyter Lab (Inside Container)
# Este script detiene Jupyter Lab desde dentro del contenedor

set -e  # Salir si ocurre algún error

echo "🛑 Deteniendo Jupyter Lab..."

# Buscar procesos de Jupyter corriendo
JUPYTER_PIDS=$(ps aux | grep '[j]upyter-lab' | awk '{print $2}')

if [ -z "$JUPYTER_PIDS" ]; then
    echo "ℹ️  No hay procesos de Jupyter Lab corriendo."
else
    echo "🔧 Deteniendo procesos de Jupyter Lab..."
    
    for PID in $JUPYTER_PIDS; do
        echo "   • Deteniendo proceso $PID..."
        kill $PID 2>/dev/null || true
    done
    
    # Esperar un poco para que los procesos terminen
    sleep 2
    
    # Verificar si aún hay procesos corriendo y forzar si es necesario
    REMAINING_PIDS=$(ps aux | grep '[j]upyter-lab' | awk '{print $2}')
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo "⚠️  Forzando detención de procesos restantes..."
        for PID in $REMAINING_PIDS; do
            kill -9 $PID 2>/dev/null || true
        done
    fi
    
    echo "✅ Procesos de Jupyter Lab detenidos."
fi

# Limpiar archivos de PID si existen
if [ -f /workspace/.jupyter/jupyter.pid ]; then
    rm -f /workspace/.jupyter/jupyter.pid
    echo "🧹 Archivo PID eliminado."
fi

# Verificar puertos en uso
PORT_IN_USE=$(netstat -tlnp 2>/dev/null | grep ':8888 ' || true)
if [ ! -z "$PORT_IN_USE" ]; then
    echo "⚠️  El puerto 8888 aún está en uso:"
    echo "$PORT_IN_USE"
    
    # Intentar liberar el puerto
    PID_USING_PORT=$(echo "$PORT_IN_USE" | awk '{print $7}' | cut -d'/' -f1)
    if [ ! -z "$PID_USING_PORT" ] && [ "$PID_USING_PORT" != "-" ]; then
        echo "🔧 Liberando puerto 8888 (PID: $PID_USING_PORT)..."
        kill -9 $PID_USING_PORT 2>/dev/null || true
    fi
fi

echo ""
echo "📊 Estado actual de procesos Jupyter:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
JUPYTER_CHECK=$(ps aux | grep '[j]upyter' || echo "Ningún proceso de Jupyter encontrado")
echo "$JUPYTER_CHECK"

echo ""
echo "🌐 Estado del puerto 8888:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PORT_CHECK=$(netstat -tlnp 2>/dev/null | grep ':8888' || echo "Puerto 8888 libre")
echo "$PORT_CHECK"

# Opción para limpiar archivos temporales
echo ""
read -p "🧹 ¿Deseas limpiar archivos de logs y configuración temporal? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f /workspace/.jupyter/jupyter.log ]; then
        rm -f /workspace/.jupyter/jupyter.log
        echo "🗑️  Log de Jupyter eliminado."
    fi
    
    # Limpiar archivos temporales de Jupyter
    find /workspace/.jupyter -name "*.tmp" -delete 2>/dev/null || true
    find /workspace/.jupyter -name "*.lock" -delete 2>/dev/null || true
    
    echo "✨ Limpieza completada."
fi

echo ""
echo "💡 Para reiniciar Jupyter Lab: ./quick-start.sh"
echo "🎉 ¡Jupyter Lab detenido exitosamente!"
