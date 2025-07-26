#!/bin/bash

# Quick Stop Script for Jupyter Lab (Inside Container)
# Este script detiene Jupyter Lab desde dentro del contenedor

set -e  # Salir si ocurre algún error

echo "🛑 Deteniendo Jupyter Lab..."

# Buscar procesos de Jupyter usando archivo PID
if [ -f /workspace/.jupyter/jupyter.pid ]; then
    JUPYTER_PID=$(cat /workspace/.jupyter/jupyter.pid)
    
    if [ ! -z "$JUPYTER_PID" ] && kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "🔧 Deteniendo Jupyter Lab (PID: $JUPYTER_PID)..."
        kill $JUPYTER_PID 2>/dev/null || true
        
        # Esperar un poco para que el proceso termine
        sleep 2
        
        # Verificar si aún está corriendo y forzar si es necesario
        if kill -0 $JUPYTER_PID 2>/dev/null; then
            echo "⚠️  Forzando detención del proceso..."
            kill -9 $JUPYTER_PID 2>/dev/null || true
            sleep 1
        fi
        
        echo "✅ Proceso de Jupyter Lab detenido."
    else
        echo "ℹ️  El proceso PID $JUPYTER_PID ya no está corriendo."
    fi
    
    # Limpiar archivo PID
    rm -f /workspace/.jupyter/jupyter.pid
else
    echo "ℹ️  No hay archivo PID de Jupyter Lab."
fi

# Limpiar archivos de PID adicionales si existen
echo "🧹 Limpiando archivos temporales..."

# Verificar puertos en uso (si netstat está disponible)
if command -v netstat &> /dev/null; then
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
    else
        echo "✅ Puerto 8888 liberado."
    fi
else
    echo "ℹ️  No se puede verificar el estado del puerto (netstat no disponible)."
fi

echo ""
echo "📊 Estado actual:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar procesos Jupyter (si está disponible)
if command -v pgrep &> /dev/null; then
    JUPYTER_PROCESSES=$(pgrep -f jupyter || echo "")
    if [ -z "$JUPYTER_PROCESSES" ]; then
        echo "✅ No hay procesos de Jupyter corriendo"
    else
        echo "⚠️  Procesos de Jupyter aún corriendo: $JUPYTER_PROCESSES"
    fi
else
    echo "ℹ️  No se puede verificar procesos (pgrep no disponible)"
fi

# Verificar archivos de configuración
if [ -d /workspace/.jupyter ]; then
    echo "📁 Archivos en .jupyter:"
    ls -la /workspace/.jupyter/ 2>/dev/null || echo "   (vacío)"
fi

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
