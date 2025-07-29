#!/usr/bin/env python3
"""
Script de configuración para el entorno Python de Practical Statistics for Data Scientists
"""

import subprocess
import sys
import os

def check_requirements():
    """Verificar que todas las dependencias estén instaladas"""
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import scipy
        import sklearn
        import statsmodels
        print("✅ Todas las dependencias principales están disponibles")
        return True
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        return False

def setup_jupyter():
    """Configurar Jupyter Lab con extensiones útiles"""
    extensions = [
        "jupyterlab-plotly",
        "jupyterlab-execute-time"
    ]
    
    for ext in extensions:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", ext], 
                         check=True, capture_output=True)
            print(f"✅ Extensión instalada: {ext}")
        except subprocess.CalledProcessError:
            print(f"⚠️  No se pudo instalar: {ext}")

def main():
    print("🐍 Configurando entorno Python para Practical Statistics...")
    print("=" * 60)
    
    # Verificar dependencias
    if not check_requirements():
        print("\n❌ Algunas dependencias faltan. Ejecuta:")
        print("pip install -r requirements.txt")
        return
    
    # Configurar Jupyter
    print("\n🔧 Configurando Jupyter Lab...")
    setup_jupyter()
    
    print("\n🎉 ¡Configuración completada!")
    print("\nPara iniciar Jupyter Lab:")
    print("jupyter lab --ip=0.0.0.0 --port=8888 --no-browser")
    print("\nO usa Docker:")
    print(".\\docker-helper.ps1 start")

if __name__ == "__main__":
    main()
